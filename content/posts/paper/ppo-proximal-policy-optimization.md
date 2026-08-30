---
title: "PPO：近端策略优化的目标函数与训练流程"
description: "基于 2017 年 PPO 原论文，梳理策略梯度方法、TRPO 的问题、PPO-Clip 的核心公式、训练流程、实验设计与方法局限。"
publishDate: "10 Sep 2025"
tags: ["paper", "llm"]
---

## 引言

论文题目为 *Proximal Policy Optimization Algorithms*。论文提出了一族新的策略梯度算法，核心方法是 PPO-Clip：在复用同一批 on-policy 数据进行多轮 minibatch 更新时，限制新旧策略对已采样动作的概率比不要偏离过远，从而在 **数据利用率、训练稳定性和实现复杂度** 之间取得平衡。

这篇论文研究的不是大语言模型，而是通用强化学习。实验覆盖 MuJoCo 连续控制、Roboschool 人形机器人和 Atari 游戏；后来 PPO 才被广泛用于 RLHF 等语言模型训练场景。

| 项目 | 内容 |
| --- | --- |
| 论文 | *Proximal Policy Optimization Algorithms* |
| 中文理解 | 近端策略优化算法 |
| 作者 | John Schulman、Filip Wolski、Prafulla Dhariwal、Alec Radford、Oleg Klimov |
| 机构 | OpenAI |
| 论文版本 | arXiv:1707.06347v2，28 Aug 2017 |
| 原文 | [arXiv:1707.06347](https://arxiv.org/abs/1707.06347) |
| 参考实现 | [OpenAI Baselines](https://github.com/openai/baselines) |

先用一个最小的强化学习背景理解论文中的对象：

- **Environment（环境）**：根据当前状态 $s_t$ 接收动作 $a_t$，返回奖励 $R_t$ 和下一个状态
- **Policy（策略）**：$\pi_\theta(a_t\mid s_t)$ 是模型选择动作的概率分布，$\theta$ 是策略网络参数
- **Value function（价值函数）**：$V(s_t)$ 估计从状态 $s_t$ 出发未来能得到多少回报
- **Advantage（优势）**：$\hat A_t$ 衡量“在这个状态下选择当前动作，比通常水平好多少或差多少”

普通策略梯度通常使用下面的梯度估计：

$$
\hat g=\hat{\mathbb E}_t\left[
\nabla_\theta\log\pi_\theta(a_t\mid s_t)\hat A_t
\right]
$$

它的直觉是：如果一个动作带来了正优势，就提高它的概率；如果优势为负，就降低它的概率。问题在于，一批数据只更新一次会浪费样本，而在同一批旧数据上无约束地更新很多次，又可能让新策略突然偏离旧策略。

:::important[PPO 一句话]
PPO 用一个带 clipping 的 surrogate objective（替代目标函数）控制策略更新：允许有益的小幅变化，但不再奖励有益方向上的过大变化。
:::

## 现有方法做法和不足

论文试图解决的不是“策略梯度能不能工作”，而是 **如何让策略梯度既能复用数据，又不会因为更新过大而失控**。论文讨论的主要方法如下：

| 方法 | 基本做法 | 论文指出的不足 |
| --- | --- | --- |
| Deep Q-learning | 学习动作价值函数，再选择价值更高的动作 | 带函数逼近的 Q-learning 在许多问题上表现不稳定；论文特别指出它尚未在连续控制基准上得到充分验证 |
| Vanilla policy gradient | 根据 $\nabla_\theta\log\pi_\theta(a_t\mid s_t)\hat A_t$ 更新策略 | 数据效率和鲁棒性较差；同一轨迹上重复优化普通目标，容易造成过大的策略变化 |
| TRPO | 最大化重要性采样后的 surrogate，同时约束新旧策略的 KL 散度 | 需要对目标和约束做近似，并使用共轭梯度求解；实现复杂，也不容易适配 dropout、参数共享和辅助任务 |
| 固定 KL penalty | 在 surrogate objective 中加入 $-\beta D_{\mathrm{KL}}$ 惩罚 | 不同任务、不同训练阶段需要不同的 $\beta$，很难找到一个始终合适的固定值 |
| 无约束地重复优化 $L^{PG}$ | 在同一批数据上做多个 epoch 的普通梯度更新 | 旧数据来自旧策略，策略变化后仍把它当成当前策略数据使用，容易产生 destructive policy update |

TRPO 的基本思想是把新策略限制在旧策略附近。它优化一个重要性采样目标，同时要求平均 KL 散度不超过阈值：

$$
\max_\theta\ \hat{\mathbb E}_t
\left[
\frac{\pi_\theta(a_t\mid s_t)}
{\pi_{\theta_{\mathrm{old}}}(a_t\mid s_t)}
\hat A_t
\right]
$$

$$
\text{subject to}\quad
\hat{\mathbb E}_t\left[
D_{\mathrm{KL}}\left(
\pi_{\theta_{\mathrm{old}}}(\cdot\mid s_t),
\pi_\theta(\cdot\mid s_t)
\right)
\right]\leq\delta
$$

TRPO 的方向是对的，但它把“不要走太远”变成了一个带约束的近似二阶优化问题。PPO 的问题意识是：能否把这个约束直接写进目标函数，让普通的一阶 SGD 或 Adam 也能得到类似的近端更新效果？

## 研究的 Motivation–Insight–Contributions

| 维度 | 论文中的核心内容 |
| --- | --- |
| Motivation | 作者希望得到一种可扩展（适用于大模型和并行实现）、数据高效且鲁棒的策略优化方法，并在只使用一阶优化的前提下达到 TRPO 的数据效率和可靠性能 |
| Insight 1 | 将训练组织为“从当前策略采样数据，再对 surrogate objective 做多轮 minibatch 优化”的交替过程 |
| Insight 2 | 用 clipped probability ratio 构造 surrogate objective：当 ratio 超出 $[1-\epsilon,1+\epsilon]$ 且继续朝有利方向变化时，去除继续变化带来的目标收益；取 clipped 与 unclipped 项的最小值后得到 pessimistic lower bound |
| Insight 3 | 作者指出，在同一 trajectory 上重复优化普通 policy-gradient loss 理论依据不足，且常导致 destructively large policy updates；clipped objective 用来处理这一问题 |
| Contributions | 提出 PPO 这一族策略优化方法与 clipped probability-ratio objective；比较不同 surrogate objective 和已有算法，并在连续控制与 Atari 任务上验证其性能、样本复杂度、实现复杂度和 wall-time |

论文将这种方法称为 proximal policy optimization，核心是通过 clipped probability ratios 构造 pessimistic estimate，并在固定采样数据上进行多轮优化。

## 核心创新点

1. **用概率比表达新旧策略的差异**

   对由旧策略 $\pi_{\theta_{\mathrm{old}}}$ 采样得到的状态动作对 $(s_t,a_t)$，定义：

   $$
   r_t(\theta)=
   \frac{\pi_\theta(a_t\mid s_t)}
   {\pi_{\theta_{\mathrm{old}}}(a_t\mid s_t)}
   $$

   当 $\theta=\theta_{\mathrm{old}}$ 时，$r_t(\theta)=1$。因此：

   - $r_t>1$：新策略提高了这个动作的概率
   - $r_t<1$：新策略降低了这个动作的概率
   - $|r_t-1|$ 越大：相对旧策略的局部变化越大

   实际实现中通常保存旧策略的 log-probability，再用

   $$
   r_t(\theta)=
   \exp\left(
   \log\pi_\theta(a_t\mid s_t)
   -
   \log\pi_{\theta_{\mathrm{old}}}(a_t\mid s_t)
   \right)
   $$

   计算概率比，避免直接操作很小的概率数值。

2. **从普通 surrogate objective 得到 PPO-Clip**

   未加约束的 surrogate objective 可以写成：

   $$
   L^{\mathrm{CPI}}(\theta)
   =
   \hat{\mathbb E}_t
   \left[
   r_t(\theta)\hat A_t
   \right]
   $$

   如果直接最大化它，当 $\hat A_t>0$ 时，优化器会持续提高该动作的概率；当 $\hat A_t<0$ 时，优化器会持续降低该动作的概率。PPO-Clip 把它改为：

   $$
   L^{\mathrm{CLIP}}(\theta)
   =
   \hat{\mathbb E}_t
   \left[
   \min\left(
   r_t(\theta)\hat A_t,\,
   \operatorname{clip}
   \left(
   r_t(\theta),1-\epsilon,1+\epsilon
   \right)\hat A_t
   \right)
   \right]
   $$

   论文举例使用 $\epsilon=0.2$，即把概率比限制在 $[0.8,1.2]$ 的区间内。这里最关键的不是“把所有 ratio 都硬截断”，而是 **取未截断项和截断项中的较小值**。

3. **为什么要根据 advantage 的正负采用不同方向的保护**

   | 情况 | 优化器原本想做什么 | PPO-Clip 的保护边界 |
   | --- | --- | --- |
   | $\hat A_t>0$ | 提高好动作的概率，即让 $r_t$ 变大 | $r_t>1+\epsilon$ 后不再奖励继续提高 |
   | $\hat A_t<0$ | 降低坏动作的概率，即让 $r_t$ 变小 | $r_t<1-\epsilon$ 后不再奖励继续降低 |
   | 朝错误方向变化 | 修正一个本来就不好的更新 | 保留未截断项，让目标继续反映变差 |

   例如 $\epsilon=0.2$：

   | $\hat A_t$ | $r_t$ | 未截断项 $r_t\hat A_t$ | 截断项 | 最终取值 |
   | ---: | ---: | ---: | ---: | ---: |
   | $+2$ | $1.4$ | $2.8$ | $1.2\times2=2.4$ | $2.4$ |
   | $-2$ | $0.6$ | $-1.2$ | $0.8\times(-2)=-1.6$ | $-1.6$ |

   第一行表示：这个动作是好动作，概率已经从旧策略的 1 倍增加到 1.4 倍；PPO 不再让这次过大的增加继续带来目标收益。第二行表示：这个动作是坏动作，概率已经降得过多；取更小的 $-1.6$，阻止优化器因为这次过度降低而继续获益。

   :::note[为什么取 min]
   取 min 会形成相对于未截断 surrogate 的保守估计：当过大的变化会让目标看起来更好时，使用截断后的较小收益；当变化本身让目标变差时，仍保留变差的信号。这样，clipping 主要抑制“有利方向上的过度更新”，而不是把所有梯度都抹掉。
   :::

4. **用一阶优化替代复杂的 trust-region 求解**

   PPO 只需要在固定的采样数据上构造 $L^{\mathrm{CLIP}}$，然后用 SGD 或 Adam 做多轮 minibatch 优化，不需要 TRPO 中的共轭梯度和显式二阶近似。这使它更容易放进常见的 actor-critic、并行采样和参数共享架构中。

5. **提供 adaptive KL penalty 作为另一种 PPO 变体**

   论文还研究了 KL 惩罚版本：

   $$
   L^{\mathrm{KLPEN}}(\theta)
   =
   \hat{\mathbb E}_t
   \left[
   r_t(\theta)\hat A_t
   -
   \beta
   D_{\mathrm{KL}}\left(
   \pi_{\theta_{\mathrm{old}}}(\cdot\mid s_t),
   \pi_\theta(\cdot\mid s_t)
   \right)
   \right]
   $$

   每轮更新后计算实际 KL 值 $d$：

   - 如果 $d<d_{\mathrm{targ}}/1.5$，令 $\beta\leftarrow\beta/2$
   - 如果 $d>d_{\mathrm{targ}}\times1.5$，令 $\beta\leftarrow2\beta$

   论文实验发现，KL penalty 在这些基准任务上通常不如 clipped surrogate，但它帮助说明了 PPO 的共同目标：控制新旧策略之间的漂移。

## 方法框架

先把 PPO 的一轮训练压缩成一条数据流：

~~~text
旧策略 πθold 与环境交互
        ↓
收集 (state, action, reward, old log-probability, value)
        ↓
用 value function 和 GAE 估计 advantage
        ↓
当前策略计算 new log-probability
        ↓
ratio = exp(new log-probability - old log-probability)
        ↓
PPO-Clip objective + value loss + entropy bonus
        ↓
固定本批数据做 K 个 epoch 的 minibatch 更新
        ↓
θold ← θ，重新与环境交互
~~~

1. **采样：先固定一批来自旧策略的数据**

   在第 $k$ 轮更新中，使用 $\pi_{\theta_{\mathrm{old}}}$ 与环境交互。论文的 actor-critic 形式使用 $N$ 个并行 actor，每个 actor 运行 $T$ 个 timestep，因此一轮得到约 $NT$ 条样本。

   每条样本至少需要保存：

   - 状态 $s_t$
   - 动作 $a_t$
   - 奖励 $R_t$
   - 旧策略对该动作的 log-probability
   - value function 的估计值

   **旧策略的 log-probability 必须固定。** 如果在 minibatch 更新过程中也不断更新分母，ratio 就不再表示“当前策略相对于采样策略改变了多少”。

2. **用 GAE 估计 advantage**

   论文使用截断的 generalized advantage estimation。先计算 temporal-difference residual：

   $$
   \delta_t=R_t+\gamma V(s_{t+1})-V(s_t)
   $$

   再把未来多个 residual 按 $\gamma\lambda$ 衰减并累加：

   $$
   \hat A_t=
   \delta_t+
   (\gamma\lambda)\delta_{t+1}+
   (\gamma\lambda)^2\delta_{t+2}
   +\cdots
   $$

   $\gamma$ 控制未来奖励的折扣，$\lambda$ 控制 advantage 估计在 bias 和 variance 之间的折中。直观地说，$\hat A_t$ 是 PPO 判断“这次动作应该被鼓励还是抑制”的依据。

3. **计算 ratio 并应用 clipping**

   当前策略 $\pi_\theta$ 对同一个已采样动作重新计算 log-probability，然后得到：

   $$
   r_t(\theta)=
   \exp\left(
   \log\pi_\theta(a_t\mid s_t)
   -
   \log\pi_{\theta_{\mathrm{old}}}(a_t\mid s_t)
   \right)
   $$

   这个 ratio 与 $\hat A_t$ 一起进入 $L^{\mathrm{CLIP}}$。注意，PPO 并不是把整个新策略的所有动作概率都裁掉，而是对 **当前批次中实际采样到的动作** 计算局部约束。

4. **联合优化 policy、value 和 exploration**

   如果 policy 和 value function 共享网络参数，论文把三部分合成一个目标：

   $$
   L^{\mathrm{CLIP+VF+S}}(\theta)
   =
   \hat{\mathbb E}_t
   \left[
   L_t^{\mathrm{CLIP}}(\theta)
   -
   c_1L_t^{\mathrm{VF}}(\theta)
   +
   c_2S[\pi_\theta](s_t)
   \right]
   $$

   其中：

   - $L_t^{\mathrm{CLIP}}$：更新策略的目标
   - $L_t^{\mathrm{VF}}=(V_\theta(s_t)-V_t^{\mathrm{target}})^2$：训练 value function 的平方误差
   - $S[\pi_\theta]$：策略熵，鼓励探索，避免过早只选择少数动作
   - $c_1,c_2$：value loss 和 entropy bonus 的权重

5. **在固定数据上做有限轮 minibatch 更新**

   将 $NT$ 条样本切成 minibatch，在同一批数据上优化 $K$ 个 epoch。每次 minibatch 更新都会重新计算当前策略的 ratio，但分母始终来自固定的 old policy。clipping 负责让这些多轮更新保持在相对保守的范围内。

6. **完成一轮后更新 old policy**

   当 $K$ 个 epoch 完成后，将当前参数设为下一轮的旧策略：

   $$
   \theta_{\mathrm{old}}\leftarrow\theta
   $$

   下一轮重新与环境交互，得到来自新策略的数据。PPO 的“数据复用”只发生在当前 old policy 采集的这一个 batch 内，不是无限期重复使用历史数据。

7. **把整个过程写成伪代码**

   ~~~text
   初始化策略参数 θ 和旧策略参数 θold

   for iteration = 1, 2, ...:
       使用 πθold 与环境交互，收集 N 个 actor 的 T 步数据
       计算 value target 和 advantage Ahat

       for epoch = 1, ..., K:
           将 NT 条样本划分为 minibatch
           计算 new log-probability 和 ratio
           计算 clipped policy loss、value loss、entropy bonus
           用 SGD/Adam 更新 θ

       θold ← θ
   ~~~

## 实验设计

论文实验围绕三个问题展开：PPO-Clip 是否优于不加保护的目标和 KL penalty？它是否能在连续控制中击败已有方法？它在高维人形控制和 Atari 离散动作环境中是否仍然有效？

| 实验 | 环境与比较对象 | 主要设置 | 评价重点 |
| --- | --- | --- | --- |
| Surrogate objective 对比 | 7 个 MuJoCo 任务；比较无 clipping/penalty、不同 $\epsilon$ 的 clipping、fixed/adaptive KL | 每个环境训练 1M timesteps；每个算法设置 3 个随机种子，共 21 次运行 | 平均归一化得分，观察目标函数和超参数的影响 |
| 连续控制算法比较 | PPO-Clip、TRPO、CEM、adaptive-step vanilla PG、A2C、A2C + trust region | 7 个 MuJoCo 环境，每个训练 1M timesteps | 学习曲线和最终控制性能 |
| 高维人形控制 | RoboschoolHumanoid、RoboschoolHumanoidFlagrun、RoboschoolHumanoidFlagrunHarder | 3D 人形机器人，需要奔跑、转向、追踪随机目标，并在部分任务中被方块攻击后恢复 | 能否处理高维连续动作和环境扰动 |
| Atari 比较 | 49 个 Atari 游戏；PPO、A2C、ACER 使用相同 policy network architecture | 三个随机试验；比较全程平均奖励和最后 100 episodes 平均奖励 | 分别观察学习速度和最终性能 |

**Surrogate objective 对比**

MuJoCo 实验把每个环境中的随机策略得分归一化为 0，把最佳结果归一化为 1，再对 7 个环境和 3 个随机种子取平均。结果如下：

| 目标函数或超参数 | 平均归一化得分 |
| --- | ---: |
| 无 clipping 或 penalty | -0.39 |
| Clipping，$\epsilon=0.1$ | 0.76 |
| Clipping，$\epsilon=0.2$ | **0.82** |
| Clipping，$\epsilon=0.3$ | 0.70 |
| Adaptive KL，$d_{\mathrm{targ}}=0.003$ | 0.68 |
| Adaptive KL，$d_{\mathrm{targ}}=0.01$ | 0.74 |
| Adaptive KL，$d_{\mathrm{targ}}=0.03$ | 0.71 |
| Fixed KL，$\beta=0.3$ | 0.62 |
| Fixed KL，$\beta=1$ | 0.71 |
| Fixed KL，$\beta=3$ | 0.72 |
| Fixed KL，$\beta=10$ | 0.69 |

无 clipping/penalty 的得分为负，主要因为 HalfCheetah 环境中的一次严重退化把平均结果拉低。这个实验支持论文的核心判断：**在相同的多轮更新设置下，PPO-Clip 比直接重复优化和 KL penalty 变体更稳定；在这组实验中 $\epsilon=0.2$ 最好。**

**连续控制和人形控制**

在 7 个 MuJoCo 环境中，PPO-Clip 几乎在所有环境上都优于论文选择的比较方法。比较对象包括 tuned TRPO、cross-entropy method、带自适应步长的 vanilla policy gradient、A2C 和 A2C + trust region。

在人形控制实验中，论文测试了三个逐渐困难的任务：

- RoboschoolHumanoid：只要求机器人向前运动
- RoboschoolHumanoidFlagrun：目标位置每隔一段时间随机变化，机器人需要转向并追踪新目标
- RoboschoolHumanoidFlagrunHarder：机器人还会被方块攻击，需要从地面恢复并继续行动

这些实验的作用不是证明 PPO 在所有机器人任务上都最优，而是展示它能够处理 **高维连续动作、目标变化和环境扰动**。

**Atari 结果**

Atari 实验使用 49 个游戏，并用三个随机试验的平均结果判断每个算法“赢得”的游戏数量：

| 评价指标 | A2C | ACER | PPO | Tie |
| --- | ---: | ---: | ---: | ---: |
| 整个训练期间的平均 episode reward | 1 | 18 | **30** | 0 |
| 最后 100 episodes 的平均 episode reward | 1 | **28** | 19 | 1 |

这两个指标揭示了一个重要区别：

- **全程平均奖励**：更看重学习速度，PPO 赢得 30 个游戏
- **最后 100 episodes 奖励**：更看重最终性能，ACER 赢得 28 个游戏，PPO 赢得 19 个

因此，论文的结论不是“PPO 在所有指标上都绝对最好”，而是它在样本复杂度、实现简单性和运行时间之间取得了有吸引力的综合平衡。

**论文使用的关键超参数**

| 实验 | 关键设置 |
| --- | --- |
| MuJoCo | $T=2048$，Adam stepsize $3\times10^{-4}$，10 epochs，minibatch 64，$\gamma=0.99$，$\lambda=0.95$ |
| Roboschool | $T=512$，15 epochs，minibatch 4096，$\gamma=0.99$，$\lambda=0.95$，actor 数量为 locomotion 32、flagrun 128 |
| Atari | $T=128$，Adam stepsize $2.5\times10^{-4}\alpha$，3 epochs，minibatch $32\times8$，8 个 actors，$\epsilon=0.1\alpha$，$c_1=1$，$c_2=0.01$ |

Atari 中的 $\alpha$ 会在训练过程中从 1 线性退火到 0，因此 clipping 区间和步长会随训练逐步收缩。

## Limitation

- **在同一 trajectory 上重复优化普通目标缺少充分依据。** 作者指出，对普通 policy-gradient loss 做多次优化在理论上并不充分，且实验证明经常会导致 destructively large policy updates；PPO 的 clipped objective 正是为处理这一问题提出的。

- **固定 KL penalty 的系数难以选取。** 作者指出，很难为不同问题，甚至同一问题不同学习阶段，选择一个始终合适的固定 $\beta$；论文实验中 KL penalty 的表现也低于 clipped surrogate objective。
