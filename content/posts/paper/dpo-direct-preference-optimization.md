---
title: "DPO：Direct Preference Optimization"
description: "基于 DPO 原论文，梳理 RLHF 的奖励建模流程、DPO 的 reward-policy 变换、偏好损失、理论依据、实验设计与方法边界。"
publishDate: "20 Sep 2025"
tags: ["paper", "llm"]
---

## 引言

*Direct Preference Optimization: Your Language Model is Secretly a Reward Model* 研究如何直接利用人类偏好数据训练语言模型。论文指出，大规模无监督语言模型虽然获得了广泛知识和部分推理能力，但很难精确控制其行为；传统 RLHF 通常需要先训练奖励模型，再通过强化学习让语言模型最大化奖励，同时不能偏离原始模型太远。

DPO 的核心做法是利用“奖励函数”和“KL 约束下的最优策略”之间的解析映射，把原本的奖励建模与强化学习流程改写成一个直接作用于策略的二分类损失。这样，模型可以直接从偏好数据中学习，而不需要显式训练奖励模型，也不需要在微调过程中持续从语言模型采样并执行强化学习。

论文在受控情感生成、摘要和单轮对话三个任务上进行实验。作者报告称，在几乎不调超参数的情况下，DPO 的效果与 PPO-based RLHF 相当或更好；在摘要和对话实验中，DPO 还取得了更高的偏好胜率或更好的效率。

:::important[DPO 一句话]
DPO 保留了 RLHF 中“提高偏好奖励、限制策略偏离参考模型”的目标，但通过 reward-policy 的变换，把它变成了直接对偏好样本进行二分类训练的损失
:::

| 项目 | 内容 |
| --- | --- |
| 论文 | *Direct Preference Optimization: Your Language Model is Secretly a Reward Model* |
| 中文理解 | 直接偏好优化：你的语言模型其实是一个隐式奖励模型 |
| 作者 | Rafael Rafailov、Archit Sharma、Eric Mitchell、Stefano Ermon、Christopher D. Manning、Chelsea Finn |
| 机构 | Stanford University、CZ Biohub |
| 会议 | NeurIPS 2023，第 37 届 Neural Information Processing Systems Conference |
| 原文 | [arXiv:2305.18290](https://arxiv.org/abs/2305.18290) |

## 现有方法做法和不足

论文将从偏好中训练语言模型的常见流程概括为三个阶段：先进行 supervised fine-tuning（SFT），再收集偏好并训练奖励模型，最后使用强化学习优化语言模型策略。

| 方法 | 基本做法 | 论文指出的问题 |
| --- | --- | --- |
| SFT / instruction tuning | 使用高质量的人类示范，通过监督学习得到下游任务模型 | 相比专家示范，人类往往更容易标注两个回答哪个更好；只依赖示范会错过这种相对偏好信息 |
| Reward modeling | 对同一个 prompt 的两个回答收集偏好，用 Bradley-Terry 模型训练奖励函数 $r_\phi(x,y)$ | 需要额外训练一个奖励模型；奖励模型只是对偏好进行估计，随后还要再进行策略优化 |
| RLHF + PPO | 使用奖励模型给语言模型反馈，再用 PPO 最大化奖励并约束策略不要偏离参考模型 | 流程复杂且可能不稳定，需要训练多个模型、在训练循环中采样语言模型，并承担较高的计算成本 |
| REINFORCE / actor-critic | 将离散文本生成转成策略梯度问题，用强化学习优化奖励 | 语言生成目标不可直接微分，训练需要策略采样；奖励、KL 约束和价值估计共同增加了实现与调参负担 |
| Preference-based RL | 从二元偏好中估计潜在评分函数，再优化得到策略 | 现有方法通常仍然先显式估计潜在评分函数，也就是奖励模型，再优化策略 |

**RLHF 的三阶段流程**

**1. SFT**

RLHF 通常先用高质量任务数据对预训练语言模型进行监督微调，得到 $\pi^{\mathrm{SFT}}$。这一步可以让模型具备对话、摘要等下游任务的基本行为。

**2. 偏好收集与奖励建模**

给定 prompt $x$，从 SFT 模型生成两个回答 $y_1$ 和 $y_2$，再由标注者选出更偏好的回答。记偏好回答为 $y_w$，不偏好回答为 $y_l$，形成离线数据集：

$$
\mathcal D=
\left\{
\left(x^{(i)},y_w^{(i)},y_l^{(i)}\right)
\right\}_{i=1}^{N}
$$

论文使用 Bradley-Terry 偏好模型描述潜在真实奖励 $r^*(x,y)$ 产生偏好的方式：

$$
p^*(y_1\succ y_2\mid x)
=
\frac{\exp\left(r^*(x,y_1)\right)}
{\exp\left(r^*(x,y_1)\right)+\exp\left(r^*(x,y_2)\right)}
$$

奖励模型 $r_\phi(x,y)$ 通过最大似然，也就是一个二分类负对数似然进行训练：

$$
\mathcal L_R(r_\phi,\mathcal D)
=
-\mathbb E_{(x,y_w,y_l)\sim\mathcal D}
\left[
\log\sigma\left(
r_\phi(x,y_w)-r_\phi(x,y_l)
\right)
\right]
$$

在语言模型中，奖励网络通常以 SFT 模型为初始化，并在 Transformer 最后一层上增加一个线性层输出一个标量奖励。论文还提到，已有方法会对奖励进行归一化，以降低方差。

**3. RL 微调**

设 $\pi_{\mathrm{ref}}$ 为参考策略，通常是初始 SFT 模型；$\pi_\theta$ 为正在训练的语言模型策略。RLHF 的目标是在提高奖励的同时限制当前策略偏离参考策略：

$$
\max_{\pi_\theta}
\mathbb E_{x\sim\mathcal D,\;y\sim\pi_\theta(y\mid x)}
\left[
r_\phi(x,y)
\right]
-\beta D_{\mathrm{KL}}
\left(
\pi_\theta(y\mid x)
\middle\|
\pi_{\mathrm{ref}}(y\mid x)
\right)
$$

其中 $\beta$ 控制偏离参考策略的程度。这个约束一方面让奖励模型主要在它熟悉的分布上工作，另一方面有助于保持生成多样性，避免模型坍缩到少数高奖励回答。由于文本生成是离散过程，上述目标通常使用强化学习优化；论文描述的标准做法是构造带 KL 惩罚的奖励，再用 PPO 训练。

## 研究的 Motivation–Insight–Contributions

| 维度 | 论文中的核心内容 |
| --- | --- |
| Motivation | 作者指出，现有从偏好中训练语言模型的方法通常需要先学习显式奖励函数，再用强化学习优化策略，流程复杂且计算成本高；DPO 旨在直接让语言模型满足人类偏好 |
| Insight 1 | KL 约束奖励最大化问题的最优策略具有解析形式，这建立了语言模型策略与奖励函数之间的映射 |
| Insight 2 | 在 Bradley-Terry 偏好模型下，偏好只依赖两个回答的奖励差；完成重参数化后，只依赖 prompt 的 partition function 项会相互抵消 |
| Insight 3 | 将偏好损失改写为策略的函数后，可以使用简单的 binary cross-entropy 直接训练策略，而不显式学习奖励函数，也不在训练过程中从策略采样 |
| Insight 4 | DPO 的更新提高偏好回答相对于不偏好回答的 log probability，并使用 dynamic、per-example importance weight，避免朴素 probability-ratio objective 导致的模型退化 |
| Contributions | 提出 Direct Preference Optimization 这一无需强化学习的偏好训练算法；给出策略与奖励之间的映射及理论分析；在情感控制、摘要和对话任务上与 PPO-based RLHF 等方法比较，实验模型规模最高达到 6B |

## 核心创新点

1. **把 KL 约束的奖励优化问题改写成策略问题**

   对任意奖励函数 $r(x,y)$，论文给出 KL 约束奖励最大化问题的最优策略：

   $$
   \pi_r(y\mid x)
   =
   \frac{1}{Z(x)}
   \pi_{\mathrm{ref}}(y\mid x)
   \exp\left(
   \frac{1}{\beta}r(x,y)
   \right)
   $$

   其中 $Z(x)$ 是保证策略归一化的 partition function：

   $$
   Z(x)
   =
   \sum_y
   \pi_{\mathrm{ref}}(y\mid x)
   \exp\left(
   \frac{1}{\beta}r(x,y)
   \right)
   $$

   将上式取对数并整理，可以得到：

   $$
   r(x,y)
   =
   \beta\log
   \frac{\pi_r(y\mid x)}
   {\pi_{\mathrm{ref}}(y\mid x)}
   +
   \beta\log Z(x)
   $$

   这一步是 DPO 的数学入口：奖励不再必须由单独的奖励网络表示，也可以由一个策略与参考策略的概率比表示。

2. **利用偏好差值消掉 partition function**

   Bradley-Terry 模型只关心两个回答之间的奖励差：

   $$
   p^*(y_1\succ y_2\mid x)
   =
   \sigma\left(
   r^*(x,y_1)-r^*(x,y_2)
   \right)
   $$

   将上面的 reward-policy 关系分别代入 $y_1$ 和 $y_2$ 后，两个回答共享的 $\beta\log Z(x)$ 会抵消。因此，最优策略的偏好概率可以只用 $\pi^*$ 和 $\pi_{\mathrm{ref}}$ 表示：

   $$
   p^*(y_1\succ y_2\mid x)
   =
   \sigma\left(
   \beta\log
   \frac{\pi^*(y_1\mid x)}
   {\pi_{\mathrm{ref}}(y_1\mid x)}
   -
   \beta\log
   \frac{\pi^*(y_2\mid x)}
   {\pi_{\mathrm{ref}}(y_2\mid x)}
   \right)
   $$

   因为不需要计算 $Z(x)$，这个表达式才可以直接用于实际训练。

3. **直接得到 DPO 偏好损失**

   用参数化策略 $\pi_\theta$ 代替未知的最优策略 $\pi^*$，对偏好数据最大化似然，得到 DPO 损失：

   $$
   \mathcal L_{\mathrm{DPO}}
   \left(
   \pi_\theta;\pi_{\mathrm{ref}}
   \right)
   =
   -\mathbb E_{(x,y_w,y_l)\sim\mathcal D}
   \left[
   \log\sigma
   \left(
   \beta\log
   \frac{\pi_\theta(y_w\mid x)}
   {\pi_{\mathrm{ref}}(y_w\mid x)}
   -
   \beta\log
   \frac{\pi_\theta(y_l\mid x)}
   {\pi_{\mathrm{ref}}(y_l\mid x)}
   \right)
   \right]
   $$

   一个样本的训练信号可以理解为：如果当前策略相对于参考策略更倾向于 $y_w$，并且更不倾向于 $y_l$，括号中的值就更大，损失更小；反之，损失会推动模型提高偏好回答的相对概率。

4. **策略同时表示语言模型和隐式奖励**

   DPO 定义了由策略隐式产生的奖励：

   $$
   \hat r_\theta(x,y)
   =
   \beta\log
   \frac{\pi_\theta(y\mid x)}
   {\pi_{\mathrm{ref}}(y\mid x)}
   $$

   这个 $\hat r_\theta$ 不是额外训练的 reward head，而是由当前策略和固定参考策略的概率比得到的隐式奖励。因此，标题中的“Secretly a Reward Model”指的是：策略的概率比可以扮演奖励模型在偏好排序中的作用。

5. **用动态权重处理当前排序错误的样本**

   论文对 DPO 损失求梯度，得到：

   $$
   \nabla_\theta\mathcal L_{\mathrm{DPO}}
   =
   -\beta\mathbb E_{\mathcal D}
   \left[
   \sigma\left(
   \hat r_\theta(x,y_l)-\hat r_\theta(x,y_w)
   \right)
   \left(
   \nabla_\theta\log\pi_\theta(y_w\mid x)
   -
   \nabla_\theta\log\pi_\theta(y_l\mid x)
   \right)
   \right]
   $$

   梯度中的第一项是动态权重。如果当前隐式奖励把不偏好回答 $y_l$ 排在偏好回答 $y_w$ 前面，$\hat r_\theta(x,y_l)-\hat r_\theta(x,y_w)$ 较大，这个样本的权重就较高；第二项则提高 $y_w$ 的似然并降低 $y_l$ 的似然。论文还指出，去掉这一权重的朴素版本可能导致语言模型退化。

6. **理论上不缩小可表示的奖励类别**

   论文定义了奖励函数的等价关系：如果两个奖励函数只相差一个只依赖 prompt 的函数 $f(x)$，则它们属于同一个等价类：

   $$
   r(x,y)-r'(x,y)=f(x)
   $$

   在 Plackett-Luce，特别是 Bradley-Terry 偏好框架下，同一等价类的奖励函数会产生相同的偏好分布；在 KL 约束的 RL 问题下，它们也会产生相同的最优策略。论文进一步证明，在温和假设下，所有与这些偏好模型一致的奖励等价类，都可以表示成：

   $$
   r(x,y)
   =
   \beta\log
   \frac{\pi(y\mid x)}
   {\pi_{\mathrm{ref}}(y\mid x)}
   $$

   因而，DPO 的重参数化并没有因为不显式训练奖励网络而损失可表示的奖励类别，并且可以恢复对应的最优策略。

## 方法框架

**1. 先统一符号**

| 符号 | 含义 |
| --- | --- |
| $x$ | prompt、用户查询或输入上下文 |
| $y_w$ | 人类偏好的回答，winner |
| $y_l$ | 人类不偏好的回答，loser |
| $\mathcal D$ | 偏好数据集 $\{(x,y_w,y_l)\}$ |
| $\pi_{\mathrm{ref}}$ | 固定的参考策略，通常为 SFT 模型 |
| $\pi_\theta$ | 正在训练的语言模型策略 |
| $r_\phi$ | RLHF 中显式训练的奖励模型 |
| $\hat r_\theta$ | DPO 中由策略概率比隐式定义的奖励 |
| $\beta$ | 控制策略偏离参考策略程度的参数 |

在这些公式里，$\pi_\theta(y\mid x)$ 表示模型在输入 $x$ 下生成完整回答 $y$ 的概率。对自回归语言模型，可以把它理解为每个 token 条件概率的乘积；在对数空间中，就是这些 token 对数概率的累加。

**2. RLHF 和 DPO 的路径差异**

两条路径可以概括为：

| 路径 | 训练步骤 | 训练时是否需要显式奖励模型 | 训练时是否需要 RL 采样 |
| --- | --- | --- | --- |
| RLHF | SFT → 偏好数据 → 训练 $r_\phi$ → PPO 优化 $\pi_\theta$ | 需要 | 需要 |
| DPO | 准备偏好数据 → 固定 $\pi_{\mathrm{ref}}$ → 直接最小化 $\mathcal L_{\mathrm{DPO}}$ | 不需要，使用 $\hat r_\theta$ | 不需要在线采样 |

DPO 并不是去掉参考模型。参考模型仍然出现在两个概率比的分母中，用来定义当前策略相对于原始行为的变化，并保留 RLHF 中的 KL 约束含义。

**3. 从 RLHF 目标推导 DPO**

推导的逻辑顺序可以压缩成四步：

1. 从 KL 约束奖励最大化目标出发

   $$\pi_r(y\mid x)=\frac{1}{Z(x)}\pi_{\mathrm{ref}}(y\mid x)\exp\left(\frac{1}{\beta}r(x,y)\right)$$

2. 用最优策略和参考策略的概率比重写奖励

   $$r(x,y)=\beta\log\frac{\pi_r(y\mid x)}{\pi_{\mathrm{ref}}(y\mid x)}+\beta\log Z(x)$$

3. 将这个表达式代入 Bradley-Terry 的奖励差

   $$r(x,y_w)-r(x,y_l)$$

   因为两个回答属于同一个 prompt，$\beta\log Z(x)$ 会抵消

4. 用参数化策略 $\pi_\theta$ 代替未知 $\pi^*$，对偏好数据做最大似然训练，得到 DPO loss

   $$\mathcal L_{\mathrm{DPO}}=-\mathbb E_{\mathcal D}\left[\log\sigma\left(\beta\log\frac{\pi_\theta(y_w\mid x)}{\pi_{\mathrm{ref}}(y_w\mid x)}-\beta\log\frac{\pi_\theta(y_l\mid x)}{\pi_{\mathrm{ref}}(y_l\mid x)}\right)\right]$$

:::note[如何理解一个训练样本]
DPO 不直接问模型“这个回答的绝对奖励是多少”，而是比较两个回答相对于参考模型的概率变化。如果当前策略给偏好回答带来的相对提升高于不偏好回答，模型更符合这个偏好样本；否则，损失会推动两者的相对概率差扩大
:::

**4. 实际训练流程**

论文给出的 DPO outline 可以整理为：

1. 对每个 prompt 准备两个回答，并获得偏好标签，形成 $\mathcal D$。
2. 如果偏好数据是由 SFT 模型采样得到的，就令 $\pi_{\mathrm{ref}}=\pi_{\mathrm{SFT}}$。
3. 如果没有可用的 SFT 模型，就先只在偏好回答 $(x,y_w)$ 上做最大似然训练，构造参考策略：

   $$\pi_{\mathrm{ref}}=\mathop{\arg\max}_{\pi}\mathbb E_{x,y_w\sim\mathcal D}\left[\log\pi(y_w\mid x)\right]$$

4. 用当前策略 $\pi_\theta$ 和固定参考策略 $\pi_{\mathrm{ref}}$ 分别计算 $y_w$、$y_l$ 的序列概率。
5. 计算两个相对概率的差，并代入 $\mathcal L_{\mathrm{DPO}}$。
6. 通过普通的最大似然式梯度下降更新 $\pi_\theta$，参考策略保持不变。
7. 在整个偏好数据集上重复 minibatch 训练，直到达到设定的训练轮数或收敛条件。

训练时不需要从当前策略重新生成回答，也不需要把生成结果送入奖励模型再用 PPO 更新。偏好数据仍然可以来自已有的公开数据集；DPO 的关键是直接复用这些离线偏好对。

## 实验设计

**实验问题**

论文围绕四个问题组织实验：

1. 在已知真实奖励的受控环境中，DPO 能否更高效地平衡奖励和 KL？
2. DPO 能否扩展到真实的人类偏好数据和更大的语言模型任务？
3. 从 Reddit 帖子迁移到新闻文章时，DPO 能否保持泛化能力？
4. GPT-4 作为自动评估器的判断，是否与人类判断具有相近的一致性？

**任务与数据**

| 任务 | 输入与目标 | 偏好数据和模型设置 |
| --- | --- | --- |
| 受控情感生成 | $x$ 是 IMDb 电影评论前缀，模型生成正面情感的续写 $y$ | 用预训练情感分类器比较两个生成结果的正面概率，GPT-2-large 在 IMDb 训练集上训练至收敛作为 SFT 模型 |
| Reddit 摘要 | $x$ 是 Reddit forum post，$y$ 是概括主要内容的摘要 | 使用 Reddit TL;DR 数据集和 Stiennon 等人收集的人类偏好；DPO、PPO、Preferred-FT 使用同一个 GPT-J SFT 模型 |
| 单轮对话 | $x$ 是人类查询，模型生成有帮助且有吸引力的回答 | 使用 Anthropic Helpful and Harmless 数据集，其中包含 17 万条对话；每条记录包含两个大模型回答和人类偏好标签 |

在单轮对话任务中没有现成的 SFT 模型。作者从预训练 Pythia-2.8B 出发，先用 Preferred-FT 在偏好回答上训练参考模型，使生成结果处于模型分布附近，再进行 DPO 训练。

**对比方法**

| 方法 | 实验中的做法 |
| --- | --- |
| Zero-shot / 2-shot prompting | 摘要任务使用 GPT-J zero-shot；对话任务使用 Pythia-2.8B 进行 2-shot prompting |
| SFT | 直接使用 SFT 模型作为基线 |
| Preferred-FT | 对偏好回答 $y_w$ 进行监督微调 |
| Unlikelihood | 提高 $y_w$ 的概率并降低 $y_l$ 的概率，并使用可选系数 $\alpha\in[0,1]$ 控制 unlikelihood 项 |
| PPO | 使用偏好数据学习奖励模型，再用 PPO 优化策略 |
| PPO-GT | 在受控情感任务中直接使用真实奖励函数的 PPO oracle |
| Best of $N$ | 从 SFT 或 Preferred-FT 模型采样 $N$ 个回答，用学习到的奖励函数选出分数最高者；测试时计算成本高，因为每个输入都要生成 $N$ 个回答 |

**评价方式**

| 场景 | 评价指标 |
| --- | --- |
| 受控情感生成 | 在真实情感分类器给出的奖励下，观察平均 reward 与参考策略的平均 sequence-level KL；sequence-level KL 是每个 timestep KL 的总和 |
| 摘要 | 生成测试集摘要，并用 GPT-4 计算相对于测试集参考摘要的平均 win rate；作者另做了人工评估 |
| 单轮对话 | 用 GPT-4 计算相对于 Anthropic HH 测试集偏好回答的 win rate |
| 人工验证 | 在 TL;DR 结果上比较 GPT-4 与人类判断，并报告 win rate 及逐判断的一致性 |

**1. 受控情感生成：reward-KL frontier**

这个实验直接使用真实情感分类器作为奖励函数，因此可以同时测量 reward 和 KL，而不是只看一个最终分数。作者对每种方法进行多次训练，并改变控制策略保守程度的超参数：

- PPO 的 target KL 为 $\{3,6,9,12\}$
- DPO 的 $\beta$ 为 $\{0.05,0.1,1,5\}$
- Unlikelihood 的 $\alpha$ 为 $\{0.05,0.1,0.5,1\}$
- Preferred-FT 使用不同随机种子
- 所有方法合计运行 22 次

训练过程中每 100 个 training steps 评估一次，直到收敛。作者报告 DPO 产生了明显更高效的 reward-KL frontier：在较低 KL 下取得更高 reward，并严格优于 PPO；即使 PPO-GT 可以访问真实奖励函数，DPO 的 frontier 仍然更好。

**2. 摘要：与 PPO 和其他方法比较**

作者在 Reddit TL;DR 测试集上改变采样温度，比较模型摘要相对于参考摘要的 GPT-4 胜率：

| 方法 | 最佳或代表性结果 |
| --- | --- |
| DPO | 温度为 0.0 时约 61% |
| PPO | 最优采样温度为 0.0 时约 57% |
| DPO 对 PPO 的人工比较 | DPO 在温度 0.25 下的摘要有 58% 的比较中胜过温度 0 的 PPO 摘要 |

论文还报告：

- DPO 的最高胜率高于 Best of $N$ baseline
- DPO 的 $\beta$ 没有经过有意义的调参，因此结果可能低估其潜力
- DPO 对采样温度更稳健；PPO 在高温度下的表现可能退化到基础 GPT-J 模型水平
- Preferred-FT 相比 SFT 没有显著提升

**3. 单轮对话：Anthropic HH**

在 Anthropic HH 测试集的单轮人机交互子集上，作者将 DPO 与 Preferred-FT 的 Best of 128、Pythia-2.8B 的 2-shot prompting 进行比较。结果显示，DPO 在各方法的最佳采样温度下表现相当或更好。

作者还评估了一个来源公开的 PPO RLHF 模型，但没有找到能让它超过基础 Pythia-2.8B 的 prompt 或采样温度。由于 PPO 和 Best of $N$ 都基于同一类奖励优化目标，作者将 Best of 128 视为 PPO-level performance 的一个近似参照。论文报告 DPO 是唯一一个在计算上高效、同时能超过 Anthropic HH 偏好回答的办法，并且效果与计算成本很高的 Best of 128 相近或更好。

**4. 输入分布变化：CNN/DailyMail**

作者把 Reddit TL;DR 实验得到的 PPO 和 DPO 策略迁移到 CNN/DailyMail 测试集中的新闻文章，并沿用 TL;DR 实验中表现最好的两个采样温度。GPT-4 相对于真实摘要的胜率如下：

| 方法 | Temperature 0 | Temperature 0.25 |
| --- | ---: | ---: |
| DPO | 0.36 | 0.31 |
| PPO | 0.26 | 0.23 |

在这个新的输入分布上，DPO 仍然显著超过 PPO。作者将此结果作为初步证据，说明 DPO 的泛化能力可以与 PPO 相近；同时，DPO 没有使用 PPO 所使用的额外无标签 Reddit TL;DR prompts。

**5. GPT-4 判断与人工判断**

作者在 TL;DR 摘要结果上进行人工研究，并使用两个 GPT-4 prompt：

- GPT-4（S）：只询问哪个摘要更好地概括文章的重要信息
- GPT-4（C）：除了概括质量，还询问摘要是否更简洁；作者发现该 prompt 会影响模型对更长、更重复摘要的偏好

三列方法都与贪心采样的 PPO 进行比较：DPO 使用温度 0.25，SFT 使用温度 0.25，PPO-1 使用温度 1.0。

| 指标 | DPO | SFT | PPO-1 |
| --- | ---: | ---: | ---: |
| N respondents | 272 | 122 | 199 |
| GPT-4（S）win % | 47 | 27 | 13 |
| GPT-4（C）win % | 54 | 32 | 12 |
| Human win % | 58 | 43 | 17 |
| GPT-4（S）-Human agreement | 70 | 77 | 86 |
| GPT-4（C）-Human agreement | 67 | 79 | 85 |
| Human-Human agreement | 65 | - | 87 |

论文指出，GPT-4 与人类的一致程度通常与人类之间的一致程度相近或更高，因此在本文的摘要实验中使用 GPT-4（C）作为主要自动评估 prompt。

## Limitation

下面只保留作者在 “Limitations & Future Work” 中明确提出的问题：

- **分布外泛化仍需要更全面的研究。** 论文在 CNN/DailyMail 新闻文章上给出了初步结果，显示 DPO policy 的泛化能力与 PPO-based model 相近，但作者指出还需要更系统的研究，例如考察 DPO policy 的 self-labeling 能否有效利用无标签 prompts。

- **直接偏好优化中的 reward over-optimization 仍是开放问题。** 作者提出需要进一步研究 reward over-optimization 在 DPO 中如何表现，并指出 Figure 3 右侧出现的轻微性能下降是否就是这种现象的实例。

- **规模扩展尚未完成。** 论文实验覆盖的模型最大约为 6B 参数；将 DPO 扩展到大几个数量级的 state-of-the-art 模型，是作者提出的后续方向。

- **自动评估会受到 prompt 影响。** 论文发现 GPT-4 计算的 win rate 会受到评估 prompt 的影响，未来需要研究如何让自动评估系统产生更高质量、更稳定的判断。
