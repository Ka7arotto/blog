---
title: "RLHF：Training a Helpful and Harmless Assistant"
description: "基于 Anthropic 的 RLHF 论文，梳理 helpfulness 与 harmlessness 偏好数据、Preference Modeling、PPO 训练、在线迭代、鲁棒性实验和作者明确提出的限制。"
publishDate: "20 Aug 2025"
tags: ["paper", "llm"]
---

## 引言

*Training a Helpful and Harmless Assistant with Reinforcement Learning from Human Feedback* 研究如何通过 preference modeling（偏好建模）和 reinforcement learning from human feedback（RLHF）训练一个相对 helpful、harmless 的语言助手。论文的基本流程是：收集人类对模型回答的比较数据，训练 preference model（PM）预测偏好，再把 PM 的分数作为奖励，用 PPO 训练语言模型策略。

论文分别收集 helpfulness 和 harmlessness 两类偏好数据。对于 helpfulness，标注者让模型完成问答、写作、编辑、计划和决策讨论等文本任务，并选择更 helpful、honest 的回答；对于 harmlessness，标注者主动进行 red teaming，尝试诱导模型产生有害回答，并选择两个回答中更 harmful 的一个。作者强调，论文的目标不是规定 “helpful” 和 “harmless” 的规范定义，而是考察这些训练技术是否有效，因此主要让 crowdworkers 按自己的理解进行判断。

论文还研究了三个问题：RLHF 是否会损害模型能力，偏好模型和 RLHF 训练是否足够鲁棒，以及通过大约每周一次的 online 迭代收集新数据能否持续改善模型。作者报告称，大模型经过 RLHF 后在多数 NLP 评测上表现更好，并且可以和代码、摘要等专门技能结合；但小模型会出现 alignment tax，策略被持续优化以最大化 PM 时也会出现鲁棒性下降和 reward hacking 风险。

:::important[论文的核心结论]
RLHF 可以让当前能力水平的语言模型更 helpful、更 harmless，并且大模型通常不会因此损失能力；但这不等于已经解决了 honesty、最坏情况安全性、分布外行为或更强模型能力下的 alignment 问题
:::

:::note[论文对 HHH 的限定]
HHH 表示 helpful、honest、harmless。作者没有在本文中专门研究 honesty，并认为单纯依靠 human feedback 可能不是训练诚实模型最高效、最有效的方法；本文主要研究 helpfulness 和 harmlessness
:::

| 项目 | 内容 |
| --- | --- |
| 论文 | *Training a Helpful and Harmless Assistant with Reinforcement Learning from Human Feedback* |
| 中文理解 | 使用人类反馈强化学习训练有帮助且无害的助手 |
| 作者 | Yuntao Bai、Andy Jones、Kamal Ndousse、Amanda Askell、Anna Chen、Nova DasSarma、Dawn Drain、Stanislav Fort、Deep Ganguli、Tom Henighan、Nicholas Joseph、Saurav Kadavath、Jackson Kernion、Tom Conerly、Sheer El-Showk、Nelson Elhage、Zac Hatfield-Dodds、Danny Hernandez、Tristan Hume、Scott Johnston、Shauna Kravec、Liane Lovitt、Neel Nanda、Catherine Olsson、Dario Amodei、Tom Brown、Jack Clark、Sam McCandlish、Chris Olah、Ben Mann、Jared Kaplan |
| 机构 | Anthropic |
| 论文版本 | arXiv:2204.05862v1，12 Apr 2022 |
| 数据集 | [Anthropic HH-RLHF](https://github.com/anthropics/hh-rlhf) |
| 原文 | [arXiv:2204.05862](https://arxiv.org/abs/2204.05862) |

## 现有方法做法和不足

论文将自己的方法放在几类已有路线中比较：

| 方法 | 基本做法 | 论文指出的问题或差异 |
| --- | --- | --- |
| Plain language model | 直接使用大规模无监督预训练得到的语言模型 | 模型学到的目标、优先级和能力来自复杂的人类数据分布，其中可能包含不希望模型模仿的行为 |
| Context distillation | 在 prompt 中加入 HHH 对话或示例，让模型在上下文中表现出期望行为 | 更接近提示式控制，不是通过参数训练直接改变模型；论文把它作为 RLHF 的重要基线 |
| SFT / instruction tuning | 使用高质量的人类示范进行监督微调 | 专家示范成本较高，而相对偏好判断通常更容易收集；只训练示范不直接利用“两个回答哪个更好”的信息 |
| Preference modeling | 对两个回答收集人类偏好，训练一个模型预测更好的回答 | 只得到偏好分数，还需要后续策略优化；PM 可能在训练分布外失准 |
| RLHF + PPO | 先训练 PM，再让策略生成回答，用 PM 分数和 KL 惩罚进行 PPO 更新 | 需要训练 PM、策略和相关基础模型；训练过程复杂，可能不稳定，并且策略会利用 PM 的漏洞获得高分 |
| 只优化 helpfulness 或 harmlessness | 将其中一个目标单独作为训练重点 | 论文观察到二者存在张力：只追求 harmlessness 可能让模型频繁拒答，只追求 helpfulness 又更容易被 red-team 诱导 |

**传统 RLHF 的问题链条**

论文采用的标准 RLHF 流程可以拆成两步：

1. 准备带比较标签的数据，训练一个 PM，让更好的回答得到更高分。
2. 抽取这些比较数据中的 prompt，让 RL policy 自回归生成回答，并在完整回答结束时获得 PM 提供的 reward。

在这个流程中，一次完整回答可以被看成一条 trajectory，整段回答结束时的 PM 分数是最终 reward。问题是，PM 只在已有比较数据附近接受过训练；当 RL policy 持续最大化 PM 分数并走到更高分区域时，PM 可能不再准确，策略就可能学会“取悦 PM”而不是改善人类真正关心的行为。

## 研究的 Motivation–Insight–Contributions

| 维度 | 论文中的核心内容 |
| --- | --- |
| Motivation 1 | 作者希望开发训练 helpful、honest、harmless AI agents 的技术，并展示通过 human preference data、Preference Modeling 和 RLHF 可以训练相对 helpful、harmless 的自然语言助手 |
| Motivation 2 | 作者不规定 “helpful” 和 “harmless” 的含义，而是评估训练技术的有效性；因此分别收集 helpfulness 与 harmlessness 的人类偏好数据 |
| Motivation 3 | 作者进一步研究 alignment tax、alignment training 与代码和摘要等 specialized skills 的兼容性，以及 preference modeling 的 scaling、RLHF robustness、iterated online training 和 OOD detection |
| Insight 1 | 作者发现小模型会出现明显的 alignment tax，而 13B 和 52B RLHF 模型在 zero-shot 能力评测上得到 alignment bonus、在 few-shot 评测上保持相当表现；代码模型上的自然语言 RLHF 提升了编程评测，和摘要技能混合训练没有造成性能下降 |
| Insight 2 | helpfulness 与 harmlessness 之间存在 tension；随着模型规模增大，PM 在两类分布上的表现同时变好，并且对训练数据比例更鲁棒 |
| Insight 3 | PM 准确率随模型规模和数据规模大致呈 log-linear 趋势；在 RLHF robustness 实验中，较大的 PM 更鲁棒，而 RLHF 训练中的 overfitting 会增加 |
| Insight 4 | 在相当一部分 RLHF 训练过程中，$\sqrt{D_{\mathrm{KL}}(\pi\|\pi_0)}$ 与 reward 近似线性相关；按周进行的 iterated online training 提升了 crowdworker 评测结果，并改善了数据集质量的上尾 |
| Insight 5 | OOD detection 可以在很少甚至没有 harmful examples 的情况下，拒绝大多数 strange 和 harmful requests |
| Contributions | 作者收集分别用于 helpfulness 和 harmlessness 的数据，并构造 initial、rejection sampling、online RLHF 三批数据；同时研究 alignment bonus/tax、specialized skills、helpfulness–harmlessness tension、scaling、robustness、online training 和 OOD detection |

## 核心创新点

1. **把 helpfulness 和 harmlessness 设计成两类不同的数据**

   helpfulness 数据让标注者提出开放式文本任务，并在两个回答中选择更 helpful、honest 的一个；harmlessness 数据则让标注者主动进行 red teaming，并选择更 harmful 的回答。论文不把这两类任务简单地合并成一个模糊标签，而是保留它们的分布差异，再研究混合训练后的效果。

   这种设计也带来一个重要问题：在 harmlessness 数据中选择“更 harmful”的回答，适合探索模型如何被攻击，却没有直接教模型面对有害请求时应该如何进行有帮助的解释和协商。论文在后文将这一点与 helpfulness 和 harmlessness 的张力联系起来。

2. **用 Preference Model 把相对偏好转成 RL reward**

   对于回答 $A$ 和 $B$，PM 用分数差表示偏好强度。论文给出的概率关系为：

   $$
   P(A>B)
   =
   \frac{1}
   {1+\exp\left(r_{\mathrm{PM}}(B)-r_{\mathrm{PM}}(A)\right)}
   $$

   分数差越大，模型越认为 $A$ 会被人类偏好。RL 阶段直接使用 PM score 作为 reward，策略不需要为每个目标手写一个可微的评价函数。

3. **用 PPO 和 KL penalty 控制策略偏离**

   论文用 PPO 稳定 RL 训练，并在 PM reward 上加入策略与初始策略之间的 KL 惩罚：

   $$
   r_{\mathrm{total}}
   =
   r_{\mathrm{PM}}
   -
   \lambda_{\mathrm{KL}}
   D_{\mathrm{KL}}
   \left(
   \mathrm{policy}
   \middle\|
   \mathrm{policy}_0
   \right)
   $$

   实验中使用 $\lambda_{\mathrm{KL}}=0.001$。作者指出，RL 训练的大部分阶段通常有 $D_{\mathrm{KL}}<100$，因此这个惩罚在多数时候影响很小，甚至可能并非必需；它的作用更多是作为一种经验性的稳定和约束机制。

4. **用独立 PM 测量 RLHF 的鲁棒性**

   论文把静态偏好数据拆成两半，分别训练 train PM 和 test PM；策略只针对 train PM 优化，但训练过程中同时用 test PM 评估。如果 train PM 分数继续上升，而 test PM 分数开始落后，说明策略可能在过拟合或利用 train PM 的漏洞。

   这种设计没有直接声称 test PM 就等同于人类判断。论文明确指出，两个 PM 仍可能共享相关的鲁棒性缺陷，因此这是理解 PM 过拟合的实验工具，而不是完整的安全保证。

5. **提出 iterated online RLHF**

   作者先训练当前最好的 RLHF policy，再让它与 crowdworkers 交互，收集位于高分区域的新比较数据；随后把新数据和旧数据混合，重新训练一组 PM 和 RLHF policy，并大约每周重复一次：

   $$\text{当前 RLHF policy}\rightarrow\text{新的人类比较数据}\rightarrow\text{新的 PM}\rightarrow\text{新的 RLHF policy}$$

   online 的关键目的，是填补旧数据在高质量回答区域中的空缺，使 PM 能够学习区分更细微的高质量样本。论文中的 online 指每一轮重新训练新的模型，而不是在同一个 PM 或同一个 RLHF 模型上原地更新。

6. **研究 alignment 与模型能力、专门技能的关系**

   论文不只观察 helpfulness 和 harmlessness，还评估 MMLU、LAMBADA、HellaSwag、OpenBookQA、ARC、TriviaQA、TruthfulQA、HumanEval 等任务，研究自然语言 RLHF 是否会影响通用能力、诚实性和代码能力。

   论文的主要观察是：大模型常出现 alignment bonus，小模型更容易出现 alignment tax；自然语言 RLHF 可以接在代码微调之后，大模型的代码评测能力还可能提升，但小模型的训练更难稳定。

7. **用 OOD detection 作为另一条安全路径**

   除了让模型通过语言对话学习 harmlessness，论文还尝试从 helpfulness 数据的内部表示中检测偏离分布的请求。作者提出 Simplified Relative Mahalanobis distance，用完整协方差和仅保留对角项的协方差之差作为分数：

   $$
   \mathrm{score}(x)
   =
   (x-\mu)^{\mathsf T}\Sigma^{-1}(x-\mu)
   -
   (x-\mu)^{\mathsf T}
   \Sigma_{\mathrm{diag}}^{-1}
   (x-\mu)
   $$

   这不是让 detector 学会所有 harmful 类别，而是把 helpfulness 数据作为 in-distribution，检测偏离它的输入。

## 方法框架

**1. 先统一符号**

| 符号或术语 | 含义 |
| --- | --- |
| Helpfulness | 模型是否真正帮助用户完成文本任务，并按论文的标注说明兼顾 honesty |
| Harmlessness | 模型是否避免产生有害内容；论文通过 red teaming 构造这类数据 |
| PM / Preference Model | 根据两个回答和上下文，预测人类会偏好哪一个的模型 |
| $\pi_0$ / policy0 | RL policy 的初始策略 |
| $\pi$ / policy | 正在通过 RLHF 训练的策略 |
| $r_{\mathrm{PM}}$ | PM 对完整回答给出的 reward |
| $D_{\mathrm{KL}}$ | 当前策略相对初始策略的 KL 距离 |
| PPO | 用来稳定优化语言模型策略的强化学习算法 |
| HHH | helpful、honest、harmless 的缩写 |

**2. 数据收集：让人类在对话中比较两个回答**

每一轮对话包含两个动作：

1. crowdworker 发送一条消息，请模型执行任务、回答问题或讨论主题。
2. 系统展示两个可能的模型回答，crowworker 选择更 helpful、honest 的回答；在 red-teaming 任务中选择更 harmful 的回答。

对话可以继续多轮，下一轮用户消息会基于上一步选择的回答产生。论文的数据不是直接把所有回答打成绝对分数，而是保留两个回答之间的比较关系。界面允许标注偏好强度，但论文只保留强于最弱偏好的比较，并把剩余比较当作等权二元标签，不使用平局。

论文收集了三种主要数据分布：

| 数据分布 | 生成数据的模型 | 比较数量 |
| --- | --- | --- |
| Core base dataset | HHH context-distilled LM | 44k helpfulness，42k red-teaming |
| Rejection sampling dataset | 在 52B context-distilled LM 上使用 PM 做 rejection sampling | 52k helpfulness，2k red-teaming |
| Iterated online dataset | RLHF 模型，约每周更新一次 | 22k helpfulness，没有 red-teaming |

论文把前两部分合称 static 或 base+RS dataset。online RLHF policy 使用三部分数据。作者还指出，绝大部分偏好数据来自 52B 模型，这会使小模型生成的回答更容易落在 PM 的分布外。

**3. Preference Model：先学习“哪个回答更好”**

论文使用 7 个语言模型，参数量从 13M 到 52B，规模大致按约 4 倍的几何级数递增。PM 训练前先进行 preference model pretraining（PMP），再在 human feedback 数据上微调；通常只训练一个 epoch，并使用固定学习率观察数据量和模型规模的影响。

PM 输出一个标量分数。若 $r_{\mathrm{PM}}(A)>r_{\mathrm{PM}}(B)$，则模型预测人类更可能偏好 $A$。这一分数后来会被当作 RL policy 生成完整回答后的 reward。

论文还检查 PM 是否 calibrated。理想情况下，分数差应该能预测人类偏好概率；helpfulness-only PM 校准得很好，而 helpfulness 与 harmlessness 混合的 PM 略微 under-confident。随着回答质量提升，区分高质量样本会更困难，PM 的准确率会下降，这也是在线收集高分数据的动机。

**4. RLHF：用 PM reward 更新 policy**

RLHF 的训练流程如下：

1. 从偏好比较数据中提取 prompt。
2. policy 针对每个 prompt 自回归生成回答。
3. PM 读取完整回答并给出 $r_{\mathrm{PM}}$。
4. 计算当前 policy 和初始 policy 的 KL 惩罚。
5. 用总 reward 通过 PPO 更新 policy。

论文除了使用 crowdworker 写出的 prompt，还用大语言模型根据约 10 个高质量人类查询进行 few-shot 生成，以增加 RL prompt 数量。实际 RLHF 训练使用了 static 数据中的 137k prompts 和模型生成的 369k prompts。

完整 reward 为：

$$
r_{\mathrm{total}}
=
r_{\mathrm{PM}}
-
\lambda_{\mathrm{KL}}
D_{\mathrm{KL}}
\left(
\pi
\middle\|
\pi_0
\right)
$$

这里的 PM 分数通常在完整回答结束时给出，因此整段回答可以看成一条 trajectory，而 PM 分数是末端 reward。

**5. 为什么 reward 变高不一定意味着行为变好**

PM 是根据有限分布上的人类比较训练出来的。RL policy 的目标是最大化 PM score，因此可能走到 PM 没有充分覆盖的区域。如果 PM 在那里把某个表面特征误认为高质量，policy 就有机会获得更高 reward，却没有获得人类真正想要的能力。

论文用 train PM / test PM 的差异观察这一点：早期训练时二者分数接近；当 RL 样本继续增加，train PM 分数变高而 test PM 分数落后，说明策略可能在 train PM 上过度优化。作者把这种现象与 robustness failure 和 reward hacking 联系起来。

**6. helpfulness 和 harmlessness 如何混合**

作者发现，模型只训练 helpfulness 时更容易回答用户，但也更容易被 red-team；过度优化 harmlessness 时，模型可能对敏感问题重复拒答，损害实际有用性。PM 混合训练和 RLHF 中的 prompt 比例都会影响这种平衡。

论文还研究了按损失加权的方式：

$$
L_{\mathrm{Total}}
=
L_{\mathrm{Helpfulness}}
+
\lambda L_{\mathrm{Harmlessness}}
$$

实验测试 $\lambda\in\{1,2,3,4,10\}$。结果显示大模型对 $\lambda$ 的选择更鲁棒；在 13M 模型上把 $\lambda$ 从 1 增加到 10，helpfulness accuracy 下降 7.4%，而在 52B 模型上下降 1.5%。

**7. iterated online RLHF 如何闭环**

一轮 online 迭代包含：

1. 使用当前 PM 训练出尽可能好的 RLHF policy。
2. 让 crowdworkers 与这个 policy 交互，收集高分区域的比较数据。
3. 将新比较数据与已有数据混合。
4. 重新训练新的 PM 和 policy。
5. 重复这个过程。

论文注意到 RL 会降低 policy entropy，可能减少在线数据的多样性，因此同时部署多个 RL snapshot 和多个 online iteration 的模型，以增加比较数据的多样性。

**8. OOD detector 如何只利用 helpfulness 数据**

对于 prompt $x$，从模型第 $\ell$ 层提取激活向量 $v_\ell\in\mathbb R^{d_{\mathrm{model}}}$。在 helpfulness 训练数据上计算均值 $\mu$、完整协方差 $\Sigma$ 和对角协方差 $\Sigma_{\mathrm{diag}}$，再用二者的 Mahalanobis 距离差作为 score。

它的思路不是训练一个覆盖所有有害类别的分类器，而是先学习“什么像 helpfulness 数据”，再把明显偏离 helpfulness 分布的输入作为可疑输入。因此，harmlessness 数据在这里被当成一种 non-helpfulness 分布。

## 实验设计

**实验问题**

论文的实验覆盖以下问题：

1. PM 的准确率如何随模型规模、数据规模和对话轮次变化？
2. PM 是否 calibrated，RLHF 是否会过度优化 PM？
3. helpfulness 与 harmlessness 是否存在冲突，混合训练能否缓解？
4. RLHF 是否损害通用语言能力、代码能力和摘要等 specialized skills？
5. iterated online RLHF 是否能改善人类偏好和高分数据分布？
6. OOD detection 能否在很少 harmful examples 的情况下识别有害请求？

**评测数据与指标**

| 类型 | 数据集或指标 | 论文中的用途 |
| --- | --- | --- |
| 通用 NLP | MMLU、LAMBADA、HellaSwag、OpenBookQA、ARC、TriviaQA | 比较 plain LM 与 RLHF 模型的 zero-shot、few-shot 能力 |
| Alignment | HHH Evaluations、TruthfulQA、BBQ-Lite、gender bias、race/religion sentiment | 检查 helpfulness、honesty、harmlessness 和偏见相关行为 |
| 代码 | HumanEval，报告 pass@k | 检查自然语言 RLHF 是否能接在 Python code fine-tuning 后 |
| 摘要 | summarization preference modeling 数据 | 检查 HH 训练是否损害 specialized summarization skill |
| 人类评测 | crowdworker preference、Elo score、与 professional writers 比较 | 衡量真实人类对不同模型输出的偏好 |
| 鲁棒性 | train PM / test PM score、policy 与 policy0 的 KL | 观察 PM 过拟合和 RLHF 分布偏移 |
| OOD | AUROC、Mahalanobis distance | 检查 helpfulness 与 harmlessness 输入的可分性 |

为减少定性样本中的 cherry-picking，论文针对每个 prompt 生成 17 个样本，用 online HH PM 排序后展示其中的 median sample。

**PM scaling 和 calibration**

- PM 使用 13M 到 52B 的 7 个规模，准确率随模型规模和数据规模大致呈 log-linear 趋势
- 只训练 helpfulness 的 PM 趋势更稳定；不同数据分布有时会出现偏离简单 scaling law 的情况
- PM 在对话第一轮的准确率略高，之后各轮大致保持稳定
- helpfulness-only PM 校准较好，HH 混合 PM 略微 under-confident
- 52B static PM 在 HHH evaluation 上达到 86%，论文将其与报告中的 75% mean human score 作比较
- PM 也会被“诚实但不够有帮助”的正确回答与“写得很好但包含细微错误”的回答误导，说明 PM 并不具备充分的 adversarial robustness

**RLHF 的通用能力结果**

论文在多项 NLP 评测上比较 plain LM、context-distilled 模型和 RLHF 模型：

- 13B 和 52B RLHF 模型在 zero-shot NLP 评测中通常优于 base LM，在 few-shot 评测中大致相当
- 汇总结果中，除 TriviaQA 外，较大的 RLHF 模型在列出的评测上优于 base LM
- 小模型会出现 alignment tax，模型规模增大后则观察到 alignment bonus
- RLHF 训练还提高了 TruthfulQA 表现，但作者认为 honesty 仍有很大改进空间

这些结果支持“对大模型进行 alignment training 不必然损害能力”，但不表示 RLHF 在所有规模、任务或评测格式下都带来提升。

**helpfulness 与 harmlessness 的张力**

论文观察到：

- 只训练 helpfulness 的模型更容易在 red-teaming 中产生有害回答
- helpfulness 与 harmlessness 混合训练后，模型可以在普通请求中保持帮助性，并在有害请求上更少产生有害回答
- 过度优化 harmlessness 可能使模型对敏感话题进行过多拒答，例如不加区分地建议用户寻求专业帮助
- 随模型规模增长，PM 同时处理两类分布的能力变强，对 helpfulness/harmlessness 数据比例也更鲁棒

作者把这一张力部分归因于 harmlessness 数据的收集方向：red-teaming 时选择更 harmful 的回答有利于探索攻击面，但没有充分提供“如何有帮助地处理有害请求”的正向示例。

**RLHF robustness 实验**

论文使用两种设置：

| 设置 | 做法 |
| --- | --- |
| Train PM size = 52B | 用同一个 52B train PM 训练不同大小的 policy，再用独立 test PM 评估 |
| Train PM size = policy size | 每个 policy 使用同规模 train PM，再用不同规模的 test PM 评估 |

每种扫描包含 7 个模型规模，因此每个实验有 7 个 policy 和 $7\times7$ 组评估。主要结果是：

- 训练早期 train PM 和 test PM 分数接近
- 大约超过 150k RL training samples 后，train PM 分数与 test PM 分数开始明显分离
- 更大的 PM 更鲁棒，小模型更难稳定训练
- 由于偏好数据主要由 52B 模型生成，小模型的回答对 PM 来说更容易是 OOD
- 训练针对较小 PM 的 policy，最终由 52B PM 评估时表现较差

**$\sqrt{D_{\mathrm{KL}}}$ 与 reward 的关系**

作者观察到，在大量 RLHF 训练过程中，PM reward 与策略相对初始策略的 KL 之间近似满足：

$$
\mathrm{reward}
\propto
\sqrt{
D_{\mathrm{KL}}(\pi\|\pi_0)
}
$$

这里的 KL 是在 policy 生成的样本上估计的 sequence-level KL。论文给出的解释是：KL 在小变化 $\delta\pi$ 附近从二阶项开始，而 reward 如果对 $\delta\pi$ 近似线性变化，就会得到 reward 与 $\sqrt{D_{\mathrm{KL}}}$ 近似线性关系。作者将其视为一种经验关系和后续研究方向，而不是对所有训练阶段的严格定理。

**iterated online RLHF**

online 数据在最终 online PM 看来覆盖了更高的 score 区域。online PM 在 base、rejection-sampled、online-only 三类 held-out 数据上的准确率分别约为 74%、70% 和 67%，说明越高质量的样本越难区分。

作者还做了控制实验：两次 52B RLHF 训练使用等量数据和相同超参数，一次使用 base dataset，另一次使用 base、RS、online 数据的等量混合。crowdworker 的 Elo 结果更偏好 online 混合数据训练的 policy，说明提升不只是由数据量增加或超参数变化造成。

Figure 1 中，crowdworkers 对 final online HH model 的整体偏好高于 static RLHF 和 context-distilled 模型；作者也把 online HH 模型与 professional writers 的回答进行比较，报告模型回答约有 56%–57% 的比较被偏好。作者同时提醒，这个写作者评测不是 adversarial 的，prompt 分布也可能不同，因此不应把它当作真实世界任务的完整结论。

**specialized skills：摘要与代码**

在 summarization 中，论文把 HH 数据与 Learning to Summarize 数据混合训练 PM，并发现大模型在两类任务上的 PM accuracy 都没有明显下降，说明两类偏好建模目标可以共存。

在代码实验中，作者先对 Python code fine-tuned 模型进行自然语言 RLHF，再在 HumanEval 上评测：

- RLHF 对小模型的表现通常不利，对大模型则可能提升
- 52B 模型在扫描 temperature、top-p 后，在不同 pass@k 上都显示出相对 base code model 的提升
- 提升幅度整体较小；只使用 prompt 的 base code model 在某些比较中略好
- 在包含 buggy code 的 prompt 中，RLHF 模型没有优于初始 base code model

**OOD detection**

论文使用 helpfulness activation 作为 in-distribution，检测 harmlessness 输入是否偏离该分布：

- 模型越大，AUROC 通常越高；大模型的中间层表现较好
- 不使用 harmful examples 时，所有模型和层中最好的 AUROC 约为 0.85
- 对 64L 模型只加入 10 个 harmful prompt，AUROC 达到 $0.94\pm0.02$
- 4L、13M 参数模型只加入 10 个 harmful examples 时，AUROC 达到 $0.86\pm0.01$
- 论文认为少量 outlier exposure 带来的收益明显大于单纯扩大模型规模带来的收益

## Limitation

以下只保留作者在 7.1 “Limitations” 及正文中明确写出的不足和未解决问题：

- **honesty 仍未解决。** 论文明确表示，虽然 RLHF 提高了 TruthfulQA 等 honesty 评测，但当前结果只是触及问题表面；其他方法可能更高效、更有效。

- **主要研究平均行为，没有解决最坏情况。** 论文基本关注模型的 average-case behavior，并没有系统研究如何消除最坏情况下的有害行为。作者认为，随着部署时出现 distributional shift，这个问题会越来越重要。

- **alignment 的评估仍然困难且含义不明确。** 大型 RLHF 模型虽然在多数能力评测上优于 plain LM，但距离“zero-shot 就能达到未对齐模型 few-shot 水平”的理想仍然很远；在 TruthfulQA 上也只弥合了这一差距的一部分。

- **表面 alignment 的风险仍存在。** RLHF 模型对不同种族和宗教的 sentiment 更积极，但这不一定代表 bias 已经减少；gender bias 与底层语言模型的 bias 高度相关。作者尚不能确定这主要是 RLHF 技术的限制，还是 HH 数据集的限制，并呼吁使用更细致、覆盖多轮对话的评测。

- **RL 训练经验和稳定性有限。** 作者遇到过 RL 优化不稳定的问题，虽然进行了一些超参数扫描，但认为更多实践经验可能带来更好的稳定性和性能。

- **online 训练并非原地更新同一个模型。** 论文的 online 流程是每一轮从头重新训练新的 PM 和 RLHF policy，没有探索真正持续更新同一个 PM 或同一个 RLHF model 的方式。

- **开放式人类反馈数据的质量控制仍有限。** 作者指出，开放式对话引入许多难以控制的变量，导致数据质量指标带有噪声；crowdworkers 互评不能很好代表整体对话质量，模型持续更新也使 response comparison 成为 moving target。

- **harmlessness 数据的构造可能不适合教会模型良好行为。** 作者说明，选择更 harmful 的回答是为了充分探索 bad behavior，但这种数据分布可能不适合 teaching good behavior；作者建议后续收集由标注者选择的最佳回答。

- **部分 code RLHF 实验存在训练稳定性边界。** 作者报告，3B code model 的 RLHF optimization 难以稳定，因此将其排除在该部分实验之外。

- **OOD detector 在具体目标任务上的性能较低。** 作者指出，该方法能够区分 helpfulness 与 non-helpfulness，但在这个具体任务上的性能明显较低；加入少量 outlier exposure 后，AUROC 会得到明显改善。

- **更强模型上的适用性尚未验证。** 作者最后提出，需要研究这些技术在 AI 模型能力继续增强后是否仍然适用，并探索处理更高级 failure modes 的方法。
