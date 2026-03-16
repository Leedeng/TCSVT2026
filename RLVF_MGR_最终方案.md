# RLVR-MGR 技术方案（修订版）

## 一、完整故事线

### 问题

现有 MGR 方法（包括我们前作）的文本端仅使用简短 label，语义贫乏，无法充分利用文本模态的判别能力。

### 方案

提出 RLVR-MGR 框架，三个核心创新：

1. **轻量 LLM 动态生成微手势描述**: Qwen3.5-0.8B 替代 DistilBERT，主动生成自然语言描述而非被动编码 label。用 GPT-5 mini 预先生成高质量 MG 描述作为 SFT 数据。

2. **Text-Guided Attention (TGA)**: 文本→Query, 视觉→Key/Value 的 Cross-Attention，文本引导视觉关注关键时空区域。

3. **Reinforcement Learning from Verifiable Rewards (RLVR) + GRPO 优化 LLM**: LLM 是策略网络，生成描述是动作，视觉-文本对齐质量是 **可验证的确定性 reward**（由冻结编码器 + TGA 计算的 cosine similarity，无需学习），GRPO 通过组内比较优化策略。**无需奖励模型（Reward Model），无需价值网络（Value Network），无需人类偏好标注。**

### 为什么是 RLVR 而不是 RLHF

RLHF 需要训练一个 Reward Model 来拟合人类偏好，成本高且引入近似误差。而在我们的场景中，reward 信号天然可验证：

- **R1（对齐奖励）**: cosine_similarity(v_hat, t_pooled) — 确定性函数，输入确定则输出确定
- **R2（判别奖励）**: 类间余弦距离 — 确定性函数
- **R3（格式奖励）**: 基于规则的长度/质量检查 — 确定性函数

三个 reward 分量均为 **确定性、可计算、无需学习的函数**，这正是 Verifiable Rewards 的定义。因此，我们直接采用 RLVR + GRPO 范式，跳过 Reward Model 训练，既简化了流水线，又避免了 reward model 拟合偏差带来的策略退化。

---

## 二、整体架构

```
阶段一 SFT: GPT-5 mini 生成描述 → 监督微调 LLM → 训练 TGA 和投影层
阶段二 GRPO: LLM 采样 G 条描述 → 可验证 reward 计算 → 组内比较 → 策略更新
阶段三 分类器: 冻结全部编码器 → 训练 MLP 分类器 → MGR
```

### 核心组件

| 组件 | 模型 | 角色 | 训练状态 |
|------|------|------|----------|
| 视觉编码器 | R(2+1)D-18 | 视频 → 时空 token 序列 | 冻结 backbone，训练投影层 |
| LLM 生成器 | Qwen3.5-0.8B (LoRA) | 策略网络，生成 MG 描述 | SFT → GRPO 优化 |
| 文本编码器 | all-MiniLM-L6-v2 | 编码描述为句向量，用于 reward 计算 | 全程冻结 |
| TGA | Cross-Attention | 文本引导视觉关注 | 阶段一训练，阶段二冻结 |

---

## 三、分步实现

---

### Step 0: 环境与数据准备

**目的**: 搭建实验环境，准备所有依赖、数据和模型。

要做的事情:
- 安装依赖：PyTorch, transformers, peft, torchvision, decord, trl（GRPO 支持）, sentence-transformers
- 下载模型：R(2+1)D-18, Qwen3.5-0.8B, `sentence-transformers/all-MiniLM-L6-v2`
- 下载数据集：iMiGUE, SMG
- 获取 OpenAI API key（用于调用 GPT-5 mini 生成 MG 描述）

---

### Step 1: 用 GPT-5 mini 生成 MG 描述

**目的**: 利用 GPT-5 mini 的强大语言能力，为每个 MG 类别生成多条高质量、多样化的描述，作为后续 SFT 阶段的训练数据。GPT-5 mini 的描述质量远超手工编写，且能覆盖动作细节、身体部位、运动模式等多个维度。

要做的事情:
- 收集所有 MG 类别的 label 列表（iMiGUE 有 11 类，SMG 有自己的类别）
- 为每个类别设计 prompt，通过 OpenAI API 调用 GPT-5 mini，生成 10-20 条不同角度的描述
- prompt 设计策略：

  **角度一：动作描述**（怎么做的）
  `"Describe the micro gesture '{label}' in one sentence, focusing on which body parts are involved and the specific motion pattern."`

  **角度二：判别描述**（与其他类别的区别）
  `"Given these micro gesture categories: {all_labels}, describe '{label}' in one sentence that clearly distinguishes it from the most similar categories."`

  **角度三：情感关联描述**
  `"Describe the micro gesture '{label}' in one sentence, focusing on what emotional state or psychological condition it might indicate."`

- 对生成的描述做质量筛选：去重、去除过长/过短、去除幻觉内容
- 最终每个类别保留 10-15 条高质量描述
- 保存为 JSON 文件，格式：`{label: [desc1, desc2, ...]}`

**为什么用 GPT-5 mini**:
- GPT-5 mini 能生成多角度、多样化的描述，覆盖面广
- 质量稳定，不受标注者个体差异影响
- API 调用成本低，可以快速迭代 prompt 策略
- 生成的描述天然是自然语言，适合作为 LLM 的 SFT 训练目标
- 论文中使用外部大模型生成训练数据是被广泛接受的做法（知识蒸馏范式）

---

### Step 2: 数据加载器

**目的**: 构建统一的数据管线，支持 SFT 和 GRPO 两个阶段的不同数据需求。

要做的事情:
- SFT 数据集：每个样本包含 (prompt, GPT-5 mini 生成的目标描述)，用于训练 LLM 的描述生成能力
- 对比学习数据集：每个样本包含 (视频片段, 描述文本, 类别索引)，用于训练 TGA 和视觉对齐
- GRPO 数据集：每个样本包含 (视频片段, prompt, 类别索引)，LLM 在训练循环中在线采样生成描述
- 视频预处理：均匀采样 16 帧，224×224，归一化（与原论文一致）

---

### Step 3: 视觉编码器

**目的**: 将视频片段编码为时空 token 序列，为 TGA 提供细粒度的时空信息供文本引导关注。

要做的事情:
- 加载 R(2+1)D-18 预训练模型，去掉全局池化和分类头
- 保留中间卷积输出，展平为 token 序列 [B, N, C]
- 添加线性投影层 → [B, N, D_align]（D_align = 384，与 all-MiniLM-L6-v2 输出维度对齐）
- 整个训练过程中 backbone 冻结，只训练投影层

---

### Step 4: 文本编码器（独立语义评判空间）

**目的**: 将 LLM 生成的文本描述编码为高质量句向量，用于与视觉特征做对比对齐，并作为 RLVR 的 verifiable reward 计算核心。

**模型选择: `sentence-transformers/all-MiniLM-L6-v2`**

要做的事情:
- 加载冻结的 all-MiniLM-L6-v2（22M 参数，384 维输出）
- 输入：LLM 生成的描述字符串
- 输出：token 序列 t_tokens [B, L, 384]（给 TGA）和池化表示 t_pooled [B, 384]（给对比损失 / reward）
- 全程冻结，不参与任何梯度更新

**为什么用 all-MiniLM-L6-v2 而不是 Qwen 的冻结副本**:

1. **句向量质量高**: all-MiniLM-L6-v2 专门用句子级对比学习训练过，其 cosine similarity 直接反映语义相似度。而 Qwen 是因果语言模型，hidden states 为 next-token prediction 优化，直接池化后算 cosine similarity 效果差，导致 reward 信号噪声大，GRPO 优化不稳定。

2. **显存高效**: 22M 参数（fp16 约 44MB），相比额外加载一个 0.8B 的 Qwen 副本（fp16 约 1.6GB），节省大量显存。GRPO 阶段显存本就紧张（LLM + reference policy + R(2+1)D + TGA），这个节省很关键。

3. **生成器与评判器解耦**: LLM（Qwen）负责生成，Sentence-Transformer 负责评判，二者模型架构完全不同。避免了同一架构"自编自导自演"可能产生的 shortcut learning——LLM 无法通过利用自身编码偏好来刷高 reward，只能生成真正语义好的描述来获得高分。

4. **Reward 信号稳定**: 冻结的、经过充分预训练的 sentence embedding 模型提供了一个稳定的语义锚定空间，保证 verifiable reward 在整个 GRPO 训练过程中的一致性。

---

### Step 5: Text-Guided Attention (TGA)

**目的**: 让文本描述引导视觉关注关键时空区域。文本 token 作为 Query 关注视觉 token 的 Key/Value，输出增强的视觉表示和可解释的 attention map。

要做的事情:
- 实现 Multi-Head Cross-Attention：文本→Query, 视觉→Key/Value
- 输入维度均为 384（与 all-MiniLM-L6-v2 和视觉投影层对齐）
- 输出：增强视觉表示 v_hat [B, 384] + attention map [B, heads, L, N]
- 添加残差连接和 FFN
- 在阶段一训练，阶段二（GRPO）中冻结

---

### Step 6: Reward 函数设计（Verifiable Rewards）

**目的**: 定义 RLVR 的可验证奖励函数——连接对比学习与强化学习的核心桥梁。所有 reward 分量均为确定性可计算函数，无需学习。

要做的事情:

**R1: 视觉-文本对齐奖励（主奖励）** — Verifiable ✓
- 描述 d_g → 文本编码器 (all-MiniLM-L6-v2) → t_g → TGA(t_g, v_tokens) → v_hat_g
- r_align = cosine_similarity(v_hat_g, t_pooled_g)
- 含义：描述与对应视频片段对齐得越好，奖励越高
- 可验证性：冻结编码器 + 冻结 TGA → 相同输入必然产生相同输出

**R2: 类间判别奖励** — Verifiable ✓
- r_discrim = -max(cosine_similarity(t_pooled_g, v_hat_j)) for j with different label
- 含义：好的描述不仅和正确视频对齐，还要和其他类别拉开距离
- 可验证性：同上，确定性函数

**R3: 格式/质量奖励** — Verifiable ✓
- 检查描述长度是否在合理范围（如 10-80 tokens）
- 检查是否包含类别相关的关键词
- 惩罚重复、退化文本
- 含义：防止 LLM 生成退化文本来刷分
- 可验证性：基于规则的确定性检查

**总奖励**: R = α × R1 + β × R2 + γ × R3

**与 RLHF 的关键区别**: RLHF 的 reward model 是一个需要训练的神经网络，存在拟合误差和过优化风险。我们的 reward 函数是由冻结的预训练模型组成的确定性管线，每个分量都可验证、可复现，完全符合 RLVR 范式。

---

### Step 7: 阶段一 — SFT 预训练

**分两个子步骤:**

#### Step 7a: SFT — 教 LLM 生成微手势描述

**目的**: 用 GPT-5 mini 生成的高质量描述作为监督信号，微调 LLM（LoRA），让它具备基本的微手势描述生成能力。这是 GRPO 的必要前置——LLM 必须先会生成合理描述，GRPO 才能在此基础上探索优化。

要做的事情:
- 训练数据：Step 1 中 GPT-5 mini 生成的描述，格式为 (prompt, target_description) 对
- 用标准的语言模型损失（next token prediction）微调 LoRA
- prompt 格式：`"Describe the micro gesture called '{label}' in one sentence:"`
- 训练后验证：让 LLM 对每个类别生成描述，人工检查质量

#### Step 7b: 对比学习预对齐 — 训练视觉投影层和 TGA

**目的**: 在 LLM 已能生成合理描述的基础上，训练视觉投影层和 TGA，建立视觉-文本对齐空间。同时使 reward 函数能提供有意义的信号。

要做的事情:
- 冻结 LLM（SFT 后的版本），对每个训练样本用 greedy decoding 生成一条描述
- 冻结文本编码器 (all-MiniLM-L6-v2) 编码描述
- 可训练：视觉投影层 + TGA
- 损失：KL 对比损失（复用前作 `loss_utils.py` 中的 `KLLoss`）
- 学习率 1e-4，训练约 30 epochs

**注意**: 此步骤结束后，reward 管线（文本编码器 + TGA + 视觉投影层）全部就位且冻结，GRPO 阶段的 verifiable reward 即可使用。

---

### Step 8: 阶段二 — GRPO 强化学习（核心）

**目的**: 在 SFT 基础上，用 GRPO + Verifiable Rewards 进一步优化 LLM 的描述生成策略。LLM 对每个类别采样多条描述，可验证的 reward 函数计算奖励，GRPO 通过组内相对比较强化高 reward 的描述策略、抑制低 reward 的策略。

要做的事情:
- 冻结：R(2+1)D backbone, 视觉投影层, TGA, all-MiniLM-L6-v2
- 可训练：仅 LLM 的 LoRA 参数
- 参考策略 π_ref：Step 7a SFT 后的 LLM（LoRA 关闭 adapter 即可，无需额外加载模型副本）

**GRPO 训练循环**:
1. 对每个 batch 中的样本 (V_i, prompt_i, label_i)
2. LLM 温度采样生成 G 条描述 {d_1, ..., d_G}
3. 每条描述经 all-MiniLM-L6-v2 + TGA 计算 verifiable reward
4. 组内归一化得到 advantage: A_g = (r_g - mean) / std
5. 计算 clipped policy ratio × advantage
6. 加 KL(π_θ || π_ref) 正则化
7. 更新 LoRA 参数

**GRPO + RLVR 的优势**:
- **无需 Reward Model**: reward 由冻结管线直接计算，零额外训练成本
- **无需 Value Network**: GRPO 用组内相对比较估计 advantage，不需要 critic
- **Reward 无漂移**: 冻结编码器保证 reward 信号在训练全程稳定一致
- **计算高效**: 相比 PPO 省掉 reward model 和 value network 的前向传播与更新

**核心超参数**: G=8, ε=0.2, β_kl=0.04, 温度=0.8, 最大生成长度=80, lr=1e-6

---

### Step 9: 阶段三 — MLP 分类器微调

**目的**: 用 GRPO 优化后的完整管线提取视觉特征，训练分类器完成 MGR。

要做的事情:
- GRPO 后的 LLM 对每个 MG 类别用 greedy decoding 生成一条最优描述
- 用该描述经 all-MiniLM-L6-v2 + TGA 提取所有训练样本的 v_hat
- 冻结所有编码器，训练 2 层 MLP 分类器
- 损失：交叉熵，学习率 1e-3，训练约 50 epochs

---

### Step 10: 消融实验

| 编号 | 配置 | 验证什么 |
|------|------|----------|
| A | Baseline: R(2+1)D 无文本 | 基线 |
| B | 前作: DistilBERT + label + Adaptive Prompting | 前作性能 |
| C | DistilBERT + label + TGA | TGA 的独立贡献 |
| D | Qwen3.5 SFT (GPT-5 mini 描述) + TGA，无 GRPO | LLM + GPT-5 mini 描述的贡献 |
| E | Qwen3.5 + 直接反传 contrastive loss（非 GRPO） | 直接反传 vs RLVR-GRPO |
| F | **完整方法**: Qwen3.5 + RLVR-GRPO (verifiable reward) | RLVR 的最终贡献 |

**文本编码器消融**（验证 Step 4 的设计决策）:

| 编号 | 文本编码器 | 验证什么 |
|------|-----------|----------|
| F1 | all-MiniLM-L6-v2（22M）| 推荐方案 |
| F2 | Frozen Qwen3.5-0.8B + EOS pooling | Causal LM 直接做编码器 |
| F3 | Frozen Qwen3.5-0.8B + EOS pooling + 可训练投影头 | 加投影头能否弥补 |
| F4 | CLIP Text Encoder（ViT-B/32）| 视觉-语言预训练编码器 |

**其他消融**:
- Group size G = {2, 4, 8, 16}
- Reward 分量消融: 仅 R1 / R1+R2 / R1+R2+R3
- 采样温度 = {0.5, 0.7, 1.0}

---

### Step 11: 对比实验

在 iMiGUE 和 SMG, MA-52 上与所有 SOTA 对比。

---

### Step 12: 可视化实验

1. **TGA Attention Map**: 不同 MG 类别的文本如何引导视觉关注不同身体部位
2. **GRPO 训练中描述质量变化**: 展示 SFT → GRPO 后，同一类别的描述如何变得更精确
3. **t-SNE**: 对比 label 特征 vs GPT-5 mini 描述特征 vs GRPO 优化后描述特征的分布
4. **混淆矩阵**: 改进前后容易混淆的类别是否改善
5. **GRPO 中高/低 reward 描述的对比**: 展示 RLVR 认为"好"和"差"的描述有什么区别
6. **Reward 曲线**: 展示 GRPO 训练过程中 R1/R2/R3 的变化趋势，证明 verifiable reward 信号稳定且持续提供优化方向

---

## 四、关键超参数

| 超参数 | 建议值 | 说明 |
|--------|--------|------|
| D_align | 384 | 投影维度（与 all-MiniLM-L6-v2 对齐） |
| τ | 0.05 | 对比损失温度 |
| LoRA r | 8 | 低秩维度 |
| LoRA alpha | 16 | 缩放因子 |
| TGA heads | 4 | 注意力头数 |
| G | 8 | GRPO group size |
| ε | 0.2 | GRPO clip range |
| β_kl | 0.04 | GRPO KL 系数 |
| 采样温度 | 0.8 | GRPO 生成多样性 |
| 最大生成长度 | 80 tokens | 描述长度 |
| SFT lr | 2e-5 | SFT LoRA 学习率 |
| GRPO lr | 1e-6 | GRPO LoRA 学习率 |
| TGA lr | 1e-4 | TGA 学习率 |
| 分类器 lr | 1e-3 | MLP 学习率 |

---

## 五、显存估算（单卡 24GB）

| 组件 | GRPO 阶段显存 |
|------|--------------|
| Qwen3.5-0.8B (LoRA, 可训练) | ~2.0 GB (fp16 + LoRA 梯度) |
| Qwen3.5-0.8B π_ref (关闭 adapter) | 0 GB (同一模型，无额外开销) |
| all-MiniLM-L6-v2 (冻结) | ~0.05 GB |
| R(2+1)D-18 (冻结) | ~0.12 GB |
| TGA (冻结) | ~0.01 GB |
| 视觉投影层 (冻结) | ~0.01 GB |
| 激活值 + batch 数据 | ~4-8 GB |
| **总计** | **~6-10 GB** |

对比原方案（使用 Qwen 冻结副本做编码器）：需额外 ~1.6 GB，且 reward 质量更差。

---

## 六、论文贡献

1. **LLM 动态描述生成**: 引入轻量 LLM 动态生成微手势描述，用 GPT-5 mini 构建高质量 SFT 数据，取代语义贫乏的类别 label，大幅增强文本模态的判别力。

2. **Text-Guided Attention**: 设计 TGA 模块，实现文本引导视觉关注关键时空区域，生成可解释的 attention map。

3. **首次将 RLVR 范式引入多模态微手势识别**: 提出以视觉-文本对比对齐质量作为 verifiable reward，结合 GRPO 优化 LLM 描述策略。与 RLHF 相比：(a) 无需训练 Reward Model，(b) 无需 Value Network，(c) reward 信号由冻结管线计算，确定性可验证，无漂移。

4. **生成器-评判器解耦设计**: 采用独立的 sentence embedding 模型作为评判空间，避免 LLM "自编自导自演"的 shortcut learning 风险，保证 reward 信号的客观性。

5. 在 iMiGUE 和 SMG MA-52 上达到新 SOTA。
