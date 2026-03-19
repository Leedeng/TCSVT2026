# TGA 评估修复 — 方案 3

## TGA 的 Motivation 与作用

### 为什么需要 TGA

在标准的视频-文本对比学习（如 CLIP）中，视觉编码器和文本编码器各自独立地将输入映射到共享空间，然后通过全局向量的余弦相似度来对齐。这存在一个问题：

**全局池化丢失了空间定位能力。** 视觉编码器将整个视频压缩为一个全局向量 `image_emb [B, D]`，丢失了"动作发生在哪里"的空间信息。对于微手势这种局部、细粒度的动作（如"用指尖摸脖子"），全局表示容易被背景、无关肢体运动等干扰稀释。

类似的问题在图像领域已被观察到：CLIP 擅长图像级分类，但在需要空间定位的任务（目标检测、分割）上表现不佳，因为全局池化抹掉了空间细节。

### TGA 解决什么问题

TGA（Text-Guided Attention）的核心思想：**让文本告诉视觉"看哪里"。**

视觉编码器输出的不是一个全局向量，而是一组时空 token 序列 `[B, N, C]`（N 个时空位置，每个位置一个特征向量）。TGA 用文本 token 作为 Query，视觉 token 作为 Key/Value，做 Cross-Attention：

```
Q = text_tokens     （"用指尖摸脖子" 的语义）
K = visual_tokens    （视频中每个时空位置的特征）
V = visual_tokens

Attention(Q, K, V) → 文本引导下的视觉表示 v_hat
```

这样，当文本是"用指尖摸脖子"时，TGA 会自动关注视频中手指和脖子区域的 token，抑制背景和无关区域。输出的 `v_hat` 是一个**文本引导的、空间聚焦的**视觉表示。

### TGA 在整个框架中的三个作用

**1. 提升视觉-文本对齐质量（训练时）**

在对比学习中，`v_hat` 比全局 `image_emb` 更好地对齐文本，因为它聚焦在文本描述的区域。这为对比损失提供更精确的梯度信号，间接改善整个对齐空间的质量。

**2. 作为 RLVR 的 reward 计算核心（GRPO 阶段）**

在最终的 RLVR 方案中，reward R1 = cosine_sim(v_hat, t_pooled)。TGA 是 reward 管线的关键组件：
- LLM 生成一条描述 → 文本编码器编码 → TGA 引导视觉关注 → cosine_sim 评分
- TGA 此时冻结，确保 reward 信号的确定性和可验证性
- 如果没有 TGA，reward 就是全局向量的对齐度，无法反映"描述是否正确引导了视觉关注"

**3. 提供可解释的注意力图（可视化）**

TGA 输出的 attention weights `[B, L, N]` 可以映射回视频的时空网格，生成热力图，展示模型在识别不同微手势时关注的身体区域。这是论文可视化实验（Step 12）的核心素材。

### TGA 与前作 Adaptive Prompting 的区别

前作（`module_utils/prompt_utils.py` 中的 `VideoSpecificPrompt`）也做了视觉-文本的交互，但方向相反：

| | 前作 Adaptive Prompting | 本文 TGA |
|---|---|---|
| 方向 | 视觉 → 修改文本 | 文本 → 引导视觉 |
| 做法 | 用视觉特征调制文本 prompt | 用文本 token 注意视觉 token |
| 输出 | 修改后的文本表示 | 文本引导的视觉表示 v_hat |
| 可解释性 | 低（文本如何被修改难以可视化） | 高（attention map 直接展示关注区域） |
| 论文故事 | "视频适应性文本增强" | "文本引导视觉关注" → 自然衔接 RLVR（"更好的文本 → 更好的关注 → 更好的识别"） |

TGA 的方向（文本→视觉）与 RLVR 的故事线天然契合：GRPO 优化 LLM 生成更好的描述 → 更好的描述通过 TGA 引导更精准的视觉关注 → 识别性能提升。这个因果链条是论文的核心叙事。

---

## 问题

当前 `train.py` 在 `use_tga=True` 时，分类器使用 `tga_features` 作为输入：

```python
# models.py forward() L183-187
cls_input = tga_features if tga_features is not None else image_embeddings
cls_logits = self.classifier(cls_input)
```

测试时每个样本的 GT label 文本被传入 TGA，label 语义直接注入 `tga_features`，分类器不需要理解视频就能判对 → Classifier acc ≈ 100%，是信息泄露。

---

## 方案概述

**训练时**：分类器始终用纯视觉特征 `image_embeddings`，TGA 通过独立的对齐损失学习。

**评估时**：报告三个指标，从不同角度衡量 TGA 的贡献。

---

## 一、训练改动

### 1.1 分类器输入改为 image_embeddings

```python
# models.py forward() 中
# 修改前：
cls_input = tga_features if tga_features is not None else image_embeddings

# 修改后：
cls_input = image_embeddings  # 分类器始终用纯视觉特征，不经过 TGA
```

### 1.2 双对比损失：全局对齐 + TGA 引导对齐

改掉分类器输入后，TGA 完全没有梯度来源。两个现有损失都不经过 TGA：

```
L_con_global → image_projection, text_projection
L_cls        → classifier, image_projection
TGA, visual_token_proj, text_token_proj → 无梯度（白加了）
```

最自然的解决方式：**把 TGA 融入对比学习结构**，新增一个 TGA 对比损失，与全局对比损失形式完全一致：

```python
# 全局对比损失（不变）— 训练 image_projection + text_projection
L_con_global = KL(image_emb @ text_emb.T / τ, target)

# TGA 对比损失（新增）— 训练 TGA + visual_token_proj + text_token_proj
L_con_tga = KL(v_hat @ text_emb.T / τ, target)

# 分类损失（改用 image_embeddings）— 训练 classifier + image_projection
L_cls = CE(classifier(image_emb), label)
```

两个对比损失各司其职：
- `L_con_global`：全局视觉 ↔ 文本对齐（和 Exp #4 一样）
- `L_con_tga`：TGA 引导的局部视觉 ↔ 文本对齐（TGA 的训练信号）

梯度流向清晰，每个模块都有明确的梯度来源：

| 模块 | 梯度来源 |
|------|---------|
| `image_projection` | L_con_global + L_cls |
| `text_projection` | L_con_global + L_con_tga（text_emb 出现在两个对比损失中） |
| `TGA` + `visual_token_proj` + `text_token_proj` | L_con_tga |
| `classifier` | L_cls |
| `image_encoder` (backbone) | 以上所有（如果 trainable=True） |

### 1.3 Kendall 自适应加权

3 个损失 → 3 任务 Kendall：

```python
loss_weighter = AdaptiveLossWeighter(num_tasks=3)
loss = loss_weighter(L_con_global, L_con_tga, L_cls)
```

当 `use_tga=False` 时退回 2 任务，与 Exp #4 一致。

---

## 二、评估改动

### 2.1 指标 1 — Zero-shot（不用 TGA）

与现有代码完全一致，不需要改动：

```python
# 预先编码所有类别文本
label_text_emb, _ = model.encode_text(label_tokens)  # [N_classes, D]

# 对每个测试样本
image_emb, _ = model.encode_image(clip)  # [1, D]
scores = image_emb @ label_text_emb.T    # [1, N_classes]
prediction = argmax(scores)
```

**衡量**：纯视觉-文本对齐空间的质量。

### 2.2 指标 2 — Zero-shot + TGA（核心新增）

每个类别的文本作为"探针"，通过 TGA 引导视觉关注，评估对齐质量：

```python
# 预先编码所有类别文本（tokens 和 pooled）
all_text_tokens = []   # list of [1, L_c, D]
all_text_pooled = []   # list of [1, D]
for c in range(N_classes):
    tokens_c, pooled_c = model.encode_text(label_tokens_c)
    proj_tokens_c = model.text_token_proj(tokens_c)
    all_text_tokens.append(proj_tokens_c)
    all_text_pooled.append(pooled_c)

# 对每个测试样本
_, visual_tokens = model.encode_image(clip)        # [1, N, 512]
proj_visual = model.visual_token_proj(visual_tokens)  # [1, N, D]

scores = []
for c in range(N_classes):
    v_hat_c, _ = model.tga(all_text_tokens[c], proj_visual)  # [1, D]
    score_c = cosine_similarity(v_hat_c, all_text_pooled[c])
    scores.append(score_c)
prediction = argmax(scores)
```

**衡量**：TGA 在推理时直接引导注意力的增益。

### 2.3 指标 3 — Classifier（不用 TGA）

与 Exp #4 完全一致：

```python
image_emb, _ = model.encode_image(clip)
cls_logits = model.classifier(image_emb)
prediction = argmax(cls_logits)
```

**衡量**：TGA 参与训练后，纯视觉分类能力有没有变好。

---

## 三、指标对比表

| 指标 | 描述 | 用到 TGA？ | 对比基准 | 回答的问题 |
|------|------|-----------|----------|------------|
| ZS | `image_emb @ label_text_emb.T` | 否 | Exp #4 ZS (63.22%) | 有 TGA 训练后，视觉-文本对齐空间变好了吗？ |
| ZS+TGA | 每个类别过 TGA → cosine score | 是（推理时） | ZS（上一行） | TGA 引导注意力在推理时有多少增益？ |
| CLS | `classifier(image_emb)` | 否 | Exp #4 CLS (64.46%) | 纯视觉分类能力变好了吗？ |

**预期结果（如果 TGA 有效）**：
- ZS+TGA > ZS（TGA 在推理时提供直接增益）
- ZS ≥ Exp#4 ZS（TGA 训练信号间接改善了对齐空间）
- CLS ≈ Exp#4 CLS（分类器不直接受益于 TGA，但也不应退化）

---

## 四、需要修改的文件

| 文件 | 改动 |
|------|------|
| `models.py` | `forward()` 中 `cls_input` 固定为 `image_embeddings` |
| `train.py` `train_epoch()` | 新增 L_tga 损失项，调整总损失 |
| `train.py` `valid_epoch()` | 新增 ZS+TGA 评估逻辑 |
| `config.py` | （可选）新增 `lambda_tga` 超参 |

---

## 五、与最终 RLVR 方案的关系

当前阶段是消融实验 C（MiniLM + label + TGA），验证 TGA 模块的独立贡献。

在最终 RLVR 方案中：
- 阶段一（Step 7b）：TGA 通过对比损失训练 → 与本方案的训练方式一致
- 阶段二（GRPO）：TGA 冻结，reward R1 = cosine_sim(v_hat, t_pooled) → 与本方案的 ZS+TGA 指标计算方式一致
- 阶段三（分类器）：冻结 TGA，用每个类别的最优描述过 TGA 提取 v_hat → 本质上是 ZS+TGA 的分类器版本

因此本方案的 ZS+TGA 评估方式，是最终 RLVR 方案的评估方式的早期验证。
