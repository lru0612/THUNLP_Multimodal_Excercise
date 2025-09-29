# Report of THUNLP Multimodal Exercise

中国人民大学高瓴人工智能学院 卢虹宇

## Task 1

### 1. Transformer 结构中的多头注意力（Multi-Head Self-Attention）
梳理多头注意力 (Multi-Head Attention) 中数据流向：
1.  **输入隐藏状态 (hidden_states)**：维度为 $(\text{batch\_size}, \text{seq\_len}, \text{hidden\_dim})$
2.  **线性映射获得 Q, K, V**：通过线性层将 `hidden_states` 映射为 Query, Key, Value 矩阵，并切分为多头，维度变为 $(\text{batch\_size}, \text{num\_heads}, \text{seq\_len}, \text{head\_dim})$
3.  **计算注意力分数**：可按需加入 mask。
    $$
    \text{Attention} = \operatorname{softmax}\left(\frac{QK^\top}{\sqrt{\text{head\_dim}}}\right)V
    $$
4.  **头部拼接与输出**：将多头计算结果拼接并经线性映射，得到最终输出，维度恢复为 $(\text{batch\_size}, \text{seq\_len}, \text{hidden\_dim})$

###  2. 多模态特征融合

首先，图像由视觉编码器 (Visual Encoder) 处理得到`vision_hidden_states`,文本由Tokenizer处理得到`vllm_embedding`,再通过`image_bound`参数将文本图像标识符替换为`vision_hidden_state`，得到拼接完成的 `vllm_embedding`再进行后续处理。

### 3. 多模态大模型的推理预测

使用transformer库提供的generate函数即可,可以指定采样策略，如greedy,beam search等。

### 4. 多模态大模型推理效果验证

|            | ` CHAIRs` | ` CHAIRi` | `Recall` | Len  |
| ---------- | :-------: | :-------: | :------: | :--: |
| `Sampling` |           |           |          |      |
| `Greedy`   |           |           |          |      |
| ` Beam`    |           |           |          |      |

## 任务2. 多模态大模型的指令微调（Supervised Finetuning， SFT）
### 1. 指令微调数据集
仿照 `ModelPreferenceDataset` 的实现方式，构建 `SupervisedDataset`。在数据加载阶段对所有样本进行随机打乱 (shuffle)，以提升模型的泛化能力。

### 2. 训练数据预处理
按照要求按批次整理字段，并注意创建mask，input_ids和position_ids都用默认padding_value=0填充，targets用-100填充，这是因为在预处理时标签中的特殊字符被替换为-100，统一设置方便之后计算loss时忽略无效位置

### 3. 指令微调损失函数
注意logits[i]对应的是label[i+1]的输出，所以需要将两者移动对齐，并计算交叉熵损失，并注意忽略无效位置。

### 4. 指令微调训练
todo:loss图像
最初使用全参数微调，eval_loss也顺利调到了0.11以下，但编写代码评测此时模型在测试集上的回答正确率时发现模型居然输出都为空，起初以为是代码的问题，于是仔细检查了从数据预处理到模型推理再到计算损失的代码，冥思苦想了好几个小时，但都没有发现问题，训了好几轮模型依然输出为空，但观察训练集后发现sample的标签都只有yes/no，比较短小，模型很有可能为了拟合直接输出为空，这就需要控制模型调参的范围。在查找资料并查看代码后发现可以用lora来微调模型，不改变模型的原有参数而是在注意力头上添加小型的注意力矩阵，以此来防止损伤模型原有的知识结构，经过lora微调后果然模型可以正确输出并且提高了在测试机上的准确率。
在lora微调的过程中，我发现通过调参训练较少的步数eval_loss就能下降至0.11以下，但此时的模型在测试集上的准确率反而相较预训练模型有所下降，但随着训练步数增加，模型的eval_loss有小幅度上升，但在测试集上的准确率也明显上升，数据如下：
<table>
  <thead>
    <tr>
      <th style="text-align:center;">Model</th>
      <th style="text-align:center;">Batch Size</th>
      <th style="text-align:center;">Learning Rate</th>
      <th style="text-align:center;">LR Scheduler</th>
      <th style="text-align:center;">Warmup Steps</th>
      <th style="text-align:center;">Steps</th>
      <th style="text-align:center;">Eval Loss</th>
      <th style="text-align:center;">Test Accuracy</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td style="text-align:center;">Lora</td>
      <td style="text-align:center;">80</td>
      <td style="text-align:center;">4e-5</td>
      <td style="text-align:center;">cosine</td>
      <td style="text-align:center;">30</td>
      <td style="text-align:center;">60</td>
      <td style="text-align:center;">0.108</td>
      <td style="text-align:center;">0.711</td>
    </tr>
    <tr>
      <td style="text-align:center;">Lora</td>
      <td style="text-align:center;">80</td>
      <td style="text-align:center;">4e-5</td>
      <td style="text-align:center;">cosine</td>
      <td style="text-align:center;">30</td>
      <td style="text-align:center;">200</td>
      <td style="text-align:center;">0.216</td>
      <td style="text-align:center;">0.835</td>
    </tr>
    <tr>
      <td style="text-align:center;">Origin</td>
      <td style="text-align:center;">-</td>
      <td style="text-align:center;">-</td>
      <td style="text-align:center;">-</td>
      <td style="text-align:center;">-</td>
      <td style="text-align:center;">-</td>
      <td style="text-align:center;">-</td>
      <td style="text-align:center;">0.753</td>
    </tr>
  </tbody>
</table>

- **Bug 修复**： ds_config_zero2.json已经指定scheduler，导致lr_scheduler_type被覆盖，学习率不衰减。
### 5. 视觉定位能力增强
#### a. 构建VG指令微调数据集
包括：train_minicpmv_grounding.json，val_minicpmv_grounding.json和grounding_test/*.json数据集。依据 [Shikra](https://github.com/shikras/shikra/tree/main)中提供的template、config构建而来，训练集和验证集的构成详见data/prepare_grounding.py,并且根据[Qwen-VL: A Versatile Vision-Language Model for Understanding, Localization, Text Reading, and Beyond](https://arxiv.org/abs/2308.12966)将坐标和描述分别加上`<box>`和`<ref>`标签，并且将坐标归一化到1000网格大小。
```json
{
    "role": "user",
    "content": "In the given <image>, could you find and tell me the coordinates of <ref>guy wearing blue</ref>?"
},
{
    "role": "assistant",
    "content": "<box>[(273, 2), (493, 541)]</box>"
}
```
#### b.实现[finetune_grounding.sh](finetune_grounding.sh)和[mllm/train/datasets_grounding.py](mllm/train/datasets_grounding.py)

仿照其他脚本和Dataset分别实现即可

### c.仿照[Qwen-VL](https://github.com/QwenLM/Qwen-VL/blob/master/eval_mm/evaluate_grounding.py) 的指标计算方式计算模型在VG测试集上的准确率

todo: 补充准确率表格

## 任务3. 多模态大模型的偏好对齐训练
### 1. 偏好数据对数概率（log probability）计算

依旧注意对齐label和logits并忽略无效位置，对logits取softmax得到概率之后根据label索引取得`per_token_logps`再计算`average_log_prob`


### 2. 偏好优化损失函数
直观理解DPO算法就是隐式地将语言模型自身构建为一个reward模型，通过最大化偏好回答与非偏好回答的相对reward差距来进行优化。同时，KL散度约束确保了模型在微调过程中不会与初始的参考模型产生过大偏离。但正因如此，模型在优化过程中往往会同时降低好坏两个回答的生成概率，只要能拉大它们之间的差距即可。而NCA方法则关注样本的绝对价值，不再比较样本对的相对reward差距，而是判断每个回答是来自理想的对齐模型，还是来自原始的预训练模型。在成对偏好（K=2）的场景下，我们应用的便是NCA的特定损失函数。代码补全方面按照给出的改进后的公式实现loss计算即可。

DPO算法的核心思想，是隐式地将语言模型自身构建为一个奖励模型，通过最大化偏好回答与非偏好回答的相对奖励差距来进行优化。同时，KL散度约束确保了模型在微调过程中不会与初始的参考模型产生过大偏离。然而，这种机制存在一个固有的“捷径”：由于损失函数只关心奖励的相对差距，模型在KL约束下，最容易的优化路径往往是同时降低好坏两个回答的生成概率，只要能拉大它们之间的差距即可。

针对这一问题，NCA方法提出了一个根本性的转变：它不再关注相对奖励，而是将任务重塑为对每个回答的绝对价值判断——即判断该回答是源自理想的对齐模型，还是原始的预训练模型。在成对偏好（K=2）的场景下，我们应用的便是NCA的特定损失函数。因此，在代码补全部分，我们将依据该公式实现新的损失计算逻辑。

### 3. 偏好对齐训练
- **Bug 修复**：发现 `preprocess.py` 的 152 行缺少 `concatenated_attention_mask` 字段，已手动添加并修复。

todo:补充loss图以及CHAIR指标