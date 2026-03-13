# my

1. 创建并运行了测试脚本，确认：
   ZipformerBlock 测试通过：输入 (32, 100, 512) → 输出 (32, 100, 512)
   Zipformer 测试通过：输入 (32, 100, 80) → 输出 (32, 25, 256)

---

## Zipformer 论文总结

**论文标题**: Zipformer: A faster and better encoder for automatic speech recognition

**发表会议**: ICLR 2024 (Oral, Top 1.2%)

**作者团队**: 小米集团新一代 Kaldi 团队（由 Kaldi 之父 Daniel Povey 领衔）

**论文链接**: https://arxiv.org/pdf/2310.11230

**代码链接**: https://github.com/k2-fsa/icefall

### 摘要

Zipformer 是一个新型的自动语音识别 (ASR) 模型，相较于 Conformer、Squeezeformer、E-Branchformer 等主流 ASR 模型，Zipformer 具有效果更好、计算更快、更省内存等优点。在 LibriSpeech、Aishell-1 和 WenetSpeech 等常用 ASR 数据集上都取得了当前最好的实验结果。

### 核心创新点

#### 1. Downsampled Encoder Structure（降采样编码器结构）
- 采用类似 U-Net 的结构，在不同帧率上学习不同时间分辨率的时域表征
- 由 6 个 encoder stack 组成，分别在 50Hz、25Hz、12.5Hz、6.25Hz、12.5Hz 和 25Hz 的采样率下进行时域建模
- 不同 stack 的 embedding 维度不同，中间 stack 的 embedding 维度更大
- 使用 Downsample 和 Upsample 模块进行特征长度的对称放缩
- 通过 Bypass 模块以可学习的方式结合 stack 的输入和输出

#### 2. Zipformer Block（更深的块结构）
- 深度约为 Conformer block 的两倍
- 通过复用注意力权重来节省计算和内存
- 将 MHSA 模块分解为两个独立模块：
  - Multi-Head Attention Weight (MHAW)：计算注意力权重
  - Self-Attention (SA)：利用注意力权重汇聚信息
- 提出 Non-Linear Attention (NLA) 模块，充分利用已计算的注意力权重
- 使用两个 Bypass 模块结合 block 输入和中间模块的输出

#### 3. BiasNorm（偏置归一化）
- 替代传统的 LayerNorm
- 去除减均值操作，保留向量长度信息
- 使用可学习的 bias 和标量来调整激活值范围
- 避免了 LayerNorm 可能导致的参数浪费和训练不稳定问题

#### 4. Swoosh 激活函数
- 提出 SwooshR 和 SwooshL 两个新的激活函数替代 Swish
- SwooshR：用于 convolution 和 Conv-Embed 模块
- SwooshL：用于 feed-forward 和 ConvNeXt 等 "normally-off" 模块
- 对负数部分有斜率，避免 Adam-type 更新量的分母太小

#### 5. ScaledAdam 优化器
- Adam 的 parameter-scale-invariant 版本
- 根据参数 scale 放缩参数更新量，确保不同 scale 的参数相对变化一致
- 显式学习参数的 scale，提供额外的放缩参数 scale 的梯度
- 使用 Eden 学习率调度器，不需要很长的 warm-up
- 可以使用更大的学习率，加快模型收敛

#### 6. Balancer 和 Whitener（激活值限制）
- Balancer：限制激活值的最小和最大平均绝对值，以及最小和最大正数比例
- Whitener：限制协方差矩阵的特征值尽可能相同，鼓励模块学习更有信息量的输出分布
- 以省内存的方式实现：前向过程是 no-op，反向过程计算限制损失函数的梯度

### 实验结果

#### 模型规模
- Zipformer-S：小规模模型
- Zipformer-M：中等规模模型
- Zipformer-L：大规模模型

#### 数据集性能
1. **LibriSpeech** (1000小时英文)
   - Zipformer-L 取得 2.00%/4.38% 的 WER（test-clean/test-other）
   - FLOPs 比 Conformer 节省 50% 以上
   - 计算速度和内存使用显著优于其他类似参数规模的模型

2. **Aishell-1** (170小时中文)
   - Zipformer-S 性能优于 ESPnet 实现的 Conformer，参数更少
   - Zipformer-M 和 Zipformer-L 都超过了其他所有模型

3. **WenetSpeech** (10000+小时中文)
   - Zipformer-M 和 Zipformer-L 在 Test-Net 和 Test-Meeting 测试集上都超过了其他所有模型
   - Zipformer-S 的效果超过 ESPnet 和 Wenet 实现的 Conformer，参数量只有 1/3

4. **GigaSpeech** (10000小时英文)
   - 66M Zipformer-M：WER 为 10.25/10.38 (dev/test)
   - 288M Zipformer：WER 为 10.07/10.19 (dev/test)

### 消融实验结果

- **Encoder structure**：移除 Downsampled encoder structure 导致 WER 上升，证明该结构不会带来信息损失
- **Block structure**：移除 NLA 或 Bypass 模块都导致性能下降，体现了 Zipformer block 的结构优势
- **Normalization layer**：将 BiasNorm 替换为 LayerNorm 导致 WER 上升，证明 BiasNorm 的优势
- **Activation function**：全部使用 SwooshR 导致 WER 上升，证明针对不同模块使用不同激活函数的有效性
- **Optimizer**：使用 Adam 替代 ScaledAdam 导致 WER 上升，证明 ScaledAdam 的优势

### 总结

Zipformer 通过一系列创新性的改进，在保持或提升模型性能的同时，显著提高了计算效率和内存使用效率。其在多个主流 ASR 数据集上取得了 SOTA 结果，证明了其在自动语音识别任务中的优越性。该模型的设计思路对未来的 ASR 模型研究具有重要参考价值。
