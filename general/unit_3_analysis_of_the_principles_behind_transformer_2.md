# UNIT 3: Transformer模型原理剖析（2）

## 一、Positional Encoding的概念及实现方法
### 概念
Positional Encoding（位置编码）是NLP中的一种技术，用于将序列中各元素的位置信息编码嵌入到向量中。它在Transformer模型中十分重要，因为Transformer模型没有像RNN那样的顺序处理机制，因此需要额外显式的引入位置信息。

### 实现方法
实际上，Positional Encoding的种类有很多，按照位置可分为：绝对位置编码（Absolute Positional Encoding）、相对位置编码（Relative Positional Encoding）和混合位置编码（Mixed Positional Encoding），按照固定/可变可分为：可学习位置编码（Learned Positional Encoding）和固定位置编码（Fixed Positional Embedding）。在这里不过多赘述，仅阐述《Attention is All You Need》论文中使用的位置编码方法。[TODO]后续会单独开一章来详细讲解各种位置编码的原理与实现方式。

在论文《Attention is All You Need》中，采用的就是一种基于正弦和余弦函数的绝对位置编码方法，这种编码方法是不需要额外进行学习的，位置 $pos$ 的编码由以下公式生成：

$$ \begin{aligned} PE_{(pos, 2i)} & = \sin \left(\frac{pos}{10000^{2i/d_{model}}}\right) \\
 PE_{(pos, 2i+1)} & = \cos \left(\frac{pos}{10000^{2i/d_{model}}}\right) \end{aligned} $$

其中，pos 代表位置，i 代表维度，即位置编码中的每个维度都对应一个正弦曲线。选择该方法是因为它可以让模型更容易关注到相对位置的信息，以为对于任何位置偏移k， $PE_{pos+k}$都可以被表示成 $PE_{pos}$ 的线性函数。

计算方式如下：

- 为了简化计算，首先定义 $\omega = \frac{1}{10000^{2i/d_{model}}}$， 即

$$ \begin{aligned} PE_{(pos, 2i)} & = \sin (\omega * pos) \\ 
  PE_{(pos, 2i + 1)} & = \cos (\omega * pos) \end{aligned} $$

- 那么
  
$$ \begin{aligned} PE_{(pos + k, 2i)} & = \sin (\omega * (pos + k)) \\ 
  PE_{(pos + k, 2i + 1)} & = \cos (\omega * (pos +k)) \end{aligned} $$

- 利用三角函数加法公式： $\sin(A+B) = \sin(A)\cos(B) + \cos(A)\sin(B)$ ； $\cos(A+B) = \cos(A)\cos(B) - \sin(A)\sin(B)$，得到
  
$$ \begin{aligned} \sin(\omega * (pos + k)) & = \sin(\omega * pos)\cos(\omega * k) + \cos(\omega * pos)\sin(\omega * k) \\ 
  \cos(\omega * (pos + k)) & = \cos(\omega * pos)\cos(\omega * k) - \sin(\omega * pos)\sin(\omega * k) \end{aligned} $$

- 重写可得：
  
$$ \begin{aligned} PE_{(pos + k, 2i)} & = PE_{(pos, 2i)}\cos(\omega * k) + PE_{(pos, 2i + 1)}\sin(\omega * k) \\ 
  PE_{(pos + k, 2i + 1)} & = PE_{(pos, 2i + 1)}\cos(\omega * k) - PE_{(pos, 2i)}\sin(\omega * k)\end{aligned} $$

- 观察结果：

  $PE_{pos+k}$ 的每个分量都可以表示为 $PE_{pos}$ 的两个相邻分量的线性组合，其中系数为 $\cos(\omega * k)$ 和 $\sin(\omega * k)$，仅依赖于 $i$ 和 $k$，与 $pos$ 无关。

代码实现请见 [PositionalEncoding 类](../code/transformer/layers.py)

## 二、Transformer 中的Feed Forward Network
### 概念
前馈神经网络（Feed Forward Network，简称FFN）是Transformer架构中的一个关键组件，位于每个编码器和解码器的自注意力子层之后。它的主要作用是增强模型的非线性表达能力，并允许模型处理更复杂的特征交互。

### 细节
1. 结构
   - FNN 通常由两个全连接（Dense）层组成，中间有一个非线性的激活函数
   - 原始的Transformer论文中使用Relu作为激活函数，但后续的研究也探究了其他选择，如现在广泛在大语言模型中应用的Gelu等
   - FNN 的输入和输出通常都会应用残差连接（Residual Connection）和层归一化（Layer Normalization），以促进梯度流动和稳定训练（为了使代码结构更规范，一般输入部分的处理会纳入为注意力部分的输出处理模块，因此在FNN的代码编写中，仅对输出进行处理即可）
2. 数学表示
   - FFN(x) = max(0, xW1 + b1)W2 + b2
   - 其中，max(0, y1)代表relu 激活函数，x1W1 + b1为第一层，x2W2 + b2为第二层，W1、b1 为第一层的权重和偏置，W2、b2为第二层的权重和偏置
3. 维度
   - 输入维度：d_model（与注意力层的输出维度相同）
   - 中间层维度：d_hidden（通常是d_model 的4倍）
   - 输出维度：d_model（与输入维度相同，主要是为了方便模块堆叠）
4. 作用
   - 引入非线性：允许模型学习更复杂的函数和特征交互
   - 增加模型容量：通过增加参数数量，提高模型表达能力
   - 特征转换：将注意力机制捕获的上下文信息转换为更丰富的高级表示
5. 计算复杂度：FFN的计算复杂度为O(n * d_model * d_hidden),其中n是序列长度。
6. 其他变体：后续一些研究提出了在Transformer中FFN的变体，如使用门控线性单元(GLU)或引入卷积操作等，[TODO]后续会单独开一章节讲解。
   
代码实现请见 [PositionwiseFeedForward 类](../code/transformer/layers.py)

## 三、Transformer 中的LayerNormalization 的原理与重要性
### 概念
Layer Normalization（LN）是一种用于深度神经网络的正则化技术，特别是在序列模型如Transformer、RNN等结构中。它通过标准化每个样本来加速训练过程，提高模型性能，并减少内部协变量偏移（Internal Covariate Shift，ICS）。

### 计算公式/原理
1. **计算均值和方差：** 对每个样本在当前层的激活值计算均值和方差
   
   设 $x$ 为一个样本的激活值向量，包含 $N$ 个神经元的激活值。均值 $\mu$ 和方差 $\sigma^2$ 的计算公式如下：

   $$\mu = \frac{1}{N} \sum_{i=1}^{N} x_i$$

   $$\sigma^2 = \frac{1}{N} \sum_{i=1}^{N}(x_i - \mu)^2$$

2. **标准化：** 将每个激活值减去均值，在除以标准差（方差的平方根），得到标准化的激活值
   
   $$\hat{x}_i = \frac{x_i - \mu}{\sqrt{\sigma^2 + \epsilon}}$$

   其中， $\epsilon$ 是一个很小的常数，防止分母为0。

3. **缩放和平移：** 引入可学习的参数缩放因子 $\gamma$ 和平移因子 $\beta$，对标准化后的激活值进行线性变换（缩放和平移）
   
   $$\hat{y}_i = \gamma \hat{x}_i + \beta$$

   这里， $\gamma$ 和 $\beta$ 都是可学习的参数，维度与 $x$ 相同，允许模型在训练过程中学习适合的数据分布。

### 代码实现
```python
import torch
import torch.nn as nn

class LayerNormalization(nn.Module):

    def __init__(self, d_model, eps=1e-6):
      super(LayerNormalization, self).__init__()
      # 定义两个可学习的参数 gamma 和 beta
      self.gamma = nn.Parameter(torch.ones(d_model))
      self.beta  = nn.Parameter(torch.zeros(d_model))
      # eps 用于防止标准化时分母为0
      self.eps = eps

    def forward(self, x):
      # 计算均值
      mean = x.mean(-1, keepdim=True)
      # 计算标准差
      std  = x.std(-1, keepdim=True)
      # 标准化
      y = (x - mean) / (std + self.eps)
      # 线性变换
      return self.gamma * y + self.beta

# 示例用法
batch_size = 64
seq_len = 128
d_model = 512

x = torch.randn(batch_size, seq_len, d_model)
layernorm = LayerNormalization(d_model)
normalized_x = layernorm(x)
```

### 重要性
1. **加速训练过程：** Layer Normalization 能够减少内部协变量偏移，使得每一层的输入分布更稳定，从而加速模型收敛。
2. **提高模型性能：** 通过标准化激活值，Layer Normalization 可以缓解梯度消失和梯度爆炸的问题，提高模型的训练效果和泛化能力。
3. **适用性广泛：** 与批量归一化（Batch Normalization）不同，Layer Normalization 对小批量或序列数据（如RNN、Transformer）更为有效，因为它对单个样本进行标准化，不依赖于批量大小。
4. **增强模型稳定性：** 通过标准化每一层的激活值，Layer Normalization 有助于在训练过程中保持模型的稳定性，减少训练过程中的波动。

## 四、Transformer 中编码器和解码器的结构差异

### 编码器（Encoder）
编码器的主要任务是将输入序列（如源语言的句子）编码成一系列连续的表示，编码器由多个相同的编码器层组成，每个编码器层包含两个主要的子层：

1. 多头注意力（Multi-Head Attention）层：用于计算输入序列的自注意力，捕捉序列内部的依赖关系
2. 前馈神经网络（Feed-Forward Network）层：对每个位置的表示进行非线性变换

每个子层后都跟着一个残差连接和层归一化，编码器的输出是一个序列的表示，这些表示包含了输入序列的上下文信息。

### 解码器（Decoder）
解码器的主要任务是生成目标序列（如目标语言的句子）。解码器也由多个相同的解码器层组成，每个解码器层包含三个主要的子层：

1. 掩码多头注意力（Masked Multi-Head Attention）层：用于计算解码输入序列的自注意力，捕捉序列内部的依赖关系
2. 编码器-解码器注意力（Encoder-Decoder Attention）层：用于计算编码器的输出序列与解码器输入序列之间的注意力，捕捉源序列与目标序列之间的关系
3. 前馈神经网络（Feed-Forward Network）层：对每个位置的表示进行非线性变换

每个子层后都跟着一个残差连接和层归一化，解码器的输入是目标序列的掩码版本（Masked version），以确保在预测下一个单词时，解码器不会看到未来的信息。