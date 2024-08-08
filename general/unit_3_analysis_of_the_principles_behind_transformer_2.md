# UNIT 3: Transformer模型原理剖析（2）

## 一、Position Encoding的概念及实现方法
### 概念
Position Encoding（位置编码）是NLP中的一种技术，用于将序列中各元素的位置信息编码嵌入到向量中。它在Transformer模型中十分重要，因为Transformer模型没有像RNN那样的顺序处理机制，因此需要额外显式的引入位置信息。

### 实现方法
实际上，Position Encoding的种类有很多，按照位置可分为：绝对位置编码（Absolute Position Encoding）、相对位置编码（Relative Position Encoding）和混合位置编码（Mixed Position Encoding），按照固定/可变可分为：可学习位置编码（Learned Position Encoding）和固定位置编码（Fixed Position Embedding）。在这里不过多赘述，仅阐述《Attention is All You Need》论文中使用的位置编码方法。后续会单独开一章来详细讲解各种位置编码的原理与实现方式。

在论文《Attention is All You Need》中，采用的就是一种基于正弦和余弦函数的绝对位置编码方法，这种编码方法是不需要额外进行学习的，位置 $pos$ 的编码由以下公式生成：

$$ \begin{aligned} PE_{(pos, 2i)} & = \sin \left(\frac{pos}{10000^{2i/d_{model}}}\right) \\ PE_{(pos, 2i+1)} & = \cos \left(\frac{pos}{10000^{2i/d_{model}}}\right) \end{aligned} $$

其中，pos 代表位置，i 代表维度，即位置编码中的每个维度都对应一个正弦曲线。选择该方法是因为它可以让模型更容易关注到相对位置的信息，以为对于任何位置偏移k， $PE_{pos+k}$都可以被表示成 $PE_{pos}$ 的线性函数。

计算方式如下：

- 为了简化计算，首先定义 $\omega = \frac{1}{10000^{2i/d_{model}}}$， 即
  
  $$\begin{aligned} PE_{(pos, 2i)} & = \sin (\omega * pos) \\ PE_{(pos, 2i + 1)} & = \cos (\omega * pos) \end{aligned}$$

- 那么
  
  $$\begin{aligned} PE_{(pos + k, 2i)} & = \sin (\omega * (pos + k)) \\ PE_{(pos + k, 2i + 1)} & = \cos (\omega * (pos +k))\end{aligned}$$

- 利用三角函数加法公式： $\sin(A+B) = \sin(A)\cos(B) + \cos(A)\sin(B)$ ； $\cos(A+B) = \cos(A)\cos(B) - \sin(A)\sin(B)$，得到
  
  $$\begin{aligned} \sin(\omega * (pos + k)) & = \sin(\omega * pos)\cos(\omega * k) + \cos(\omega * pos)\sin(\omega * k) \\
  \cos(\omega * (pos + k)) & = \cos(\omega * pos)\cos(\omega * k) - \sin(\omega * pos)\sin(\omega * k) \end{aligned}$$

- 重写可得：
  $$\begin{aligned} PE_{(pos + k, 2i)} & = PE_{(pos, 2i)}\cos(\omega * k) + PE_{(pos, 2i + 1)}\sin(\omega * k) \\
  PE_{(pos + k, 2i + 1)} & = PE_{(pos, 2i + 1)}\cos(\omega * k) - PE_{(pos, 2i)}\sin(\omega * k)\end{aligned}$$

- 观察结果：

  $PE_{pos+k}$ 的每个分量都可以表示为 $PE_{pos}$ 的两个相邻分量的线性组合，其中系数为 $\cos(\omega * k)$ 和 $\sin(\omega * k)$，仅依赖于 $i$ 和 $k$，与 $pos$ 无关。

代码实现请见 [PostionEncoder 类](../code/transformer/layers.py)
