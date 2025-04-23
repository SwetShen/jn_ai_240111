# RNN的梯度消失和梯度爆炸

## 1. 引入

DNN中存在的两个常见问题是梯度消失和梯度爆炸。这两个问题都涉及到梯度在反向传播过程中的传递和更新。分别讨论这两个问题：

### 1. 梯度消失（Vanishing Gradient）：

在深度神经网络中，特别是很深的网络中，梯度消失是一个常见问题。

这指的是在反向传播过程中，网络较深层的权重更新梯度变得非常小，甚至趋于零。这样的话，底层的权重几乎没有更新，导致网络难以学习到底层的特征。

#### 原因

在反向传播中，每一层的梯度都是通过链式法则计算得到的，梯度值是前一层梯度和权重的乘积。当这个乘积小于1时，通过多个层传递下来的梯度就会指数级地减小，最终趋近于零。

#### 解决方法

- 使用激活函数：选择合适的激活函数，如ReLU（Rectified Linear Unit），Leaky ReLU等。这些激活函数能够在一定程度上缓解梯度消失问题。
- 使用批标准化（Batch Normalization）：通过规范化输入数据，可以加速训练过程并减轻梯度消失问题。
- 使用残差连接（Residual Connections）：在网络中添加跳跃连接，允许梯度直接通过跳跃连接传播，有助于缓解梯度消失。

### 2. 梯度爆炸（Exploding Gradient）：

与梯度消失相反，梯度爆炸是指在反向传播中，网络某一层的梯度变得非常大，甚至趋于无穷。这会导致权重的更新值变得非常大，破坏网络的稳定性。

#### 原因

当网络权重初始化较大时，反向传播中的梯度也会变得较大。在网络层数较多的情况下，这些大的梯度会导致权重的更新值变得非常大。

#### 解决方法

- 权重初始化：使用适当的权重初始化方法，如Xavier初始化，可以缓解梯度爆炸问题。
  ```py
  import torch.nn.init as init
  init.xavier_normal_(self.fc.weight)
  init.xavier_uniform_(self.fc.weight)
  ```
- 梯度裁剪（Gradient Clipping）：设置一个梯度阈值，当梯度超过这个阈值时，将其裁剪为阈值，防止梯度爆炸。
  ```py
  import torch.nn.utils as utils
  # 梯度裁剪
  max_norm = 1.0  # 设置梯度裁剪的阈值
  utils.clip_grad_norm_(model.parameters(), max_norm)
  ```
- 使用梯度规范化技术：如梯度归一化（Gradient Normalization）等，通过规范化梯度来控制其大小。
  ```py
  for param in model.parameters():
    if param.grad is not None:
        # 求梯度L2范数，可以理解成梯度向量的长度
        grad_norm = torch.norm(param.grad)
        # 用梯度向量除以长度，得到单位一的向量
        param.grad = param.grad / grad_norm
  ```
在实际应用中，通常需要综合使用这些方法，根据具体情况来解决梯度消失或梯度爆炸的问题。

那么在RNN中，确实也会发生梯度消失和梯度爆炸的问题。

## 2. 基础原理

### 2.1 RNN结构

此处采用RNN的2D展开形式，这个图在前向和反向时比较直观。

可以得到隐藏状态更新公式：

$$
S_{t}=\sigma(UX_{t}+WS_{t-1})
$$

更新前向时输出计算的公式：

$$
{\hat{y}}_ {t}=softmax(VS_{t})
$$

其中：

- $X_{t}$，表示在时间步t的输入。
- $S_{t}$：t时刻隐藏层状态，通过激活函数$\sigma$（通常为sigmoid或tanh）对输入$UX_{t}$和上一个时间步的隐藏状态$WS_{t-1}$的加权和进行非线性转换。
- $\sigma$：激活函数
- $U$：$W_{ih}$，即输入到隐藏层的权重矩阵
- $V$：$W_{ho}$，即隐藏层到输出的权重矩阵
- $W$：$W_{hh}$，即时间步之间，隐藏状态到隐藏状态之间的权重矩阵
- ${\hat{y}}_ {t}$：t时刻输出,通过softmax函数对隐藏状态$VS_{t}$进行转换。

这个结构表明RNN在每个时间步都考虑当前输入和前一个时间步的隐藏状态，使其能够捕捉序列信息。而交叉熵损失函数则用于衡量模型输出与实际标签之间的差异，是常用于分类问题的损失函数。

### 2.2 损失函数

由于是分类网络，所以损失函数选择的是交叉熵损失函数。

每一个输出都可以计算其损失函数，某个时间步输出 ${\hat{y}}_ {t}$ 的的损失函数为：

$$
{L}_{t}=-y_{t}\log\hat{y}_{t}
$$

那么，某个时间步的损失为${L}_{t}$，整体损失为所有时间步的损失的之和：

$$
{L}=-\sum_{t}^{T}y_{t}\log\hat{y}_{t}
$$

t到T指的是所有时间步。

交叉熵损失函数则用于衡量模型输出与实际标签之间的差异，是常用于分类问题的损失函数。

### 2.3 反向传播

在循环神经网络（RNN）中，通过`时间展开（Backpropagation Through Time, BPTT）`计算梯度。

反向传播需要结合网络结构和简化图来分析。

这里不管DNN那条链的反向传播过程，只看时间步之间的反向传播过程。

1. 可以看到，其中第一个时间步的损失对W进行求导，即：

$$
{\frac{\partial L_{1}}{\partial W}}={\frac{\partial L_{1}}{\partial{\hat{y}}_{1}}}\cdot{\frac{\partial{\hat{y}}_{1}}{\partial S_{1}}}\cdot{\frac{\partial S_{1}}{\partial W}}
$$

第一时间步的反向传播比较容易理解，由于没有前一个时间步的隐藏状态作为输入，所以根据链式求导法则，直接使用第一个时间步损失$L_{1}$对第一个时间步${\hat{y}}_{1}$的导数 * 第一个时间步${\hat{y}}_{1}$对第一个时间步的隐藏层$S_{1}$的导数 * 第一个时间步的隐藏层$S_{1}$对$W$的导数，这个过程中的激活函数只是一个复合函数求导，不会对整体的运算过程产生影响，所以先忽略掉它。

2. 接下来是第二个时间步的反向传播：

$$
{\frac{\partial L_{2}}{\partial W}}={\frac{\partial L_{2}}{\partial{\hat{y}}_{2}}}\cdot{\frac{\partial{\hat{y}}_{2}}{\partial S_{2}}}\cdot\left({\frac{\partial S_{2}}{\partial W}}+{\frac{\partial S_{2}}{\partial S_{1}}}\cdot{\frac{\partial S_{1}}{\partial W}}\right)
$$

第二个时间步的反向传播较为复杂，对于$L_{2}$对$S_{2}$的导数比较容易，就是分路的求导，那么接下来，看括号里面的：

实际上，前向计算过程为：$S_{2}=W\cdot{S_{1}}$（忽略掉激活函数），由于${S_{1}}$也有$W$，所以不是一个常数，根据前导后不导加上后导前不导的导数特性，得到$S_{2}$对$W$求导加上$S_{2}$对$S_{1}$里面的$W$求导。于是得到以上求导公式。

3. 接下来是第三个时间步的反向传播：

$$
{\frac{\partial{L_{3}}}{\partial W}}={\frac{\partial{L_{3}}}{\partial{\hat{y}_{3}}}}\cdot{\frac{\partial{\hat{y}_{3}}}{\partial S_{3}}}\cdot\left({\frac{\partial S_{3}}{\partial W}}+{\frac{\partial S_{3}}{\partial S_{2}}}\cdot{\frac{\partial S_{2}}{\partial W}}+{\frac{\partial S_{3}}{\partial S_{2}}}\cdot{\frac{\partial S_{2}}{\partial S_{1}}}\cdot{\frac{\partial S_{1}}{\partial W}}\right)
$$

根据前面的规律，得到以上公式。

总结规律，得到任意一个损失值对W的反向传播公式为：

$$
{\frac{\partial{L_{t}}}{\partial W}}={\frac{\partial{L_{3}}}{\partial{\hat{y_{t}}}}}\cdot{\frac{\partial{\hat{y_{t}}}}{\partial W_{t}}}\cdot\sum_{i=1}^{t}{\frac{\partial S_{t}}{\partial S_{i}}}\cdot{\frac{\partial S_{i}}{\partial W}}
$$

$S_t$ 不仅受到当前时间步 $t$ 的影响，还受到之前时间步 $i$ 的影响。通过累加，确保了对所有时间步的依赖都被考虑，从而计算出相对于权重矩阵 $W$ 的总体梯度。

### 2.4 激活函数

这里展示Sigmoid激活函数和tanh激活函数.

让我们讨论一下sigmoid和tanh函数以及它们的导数的范围：

1. **Sigmoid函数：**
   
   - Sigmoid函数的表达式为：$\sigma(x) = \frac{1}{1 + e^{-x}}$
   - Sigmoid函数的导数：$\frac{d}{dx}\sigma(x) = \sigma(x)(1 - \sigma(x))$
   - 范围：Sigmoid函数的输出范围在(0, 1)之间。对于导数来说，由于它是sigmoid函数的乘积形式，其范围是在(0, 0.25]。
2. **tanh函数：**
   
   - tanh函数的表达式为：$\tanh(x) = \frac{e^{x} - e^{-x}}{e^{x} + e^{-x}}$
   - tanh函数的导数：$\frac{d\tanh(x)}{dx} = 1 - \tanh^2(x)$
   - 范围：tanh函数的输出范围在(-1, 1)之间。对于导数来说，其范围是在(0, 1]。

可以看到这两者的激活函数的导数的范围分为是：(0, 0.25]和(0, 1]。

这个部分待会就会用到。

### 2.5 梯度爆炸/梯度消失

在任意一个损失值对W的反向传播公式中，取出其中的 ${\frac{\partial S_{t}}{\partial S_{i}}}$ 进行解析。

$$
{\frac{\partial S_{t}}{\partial S_{i}}}=\prod_{k=i}^{t-1}{\frac{\partial S_{k+1}}{\partial S_{k}}}
$$

例如：$S_{k+1}=tanh(W\cdot{S_{k}})$，求 ${\frac{\partial S_{k+1}}{\partial S_{k}}}$，根据求导法则，即对 $tanh()$ 求导，并对内部的 $W\cdot{S_{k}}$ 求导，并相乘。得到：

$$
{\frac{\partial S_{k+1}}{\partial S_{k}}}={\sigma^{\prime}}\cdot{W}
$$

在上一个组件中了解到，如果激活函数是 $sigmoid()$ 的话，它的导数的范围是(0, 0.25]，那么如果$W$<4时，整体导数就会在0到1之间，由于连乘的特性，导致越来越小，越来越接近于0，那么造成了梯度消失，如果$W$很大，连乘后导数很大，造成梯度爆炸。

$tanh()$ 函数也是一样的。但是会比 $sigmoid()$ 好很多。所以默认的激活函数是 $tanh()$。


