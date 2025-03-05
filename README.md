# Deep-Learning-Notes

# The Little Book of Deep Learning

## 深度学习小书

### Francois Fleuret

#### 日内瓦大学

---

## 目录

### 第一部分：基础

1. **机器学习**
   - 1.1 从数据中学习
   - 1.2 基函数回归
   - 1.3 欠拟合与过拟合
   - 1.4 模型类别

2. **高效计算**
   - 2.1 GPU、TPU 和批处理
   - 2.2 张量

3. **训练**
   - 3.1 损失函数
   - 3.2 自回归模型
   - 3.3 梯度下降
   - 3.4 反向传播
   - 3.5 深度的价值
   - 3.6 训练协议
   - 3.7 规模的好处

### 第二部分：深度模型

4. **模型组件**
   - 4.1 层的概念
   - 4.2 线性层
   - 4.3 激活函数
   - 4.4 池化
   - 4.5 Dropout
   - 4.6 归一化层
   - 4.7 跳跃连接
   - 4.8 注意力层
   - 4.9 词嵌入
   - 4.10 位置编码

5. **架构**
   - 5.1 多层感知机
   - 5.2 卷积网络
   - 5.3 注意力模型

### 第三部分：应用

6. **预测**
   - 6.1 图像去噪
   - 6.2 图像分类
   - 6.3 目标检测
   - 6.4 语义分割
   - 6.5 语音识别
   - 6.6 文本-图像表示
   - 6.7 强化学习

7. **合成**
   - 7.1 文本生成
   - 7.2 图像生成

8. **计算分裂**
   - 8.1 提示工程
   - 8.2 量化
   - 8.3 适配器
   - 8.4 模型合并

---

## 前言

当前人工智能的进步是由 Krizhevsky 等人在 2012 年展示的，他们证明了二十年前设计的神经网络可以通过简单地扩大规模和数据集来大幅超越当时最先进的图像识别方法。这一突破得益于图形处理单元（GPU），这些高度并行的消费级计算设备最初是为实时图像合成开发的，后来被重新用于人工神经网络。

自那时起，在“深度学习”这一总称下，网络结构的创新、训练策略的改进以及专用硬件的开发使得模型的规模和训练数据的数量呈指数级增长。深度学习模型的应用范围从计算机视觉和机器人技术扩展到语音处理，并在 2020 年后推动了具有通用推理能力的大型语言模型的发展。

尽管深度学习的核心并不难理解，但它结合了线性代数、微积分、概率、优化、信号处理、编程、算法和高性能计算等多种组件，使得学习过程变得复杂。本书并不试图面面俱到，而是专注于理解几个重要模型所需的背景知识。这种简洁的方式在发布后的 12 个月内获得了超过 50 万次的下载量。

如果你不是从官方 URL 获取本书，请访问 [https://fleuret.org/public/lbdl.pdf](https://fleuret.org/public/lbdl.pdf)，以便统计读者数量。

---

**Chapter 1: Machine Learning**

### 1.1 Learning from data

The simplest use case for a model trained from data is when a signal x is accessible, for instance, the picture of a license plate, from which one wants to predict a quantity y, such as the string of characters written on the plate.

**从数据中学习**

从数据中训练模型最简单的用例是当一个信号 x 可获取时，例如车牌的图片，人们希望从中预测一个量 y，比如车牌上写的字符字符串。

---

In many real-world situations where x is a high-dimensional signal captured in an uncontrolled environment, it is too complicated to come up with an analytical recipe that relates x and y.

在许多真实世界的情境中，x 是在不可控环境中捕获的高维信号，想要找到一个将 x 和 y 关联起来的解析公式过于复杂。

---

What one can do is to collect a large training set 𝒟 of pairs (xn, yn), and devise a parametric model f. This is a piece of computer code that incorporates trainable parameters w that modulate its behavior, and such that, with the proper values w∗, it is a good predictor.

人们可以做的是收集大量的训练集 𝒟，包括 (xn, yn) 对，并设计一个参数模型 f。这是一段包含可训练参数 w 的计算机代码，这些参数调节其行为，并且通过适当的参数值 w∗，它可以成为一个良好的预测器。

---

“Good” here means that if an x is given to this piece of code, the value \( \hat{y} = f(x; w^*) \) it computes is a good estimate of the y that would have been associated with x in the training set had it been there.

这里的“良好”意味着，如果给这段代码一个 x，它计算出的值 \( \hat{y} = f(x; w^*) \) 是对 y 的一个良好估计，假如 x 出现在训练集中，y 就会与 x 关联。

---

This notion of goodness is usually formalized with a loss \( \mathcal{L}(w) \) which is small when \( f(\cdot; w) \) is good on 𝒟. Then, training the model consists of computing a value \( w^* \) that minimizes \( \mathcal{L}(w^*) \).

这种良好的概念通常通过损失函数 \( \mathcal{L}(w) \) 来形式化，当 \( f(\cdot; w) \) 在训练集 𝒟 上表现良好时，损失较小。因此，训练模型的过程是计算一个最小化 \( \mathcal{L}(w^*) \) 的值 \( w^* \)。

---

Most of the content of this book is about the definition of f, which, in realistic scenarios, is a complex combination of pre-defined sub-modules.

本书的大部分内容都围绕着 f 的定义，在现实场景中，f 是预定义子模块的复杂组合。

---

### 1.4 Categories of models

We can organize the use of machine learning models into three broad categories:

**模型类别**

我们可以将机器学习模型的使用划分为三大类：

---

- **Regression** consists of predicting a continuous-valued vector \( y \in \mathbb{R}^K \), for instance, a geometrical position of an object, given an input signal \( x \).
  
  **回归** 指的是预测一个连续值向量 \( y \in \mathbb{R}^K \)，例如，根据输入信号 \( x \) 预测物体的几何位置。

---

- **Classification** aims at predicting a value from a finite set \( \{1, ..., C\} \), for instance, the label \( y \) of an image \( x \).
  
  **分类** 旨在从一个有限集合 \( \{1, ..., C\} \) 中预测一个值，例如，给定图像 \( x \) 预测其标签 \( y \)。

---

- **Density modeling** has as its objective to model the probability density function of the data \( \mu_X \), for instance, images.
  
  **密度建模** 旨在对数据的概率密度函数 \( \mu_X \) 进行建模，例如图像数据。

---

Both regression and classification are generally referred to as **supervised learning**, since the value to be predicted must be provided as a target during training. On the contrary, density modeling is usually seen as **unsupervised learning**, as it does not require labeled data.

回归和分类通常被称为 **监督学习**，因为在训练过程中，必须提供一个目标值进行预测。相反，密度建模通常被视为 **无监督学习**，因为它不需要标注数据。

---

These three categories are not disjoint; for instance, classification can be cast as class-score regression, or discrete sequence density modeling as iterated classification.

这三类并不是相互独立的。例如，分类问题可以被视为类别分数的回归，而离散序列的密度建模可以被看作是迭代的分类问题。

---

Furthermore, they do not cover all cases. One may want to predict compounded quantities, multiple classes, or model a density conditional on a signal.

此外，这三类并不能涵盖所有情况。有时，我们可能需要预测复合量、多个类别，或者对某个信号的条件密度进行建模。

---


**Chapter 2: Efficient Computation**

### 2.1 GPUs, TPUs, and batches

From an implementation standpoint, deep learning is about executing heavy computations with large amounts of data.

**高效计算**

从实现的角度来看，深度学习涉及使用大量数据执行高强度计算。

---

The **Graphical Processing Units (GPUs)** have been instrumental in the success of the field by allowing such computations to be run on affordable hardware.

**图形处理单元（GPUs）** 在该领域的成功中发挥了关键作用，使得这些计算可以在经济实惠的硬件上运行。

---

The importance of their use, and the resulting technical constraints on the computations that can be done efficiently, force the research in the field to constantly balance mathematical soundness and implementability of novel methods.

它们的使用至关重要，并且由此产生的技术约束决定了可以高效完成的计算类型，这促使该领域的研究不断在数学严谨性和新方法的可实现性之间找到平衡。

---

Graphical Processing Units were originally designed for real-time image synthesis, which requires highly parallel architectures that happen to be well suited for deep models.

图形处理单元最初是为实时图像合成而设计的，这需要高度并行的架构，而这种架构恰好非常适合深度学习模型。

---

As their usage for AI has increased, GPUs have been equipped with dedicated **tensor cores**, and deep-learning specialized chips such as Google's **Tensor Processing Units (TPUs)** have been developed.

随着它们在人工智能中的使用增加，GPU 已经配备了专门的 **张量核心（tensor cores）**，并且开发出了专门用于深度学习的芯片，如谷歌的 **张量处理单元（TPUs）**。

---

A GPU possesses several thousand parallel units and its own fast memory. The limiting factor is usually not the number of computing units, but the **read-write operations to memory**.

GPU 具有数千个并行计算单元和专用的高速内存。通常的限制因素并不是计算单元的数量，而是 **对内存的读写操作**。

---

The slowest link is between the **CPU memory and the GPU memory**, and consequently, one should avoid copying data across devices.

最慢的环节在于 **CPU 内存与 GPU 内存之间的传输**，因此应尽量避免在设备之间复制数据。

---

Moreover, the structure of the GPU itself involves multiple levels of **cache memory**, which are smaller but faster, and computation should be organized to avoid unnecessary memory transfers between these different caches.

此外，GPU 本身的架构涉及多个层次的 **缓存内存**，这些缓存较小但速度更快，因此计算应尽量避免在这些不同的缓存之间进行不必要的数据传输。

---

This is achieved, in particular, by organizing the computation in **batches** of samples that can fit entirely in the GPU memory and are processed in parallel.

为此，计算通常采用 **批处理（batches）**，即将多个样本组合成一个批次，使其能够完全装入 GPU 内存并进行并行处理。

---

A standard GPU has a theoretical peak **performance** of **10¹³–10¹⁴ floating-point operations (FLOPs) per second**, and its memory typically ranges from **8 to 80 gigabytes**.

标准 GPU 的理论峰值 **计算性能** 约为 **10¹³–10¹⁴ 次浮点运算（FLOPs）每秒**，其内存通常在 **8 到 80 GB** 之间。

---

The standard **FP32 (32-bit floating-point encoding)** is commonly used for numerical precision, but empirical results show that using **16-bit or lower** precision does not significantly degrade performance.

标准的 **FP32（32 位浮点编码）** 常用于数值计算，但经验表明，使用 **16 位或更低** 精度不会显著降低性能。

---

We will come back to the **large size of deep architectures** in Section 3.7.

我们将在 **3.7 节** 进一步探讨 **深度架构的规模问题**。

---

### 2.2 Tensors

**张量（Tensors）**

GPUs and deep learning frameworks such as PyTorch or JAX manipulate the quantities to be processed by organizing them as **tensors**, which are series of scalars arranged along several discrete axes.

GPU 和深度学习框架（如 PyTorch 或 JAX）通过**张量（tensors）** 组织和处理数据，张量是按照多个离散轴排列的一系列标量。

---

Tensors are elements of \( \mathbb{R}^{N_1 \times \dots \times N_D} \), generalizing the notion of vectors and matrices.

张量是 \( \mathbb{R}^{N_1 \times \dots \times N_D} \) 中的元素，是向量和矩阵的广义表示。

---

Tensors are used to represent both the signals to be processed, the **trainable parameters** of the models, and the intermediate quantities they compute. The latter are called **activations**, in reference to neuronal activations.

张量用于表示待处理的信号、模型的**可训练参数**，以及计算出的中间量。后者被称为 **激活值（activations）**，类似于神经元的激活状态。

---

For instance, a time series is naturally encoded as a **T × D** tensor, where **T** is its duration and **D** is the dimension of the feature representation at every time step, often referred to as the **number of channels**.

例如，时间序列通常被编码为 **T × D** 形状的张量，其中 **T** 是持续时间，**D** 是每个时间步特征表示的维度，通常被称为 **通道数（channels）**。

---

Similarly, a 2D-structured signal can be represented as a **D × H × W** tensor, where **H** and **W** are its height and width. An RGB image would correspond to **D = 3**, but the number of channels can grow up to several thousands in large models.

类似地，二维结构化信号可以表示为 **D × H × W** 形状的张量，其中 **H** 和 **W** 分别表示高度和宽度。对于 RGB 图像，**D = 3**，但在大型模型中，通道数可能会增长到几千个。

---

**Chapter 3: Training**

### 3.1 Losses

**损失函数**

As introduced in Section 1.1, training a model consists of minimizing a loss \( \mathcal{L}(w) \) which reflects the performance of the predictor \( f(\cdot; w) \) on a training set \( \mathcal{D} \).

如 1.1 节所述，训练模型的过程是最小化损失函数 \( \mathcal{L}(w) \)，该损失函数反映了预测函数 \( f(\cdot; w) \) 在训练集 \( \mathcal{D} \) 上的性能。

---

Since models are usually extremely complex, and their performance is directly related to how well the loss is minimized, this minimization is a key challenge, which involves both computational and mathematical difficulties.

由于模型通常极为复杂，并且它们的性能直接与损失最小化的程度相关，因此这一最小化过程是一个关键挑战，涉及计算和数学上的难题。

---

The example of the **mean squared error (MSE)** from Equation (1.1) is a standard loss for predicting a continuous value.

在 1.1 节中的 **均方误差（MSE）** 是预测连续值时的一种标准损失函数。

---

For **density modeling**, the standard loss is the **likelihood of the data**. If \( f(x; w) \) is to be interpreted as a normalized log-probability or log-density, the loss is the opposite of the sum of its values over training samples, which corresponds to the likelihood of the dataset.

对于 **密度建模**，标准损失函数是 **数据的似然性（likelihood）**。如果 \( f(x; w) \) 被解释为归一化的对数概率或对数密度，则损失是其在训练样本上的值之和的相反数，对应于数据集的似然性。

---

### 3.2 Autoregressive models

**自回归模型**

A key class of methods, particularly for dealing with discrete sequences in natural language processing and computer vision, are the **autoregressive models**.

一个重要的方法类别，特别是在自然语言处理和计算机视觉中的离散序列处理，便是 **自回归模型（autoregressive models）**。

---

#### The chain rule for probabilities

**概率的链式法则**

Such models put to use the **chain rule** from probability theory:

此类模型利用了概率论中的 **链式法则（chain rule）**：

\[
P(X_1 = x_1, X_2 = x_2, ..., X_T = x_T) = P(X_1 = x_1) \times P(X_2 = x_2 | X_1 = x_1) \times ... \times P(X_T = x_T | X_1 = x_1, ..., X_{T-1} = x_{T-1}).
\]

---

Although this decomposition is valid for a random sequence of any type, it is particularly efficient when the signal of interest is a sequence of **tokens** from a finite **vocabulary**.

尽管这种分解对任何类型的随机序列都适用，但当目标信号是来自有限 **词汇表（vocabulary）** 的 **令牌（tokens）** 序列时，它尤其高效。

---

With the convention that the additional token \( \emptyset \) stands for an “unknown” quantity, we can represent the event \( \{X_1 = x_1, ..., X_t = x_t\} \) as the vector \( (x_1, ..., x_t, \emptyset, ..., \emptyset) \).

按照惯例，额外的令牌 \( \emptyset \) 代表“未知”值，我们可以将事件 \( \{X_1 = x_1, ..., X_t = x_t\} \) 表示为向量 \( (x_1, ..., x_t, \emptyset, ..., \emptyset) \)。

---

Then, a model \( f \) which, given such an input, computes a vector of logits corresponding to \( P(X_t | X_1 = x_1, ..., X_{t-1} = x_{t-1}) \), allows one to sample one token given the previous ones.

然后，模型 \( f \) 在给定此类输入时计算一组对应于 \( P(X_t | X_1 = x_1, ..., X_{t-1} = x_{t-1}) \) 的 **logits**，从而允许根据之前的令牌采样下一个令牌。

---

The chain rule ensures that by sampling \( T \) tokens \( x_t \), one at a time given the previously sampled \( x_1, ..., x_{t-1} \), we get a sequence that follows the joint distribution. This is an **autoregressive generative model**.

链式法则保证了通过逐个采样 \( T \) 个令牌 \( x_t \)，在每个步骤基于之前的 \( x_1, ..., x_{t-1} \)，可以生成符合联合分布的序列。这便是 **自回归生成模型（autoregressive generative model）**。

---

Training such a model can be done by minimizing the sum across training sequences and time steps of the **cross-entropy loss**:

训练此类模型的方法是最小化整个训练序列和时间步的 **交叉熵损失（cross-entropy loss）**：

\[
\mathcal{L}_{ce}(w) = \sum_{n=1}^{N} \sum_{t=1}^{T} -\log P(X_t = x_t | X_1 = x_1, ..., X_{t-1} = x_{t-1}; w).
\]

---

### Chapter 3.3 Gradient Descent

#### 3.3 梯度下降

Except in specific cases like the linear regression we saw in § 1.2, the optimal parameters \( w^* \) do not have a closed-form expression. In the general case, the tool of choice to minimize a function is gradient descent. It starts by initializing the parameters with a random \( w_0 \), and then improves this estimate by iterating gradient steps, each consisting of computing the gradient of the loss with respect to the parameters, and subtracting a fraction of it:

除了我们在§1.2中看到的线性回归等特定情况外，最优参数\( w^* \)通常没有闭式解。在一般情况下，最小化函数的首选工具是梯度下降。它从随机初始化参数\( w_0 \)开始，然后通过迭代梯度步骤来改进这个估计值，每一步包括计算损失函数相对于参数的梯度，并减去其中的一部分：

\[w_{n+1} = w_n - \eta \nabla \mathcal{L}_{|w}(w_n).\]

(3.1)

This procedure corresponds to moving the current estimate a bit in the direction that locally decreases \(\mathcal{L}(w)\) maximally, as illustrated in Figure 3.2.

这个过程对应于将当前估计值沿着局部最大减少\(\mathcal{L}(w)\)的方向移动一小步，如图3.2所示。

#### 知识点讲解：
- **梯度下降（Gradient Descent）**：梯度下降是一种优化算法，用于最小化损失函数。它通过计算损失函数相对于模型参数的梯度，并沿着梯度的反方向更新参数，从而逐步逼近损失函数的最小值。
- **学习率（Learning Rate）**：公式中的\(\eta\)是学习率，控制每次更新参数的步长。学习率过大可能导致优化过程不稳定，过小则可能导致收敛速度过慢。
- **随机初始化（Random Initialization）**：在开始梯度下降之前，模型参数通常会被随机初始化，以避免陷入局部最优解。

#### Learning rate

The hyper-parameter \(\eta\) is called the learning rate. It is a positive value that modulates how quickly the minimization is done, and must be chosen carefully.

超参数\(\eta\)称为学习率。它是一个正值，调节最小化的速度，必须谨慎选择。

If it is too small, the optimization will be slow at best, and may be trapped in a local minimum early. If it is too large, the optimization may bounce around a good minimum and never descend into it. As we will see in § 3.6, it can depend on the iteration number \( n \).

如果学习率太小，优化过程会非常缓慢，甚至可能过早陷入局部最小值。如果学习率太大，优化过程可能会在最小值附近震荡，无法收敛。我们将在§3.6中看到，学习率可以依赖于迭代次数\( n \)。

#### 知识点讲解：
- **学习率的选择**：学习率的选择对梯度下降的效果至关重要。过小的学习率会导致收敛速度过慢，而过大的学习率可能导致优化过程不稳定，甚至无法收敛。
- **学习率调度（Learning Rate Scheduling）**：在某些情况下，学习率会随着迭代次数的增加而逐渐减小，这有助于在优化初期快速接近最小值，而在后期精细调整参数。

#### Stochastic Gradient Descent

All the losses used in practice can be expressed as an average of a loss per small group of samples, or per sample such as:

在实际应用中，所有的损失函数都可以表示为每个小样本组或每个样本的损失的平均值，例如：

\[\mathcal{L}(w) = \frac{1}{N} \sum_{n=1}^{N} \ell_n(w),\]

where \(\ell_n(w) = L(f(x_n; w), y_n)\) for some \(L\), and the gradient is then:

其中\(\ell_n(w) = L(f(x_n; w), y_n)\)对于某个\(L\)，梯度则为：

\[\nabla \mathcal{L}_{|w}(w) = \frac{1}{N} \sum_{n=1}^{N} \nabla \ell_n |w(w)|. \tag{3.2}\]

The resulting gradient descent would compute exactly the sum in Equation 3.2, which is usually computationally heavy, and then update the parameters according to Equation 3.1. However, under reasonable assumptions of exchangeability, for instance, if the samples have been properly shuffled, any partial sum of Equation 3.2 is an unbiased estimator of the full sum, albeit noisy. So, updating the parameters from partial sums corresponds to doing more gradient steps with noisier estimates of the gradient. Due to the redundancy in the data, this happens to be a far more efficient strategy.

由此产生的梯度下降将精确计算公式3.2中的和，这通常计算量很大，然后根据公式3.1更新参数。然而，在合理的可交换性假设下，例如，如果样本已经被适当打乱，公式3.2的任何部分和都是全和的无偏估计，尽管有噪声。因此，从部分和更新参数相当于用更嘈杂的梯度估计进行更多的梯度步骤。由于数据中的冗余，这实际上是一种更高效的策略。

#### 知识点讲解：
- **随机梯度下降（Stochastic Gradient Descent, SGD）**：与传统的梯度下降不同，SGD每次只使用一个或一小部分样本来计算梯度，从而大大减少了计算量。虽然每次更新的梯度估计有噪声，但由于数据冗余，SGD仍然能够有效地收敛。
- **小批量梯度下降（Mini-batch Gradient Descent）**：SGD的一种变体，每次使用一个小批量的样本来计算梯度，既减少了噪声，又保持了较高的计算效率。

We saw in § 2.1 that processing a batch of samples small enough to fit in the computing device’s memory is generally as fast as processing a single one. Hence, the standard approach is to split the full set _Ø into batches, and to update the parameters from the estimate of the gradient computed from each. This is called mini-batch stochastic gradient descent, or stochastic gradient descent (SGD) for short.

我们在§2.1中看到，处理一个足够小的样本批次以适应计算设备的内存通常与处理单个样本一样快。因此，标准的方法是将整个数据集_Ø分成批次，并根据每个批次计算的梯度估计更新参数。这被称为小批量随机梯度下降，或简称为随机梯度下降（SGD）。

It is important to note that this process is extremely gradual, and that the number of minibatches and gradient steps are typically of the order of several million.

需要注意的是，这个过程非常渐进，小批量和梯度步骤的数量通常达到数百万次。

As with many algorithms, intuition breaks down in high dimensions, and although it may seem that this procedure would be easily trapped in a local minimum, in reality, due to the number of parameters, the design of the models, and the stochasticity of the data, its efficiency is far greater than one might expect.

与许多算法一样，直觉在高维情况下会失效，尽管看起来这个过程很容易陷入局部最小值，但实际上，由于参数的数量、模型的设计以及数据的随机性，其效率远远超出预期。

#### 知识点讲解：
- **高维优化**：在高维空间中，梯度下降的表现可能与低维空间中的直觉不同。由于参数数量庞大，模型设计复杂，梯度下降在高维空间中仍然能够有效地找到全局最小值。
- **局部最小值与全局最小值**：虽然梯度下降可能会陷入局部最小值，但在深度学习中，由于模型的复杂性和数据的随机性，梯度下降通常能够找到足够好的解，即使不是全局最小值。

Plenty of variations of this standard strategy have been proposed, and the most popular is Adam [Kingma and Ba, 2014], which keeps running estimates of the mean and variance of each component of the gradient, and normalizes them automatically, avoiding scaling issues and different training speeds in different parts of a model.

已经提出了许多这种标准策略的变体，其中最流行的是Adam [Kingma and Ba, 2014]，它保持对梯度每个分量的均值和方差的运行估计，并自动归一化它们，避免了模型不同部分的缩放问题和不同的训练速度。

#### 知识点讲解：
- **Adam优化器**：Adam是一种自适应学习率优化算法，结合了动量法和RMSProp的优点。它通过计算梯度的一阶矩（均值）和二阶矩（方差）来调整每个参数的学习率，从而在不同参数上实现更稳定的训练。
- **自适应学习率**：Adam通过自适应地调整每个参数的学习率，避免了手动调整学习率的麻烦，并且在处理稀疏梯度时表现良好。

---

### 总结：
- **梯度下降**是深度学习中用于最小化损失函数的核心优化算法。
- **学习率**控制参数更新的步长，过大或过小都会影响优化效果。
- **随机梯度下降（SGD）**通过使用小批量数据计算梯度，提高了计算效率。
- **Adam优化器**通过自适应学习率机制，进一步提升了训练的稳定性和效率。

### Chapter 3.4 Backpropagation

#### 3.4 反向传播

Using gradient descent requires a technical means to compute \(\nabla \mathcal{C}_{|w}(w)\) where \(\mathcal{C} = L(f(x;w);y)\). Given that \(f\) and \(L\) are both compositions of standard tensor operations, as for any mathematical expression, the chain rule from differential calculus allows us to get an expression of it.

使用梯度下降需要一个技术手段来计算\(\nabla \mathcal{C}_{|w}(w)\)，其中\(\mathcal{C} = L(f(x;w);y)\)。由于\(f\)和\(L\)都是由标准的张量操作组成的，就像任何数学表达式一样，微积分中的链式法则允许我们得到它的表达式。

For the sake of making notation lighter, we will not specify at which point gradients are computed, since the context makes it clear.

为了简化符号，我们将不指定在哪个点计算梯度，因为上下文已经明确了。

#### 知识点讲解：
- **反向传播（Backpropagation）**：反向传播是深度学习中用于计算损失函数相对于模型参数的梯度的算法。它通过链式法则从输出层向输入层逐层计算梯度。
- **链式法则（Chain Rule）**：链式法则是微积分中的一个基本法则，用于计算复合函数的导数。在深度学习中，链式法则用于计算损失函数相对于每一层参数的梯度。

#### Forward and backward passes

Consider the simple case of a composition of mappings:

考虑一个简单的映射组合情况：

\[ f = f^{(D)} \circ f^{(D-1)} \circ \cdots \circ f^{(1)}. \]

The output of \( f(x; w) \) can be computed by starting with \( x^{(0)} = x \) and applying iteratively:

\( f(x; w) \)的输出可以通过从\( x^{(0)} = x \)开始并迭代应用以下公式来计算：

\[ x^{(d)} = f^{(d)} \left( x^{(d-1)}; w_d \right), \]

with \( x^{(D)} \) as the final value.

其中\( x^{(D)} \)是最终值。

The individual scalar values of these intermediate results \( x^{(d)} \) are traditionally called activations in reference to neuron activations, the value \( D \) is the depth of the model, the individual mappings \( f^{(d)} \) are referred to as layers, as we will see in § 4.1, and their sequential evaluation is the forward pass (see Figure 3.3, top).

这些中间结果\( x^{(d)} \)的各个标量值传统上被称为激活值，参考神经元的激活，值\( D \)是模型的深度，各个映射\( f^{(d)} \)被称为层，我们将在§4.1中看到，它们的顺序评估是前向传播（见图3.3，顶部）。

#### 知识点讲解：
- **前向传播（Forward Pass）**：前向传播是指从输入层到输出层逐层计算每一层的激活值的过程。这是神经网络进行预测的基础。
- **激活值（Activations）**：激活值是每一层神经元的输出，通常通过激活函数（如ReLU）对线性变换的结果进行非线性变换得到。

Conversely, the gradient \( \nabla_{\ell} |_{x^{(d-1)}} \) of the loss with respect to the output \( x^{(d-1)} \) of \( f^{(d-1)} \) is the product of the gradient \( \nabla_{\ell} |_{x^{(d)}} \) with respect to the output of \( f^{(d)} \) multiplied by the Jacobian \( J_f^{(d-1)} |_x \) of \( f^{(d-1)} \) with respect to its variable \( x \). Thus, the gradients with respect to the outputs of all the \( f^{(d)} \)'s can be computed recursively backward, starting with \( \nabla_{\ell} |_{x^{(D)}} = \nabla L |_x \).

相反，损失相对于\( f^{(d-1)} \)的输出\( x^{(d-1)} \)的梯度\( \nabla_{\ell} |_{x^{(d-1)}} \)是损失相对于\( f^{(d)} \)的输出\( x^{(d)} \)的梯度\( \nabla_{\ell} |_{x^{(d)}} \)与\( f^{(d-1)} \)相对于其变量\( x \)的雅可比矩阵\( J_f^{(d-1)} |_x \)的乘积。因此，相对于所有\( f^{(d)} \)的输出的梯度可以从\( \nabla_{\ell} |_{x^{(D)}} = \nabla L |_x \)开始递归地向后计算。

#### 知识点讲解：
- **反向传播的计算**：反向传播通过链式法则从输出层向输入层逐层计算梯度。每一层的梯度是后一层梯度与该层激活函数的导数的乘积。
- **雅可比矩阵（Jacobian Matrix）**：雅可比矩阵是一个矩阵，其元素是多元函数的偏导数。在反向传播中，雅可比矩阵用于表示每一层激活函数相对于输入的导数。

And the gradient that we are interested in for training, that is \( \nabla c |_{w_d} \), is the gradient with respect to the output of \( f(d) \) multiplied by the Jacobian \( J_f (d) |_{w} \) of \( f(d) \) with respect to the parameters.

我们训练中感兴趣的梯度，即\( \nabla c |_{w_d} \)，是相对于\( f(d) \)的输出的梯度乘以\( f(d) \)相对于参数的雅可比矩阵\( J_f (d) |_{w} \)。

This iterative computation of the gradients with respect to the intermediate activations, combined with that of the gradients with respect to the layers’ parameters, is the backward pass (see Figure 3.3, bottom). The combination of this computation with the procedure of gradient descent is called backpropagation.

这种相对于中间激活值的梯度的迭代计算，结合相对于层参数的梯度的计算，就是反向传播（见图3.3，底部）。这种计算与梯度下降过程的结合被称为反向传播。

#### 知识点讲解：
- **反向传播与梯度下降的结合**：反向传播用于计算梯度，而梯度下降则利用这些梯度来更新模型参数。两者结合构成了深度学习模型训练的核心过程。

In practice, the implementation details of the forward and backward passes are hidden from programmers. Deep learning frameworks are able to automatically construct the sequence of operations to compute gradients.

在实践中，前向传播和反向传播的实现细节对程序员是隐藏的。深度学习框架能够自动构建计算梯度的操作序列。

A particularly convenient algorithm is Autograd [Baydin et al., 2015], which tracks tensor operations and builds, on the fly, the combination of operators for gradients. Thanks to this, a piece of imperative programming that manipulates tensors can automatically compute the gradient of any quantity with respect to any other.

一个特别方便的算法是Autograd [Baydin et al., 2015]，它跟踪张量操作并动态构建梯度计算的算子组合。多亏了这一点，一段操作张量的命令式编程可以自动计算任何量相对于任何其他量的梯度。

#### 知识点讲解：
- **Autograd**：Autograd是一种自动微分工具，能够自动计算复杂函数的梯度。它通过跟踪张量操作并动态构建计算图来实现这一功能。
- **自动微分（Automatic Differentiation）**：自动微分是一种计算函数导数的技术，广泛应用于深度学习框架中，用于自动计算梯度。

#### Resource usage

Regarding the computational cost, as we will see, the bulk of the computation goes into linear operations, each requiring one matrix product for the forward pass and two for the products by the Jacobians for the backward pass, making the latter roughly twice as costly as the former.

关于计算成本，正如我们将看到的，大部分计算都用于线性操作，前向传播每个操作需要一个矩阵乘积，而反向传播每个操作需要两个矩阵乘积（一个用于雅可比矩阵），使得后者的成本大约是前者的两倍。

The memory requirement during inference is roughly equal to that of the most demanding individual layer. For training, however, the backward pass requires keeping the activations computed during the forward pass to compute the Jacobians, which results in a memory usage that grows proportionally to the model’s depth. Techniques exist to trade the memory usage for computation by either relying on reversible layers [Gomez et al., 2017], or using checkpointing, which consists of storing activations for some layers only and recomputing the others on the fly with partial forward passes during the backward pass [Chen et al., 2016].

推理期间的内存需求大致等于最耗资源的单个层的内存需求。然而，在训练期间，反向传播需要保留前向传播期间计算的激活值以计算雅可比矩阵，这导致内存使用量随模型深度成比例增长。存在一些技术可以通过依赖可逆层[Gomez et al., 2017]或使用检查点（checkpointing）来在内存使用和计算之间进行权衡，检查点技术包括仅存储某些层的激活值，并在反向传播期间通过部分前向传播重新计算其他层的激活值[Chen et al., 2016]。

#### 知识点讲解：
- **内存与计算权衡**：在深度学习中，内存使用和计算成本之间存在权衡。反向传播需要存储前向传播的中间结果，这可能导致内存使用量随模型深度增加。通过使用可逆层或检查点技术，可以减少内存使用，但会增加计算成本。
- **可逆层（Reversible Layers）**：可逆层是一种特殊的设计，允许在反向传播期间通过前向传播重新计算激活值，从而减少内存使用。
- **检查点技术（Checkpointing）**：检查点技术通过存储部分层的激活值并在需要时重新计算其他层的激活值，来减少内存使用。

#### Vanishing gradient

A key historical issue when training a large network is that when the gradient propagates backwards through an operator, it may be scaled by a factor smaller than one, resulting in an exponential decrease of its magnitude. This is called the vanishing gradient, and it may make the training impossible, or, in its milder form, cause different parts of the model to be updated at different speeds, degrading their co-adaptation [Glorot and Bengio, 2010].

训练大型网络时的一个关键历史问题是，当梯度通过一个操作符向后传播时，它可能会被一个小于1的因子缩放，导致其幅值呈指数下降。这被称为梯度消失，它可能使训练无法进行，或者在其较温和的形式中，导致模型的不同部分以不同的速度更新，从而降低它们的协同适应能力[Glorot and Bengio, 2010]。

#### 知识点讲解：
- **梯度消失（Vanishing Gradient）**：梯度消失是指在深层网络中，梯度在反向传播过程中逐渐变小，导致深层网络的参数更新非常缓慢甚至停止。这是训练深层网络时的一个常见问题。
- **协同适应（Co-adaptation）**：协同适应是指网络中不同部分的参数相互适应，以达到最佳性能。梯度消失可能导致不同部分的参数更新速度不一致，从而影响网络的整体性能。

---

### 总结：
- **反向传播**是深度学习中用于计算梯度的核心算法，通过链式法则从输出层向输入层逐层计算梯度。
- **前向传播**用于计算每一层的激活值，而反向传播用于计算梯度。
- **Autograd**等自动微分工具使得梯度计算自动化，简化了深度学习模型的实现。
- **梯度消失**是训练深层网络时的一个常见问题，可以通过使用适当的激活函数和初始化方法来缓解。

由于文件内容较长，我将继续从Chapter 3.5开始翻译，并逐步完成整个PDF的翻译。以下是Chapter 3.5的翻译和知识点讲解：

---

### Chapter 3.5 The Value of Depth

#### 3.5 深度的价值

As the term "deep learning" indicates, useful models are generally compositions of long series of mappings. Training them with gradient descent results in a sophisticated co-adaptation of the mappings, even though this procedure is gradual and local.

正如“深度学习”一词所示，有用的模型通常是由一系列长映射组成的。通过梯度下降训练它们会导致这些映射之间复杂的协同适应，尽管这个过程是渐进的和局部的。

We can illustrate this behavior with a simple model \(\mathbb{R}^2 \rightarrow \mathbb{R}^2\) that combines eight layers, each multiplying its input by a \(2 \times 2\) matrix and applying Tanh per component, with a final linear classifier. This is a simplified version of the standard Multi-Layer Perceptron that we will see in § 5.1.

我们可以用一个简单的模型\(\mathbb{R}^2 \rightarrow \mathbb{R}^2\)来说明这种行为，该模型结合了八层，每层将其输入乘以一个\(2 \times 2\)矩阵，并对每个分量应用Tanh，最后是一个线性分类器。这是我们在§5.1中将要看到的标准多层感知器（MLP）的简化版本。

If we train this model with SGD and cross-entropy on a toy binary classification task (Figure 3.4, top left), the matrices co-adapt to deform the space until the classification is correct, which implies that the data have been made linearly separable before the final affine operation (Figure 3.4, bottom right).

如果我们在一个玩具二分类任务上使用SGD和交叉熵训练这个模型（图3.4，左上），矩阵会协同适应以变形空间，直到分类正确，这意味着在最后的仿射操作之前，数据已经被线性可分（图3.4，右下）。

#### 知识点讲解：
- **深度模型（Deep Models）**：深度模型由多个层次组成，每一层都对输入进行一定的变换。通过多层变换，模型可以学习到复杂的特征表示。
- **协同适应（Co-adaptation）**：在深度模型中，不同层的参数会相互适应，以共同优化模型的性能。这种协同适应使得模型能够学习到复杂的非线性关系。

Such an example gives a glimpse of what a deep model can achieve; however, it is partially misleading due to the low dimension of both the signal to process and the internal representations. Everything is kept in 2D here for the sake of visualization, but in practice, the signal to process is often of very high dimension, and the internal representations are of even higher dimension, which, in particular, facilitates the optimization by providing many degrees of freedom.

这个例子展示了深度模型可以实现的效果；然而，由于信号和内部表示的维度较低，这个例子有些误导。这里为了可视化，所有内容都保持在2D，但在实践中，要处理的信号通常具有非常高的维度，而内部表示的维度甚至更高，这尤其通过提供许多自由度来促进优化。

#### 知识点讲解：
- **高维信号（High-Dimensional Signals）**：在实际应用中，输入信号通常是高维的，如图像、文本等。深度模型通过多层变换，能够有效地处理这些高维信号。
- **自由度（Degrees of Freedom）**：高维内部表示为模型提供了更多的自由度，使得模型能够更好地拟合复杂的数据分布。

Empirical evidence accumulated over twenty years demonstrates that state-of-the-art performance across application domains necessitates models with tens of layers, such as residual networks (see § 5.2) or Transformers (see § 5.3).

二十年来积累的经验证据表明，跨应用领域的最先进性能需要具有数十层的模型，如残差网络（见§5.2）或Transformer（见§5.3）。

Theoretical results show that, for a fixed computational budget or number of parameters, increasing the depth leads to a greater complexity of the resulting mapping [Telgarsky, 2016].

理论结果表明，在固定的计算预算或参数数量的情况下，增加深度会导致生成的映射具有更大的复杂性[Telgarsky, 2016]。

#### 知识点讲解：
- **深度与模型复杂性（Depth and Model Complexity）**：增加模型的深度可以增加模型的表达能力，使其能够学习到更复杂的映射关系。然而，这也增加了训练的难度，如梯度消失和梯度爆炸问题。
- **残差网络（Residual Networks）**：残差网络通过引入跳跃连接（skip connections）来解决深度模型中的梯度消失问题，使得训练非常深的网络成为可能。

---

### Chapter 3.6 Training Protocols

#### 3.6 训练协议

Training a deep network requires defining a protocol to make the most of computation and data, and to ensure that performance will be good on new data.

训练深度网络需要定义一个协议，以充分利用计算和数据，并确保在新数据上的性能良好。

As we saw in § 1.3, the performance on the training samples may be misleading, so in the simplest setup one needs at least two sets of samples: one is a training set, used to optimize the model parameters, and the other is a test set, to evaluate the performance of the trained model.

正如我们在§1.3中看到的，训练样本上的性能可能具有误导性，因此在最简单的设置中，至少需要两组样本：一组是训练集，用于优化模型参数，另一组是测试集，用于评估训练后模型的性能。

Additionally, there are usually hyper-parameters to adapt, in particular, those related to the model architecture, the learning rate, and the regularization terms in the loss. In that case, one needs a validation set that is disjoint from both the training and test sets to assess the best configuration.

此外，通常还需要调整超参数，特别是与模型架构、学习率和损失中的正则化项相关的超参数。在这种情况下，需要一个与训练集和测试集都不重叠的验证集来评估最佳配置。

The full training is usually decomposed into epochs, each of which corresponds to going through all the training examples once. The usual dynamic of the losses is that the training loss decreases as long as the optimization runs, while the validation loss may reach a minimum after a certain number of epochs and then start to increase, reflecting an overfitting regime, as introduced in § 1.3 and illustrated in Figure 3.5.

完整的训练通常被分解为多个epoch，每个epoch对应于遍历所有训练样本一次。损失的通常动态是，只要优化运行，训练损失就会减少，而验证损失可能会在一定数量的epoch后达到最小值，然后开始增加，反映出过拟合的情况，如§1.3中介绍并在图3.5中所示。

#### 知识点讲解：
- **训练集、验证集和测试集（Training, Validation, and Test Sets）**：训练集用于训练模型，验证集用于调整超参数和选择模型，测试集用于最终评估模型性能。
- **过拟合（Overfitting）**：过拟合是指模型在训练集上表现良好，但在新数据上表现不佳的现象。通过监控验证损失，可以检测和防止过拟合。

Paradoxically, although they should suffer from severe overfitting due to their capacity, large models usually continue to improve as training progresses. This may be due to the [inductive] bias of the model becoming the main driver of performance, and the optimization process being able to find configurations that generalize well despite the large number of parameters.

矛盾的是，尽管由于它们的容量，大型模型应该遭受严重的过拟合，但它们通常随着训练的进行而继续改进。这可能是由于模型的[归纳]偏差成为性能的主要驱动力，并且优化过程能够找到尽管参数数量众多但仍能很好泛化的配置。

#### 知识点讲解：
- **归纳偏差（Inductive Bias）**：归纳偏差是指模型在学习过程中对某些假设的偏好。深度模型的归纳偏差使其能够在大量参数的情况下仍然具有良好的泛化能力。
- **泛化（Generalization）**：泛化是指模型在新数据上的表现能力。深度模型通过优化过程能够找到泛化良好的配置。

An important design choice is the learning rate schedule during training, that is, the specification of the value of the learning rate at each iteration of the gradient descent. The general policy is that the learning rate should be initially large to avoid having the optimization being trapped in a bad local minimum early, and that it should get smaller so that the optimized parameter values do not bounce around and reach a good minimum in a narrow valley of the loss landscape.

一个重要的设计选择是训练期间的学习率调度，即在梯度下降的每次迭代中指定学习率的值。一般策略是，学习率最初应该较大，以避免优化过程过早陷入不良的局部最小值，然后应该逐渐减小，以便优化的参数值不会在损失函数的狭窄谷底中震荡，并达到良好的最小值。

#### 知识点讲解：
- **学习率调度（Learning Rate Scheduling）**：学习率调度是指在训练过程中动态调整学习率。常见的学习率调度方法包括学习率衰减和余弦退火等。
- **局部最小值（Local Minima）**：局部最小值是指损失函数中的一个低点，但不是全局最低点。通过适当的学习率调度，可以避免优化过程陷入局部最小值。

The training of very large models may take months on thousands of powerful GPUs and have a financial cost of several million dollars. At this scale, the training may involve many manual interventions, informed, in particular, by the dynamics of the loss evolution.

非常大的模型的训练可能需要数千个强大的GPU花费数月时间，并且财务成本可能高达数百万美元。在这种规模下，训练可能涉及许多手动干预，特别是根据损失演变的动态进行干预。

#### 知识点讲解：
- **大规模训练（Large-Scale Training）**：训练非常大的模型需要大量的计算资源和时间。在这种情况下，训练过程通常需要人工干预，以监控和调整训练过程。

---

### Chapter 3.7 The Benefits of Scale

#### 3.7 规模的好处

There is an accumulation of empirical results showing that performance, for instance, estimated through the loss on test data, improves with the amount of data according to remarkable scaling laws, as long as the model size increases correspondingly [Kaplan et al., 2020] (see Figure 3.6).

有大量的经验结果表明，只要模型规模相应增加，性能（例如通过测试数据上的损失估计）会随着数据量的增加而提高，遵循显著的缩放定律[Kaplan et al., 2020]（见图3.6）。

Benefiting from these scaling laws in the multi-billion sample regime is possible in part thanks to the structure of deep models which can be scaled up arbitrarily, as we will see, by increasing the number of layers or feature dimensions. But it is also made possible by the distributed nature of the computation they implement, and by the stochastic gradient descent, which requires only a fraction of the data at a time and can operate with datasets whose size is orders of magnitude greater than that of the computing device’s memory. This has resulted in an exponential growth of the models, as illustrated in Figure 3.7.

在数十亿样本的规模下受益于这些缩放定律，部分归功于深度模型的结构，这些结构可以通过增加层数或特征维度来任意扩展。但也归功于它们实现的分布式计算性质，以及随机梯度下降，它每次只需要一部分数据，并且可以处理比计算设备内存大几个数量级的数据集。这导致了模型的指数增长，如图3.7所示。

#### 知识点讲解：
- **缩放定律（Scaling Laws）**：缩放定律描述了模型性能如何随着数据量和模型规模的增加而提高。深度模型能够通过增加层数和特征维度来扩展，从而受益于这些缩放定律。
- **分布式计算（Distributed Computing）**：深度模型的训练通常需要分布式计算，以处理大规模数据集和模型参数。

Typical vision models have 10–100 million trainable parameters and require \( 10^{18} \sim 10^{19} \) FLOPs for training [He et al., 2015; Sevilla et al., 2022]. Language models have from 100 million to hundreds of billions of trainable parameters and require \( 10^{20}-10^{23} \) FLOPs for training [Devlin et al., 2018; Brown et al., 2020; Chowdhery et al., 2022; Sevilla et al., 2022]. These latter models require machines with multiple high-end GPUs.

典型的视觉模型具有10到1亿个可训练参数，并且需要\( 10^{18} \sim 10^{19} \)次浮点运算（FLOPs）进行训练[He et al., 2015; Sevilla et al., 2022]。语言模型具有从1亿到数千亿个可训练参数，并且需要\( 10^{20}-10^{23} \)次浮点运算进行训练[Devlin et al., 2018; Brown et al., 2020; Chowdhery et al., 2022; Sevilla et al., 2022]。这些模型需要配备多个高端GPU的机器。

#### 知识点讲解：
- **计算需求（Computational Requirements）**：训练大规模深度模型需要大量的计算资源，特别是语言模型，其参数数量和计算需求都非常庞大。
- **GPU加速（GPU Acceleration）**：GPU由于其并行计算能力，被广泛用于加速深度模型的训练。

Training these large models is impossible using datasets with a detailed ground-truth costly to produce, which can only be of moderate size. Instead, it is done with datasets automatically produced by combining data available on the internet with minimal curation, if any. These sets may combine multiple modalities, such as text and images from web pages, or sound and images from videos, which can be used for large-scale supervised training.

使用需要高成本生成的详细标注数据集来训练这些大型模型是不可能的，这些数据集只能具有中等规模。相反，训练是通过自动生成的数据集完成的，这些数据集通过组合互联网上可用的数据并尽可能少地进行整理。这些数据集可能结合多种模态，例如来自网页的文本和图像，或来自视频的声音和图像，这些可以用于大规模监督训练。

#### 知识点讲解：
- **大规模数据集（Large-Scale Datasets）**：训练大规模模型需要大规模数据集，这些数据集通常通过自动收集和整理互联网上的数据生成。
- **多模态数据（Multimodal Data）**：多模态数据是指包含多种类型数据（如文本、图像、声音等）的数据集。这些数据可以用于训练能够处理多种输入类型的模型。

As of 2024, the most powerful models are the ones with the largest number of parameters, such as GPT-4 [Brown et al., 2020] and PaLM [Chowdhery et al., 2022], which we will see in § 5.3 and § 7.1, trained on extremely large text datasets (see Table 3.1).

截至2024年，最强大的模型是具有最多参数的模型，例如GPT-4 [Brown et al., 2020]和PaLM [Chowdhery et al., 2022]，我们将在§5.3和§7.1中看到，这些模型是在极大的文本数据集上训练的（见表3.1）。

#### 知识点讲解：
- **大规模语言模型（Large-Scale Language Models）**：大规模语言模型如GPT-4和PaLM具有数十亿甚至数千亿个参数，能够处理复杂的自然语言任务。
- **文本数据集（Text Datasets）**：这些模型通常在大规模文本数据集上进行训练，这些数据集包含来自互联网的大量文本数据。

---

### 总结：
- **深度模型**通过多层变换能够学习到复杂的特征表示，增加深度可以提高模型的表达能力。
- **训练协议**包括使用训练集、验证集和测试集来优化模型和评估性能，防止过拟合。
- **大规模训练**需要大量的计算资源和时间，通常需要分布式计算和GPU加速。
- **缩放定律**描述了模型性能如何随着数据量和模型规模的增加而提高，深度模型能够通过增加层数和特征维度来扩展。



好的，我将继续翻译接下来的章节。以下是Chapter 4的翻译和知识点讲解：

---

### Chapter 4 Model Components

#### 4 模型组件

A deep model is nothing more than a complex tensorial computation that can ultimately be decomposed into standard mathematical operations from linear algebra and analysis. Over the years, the field has developed a large collection of high-level modules with a clear semantic, and complex models combining these modules, which have proven to be effective in specific application domains.

深度模型不过是一个复杂的张量计算，最终可以分解为线性代数和分析中的标准数学操作。多年来，该领域已经开发了大量具有明确语义的高级模块，以及结合这些模块的复杂模型，这些模型在特定应用领域中被证明是有效的。

Empirical evidence and theoretical results show that greater performance is achieved with deeper architectures, that is, long compositions of mappings. As we saw in section § 3.4, training such a model is challenging due to the vanishing gradient, and multiple important technical contributions have mitigated this issue.

经验证据和理论结果表明，通过更深的架构（即长映射组合）可以实现更好的性能。正如我们在§3.4中看到的，训练这样的模型由于梯度消失问题而具有挑战性，多个重要的技术贡献已经缓解了这个问题。

#### 知识点讲解：
- **深度模型的结构**：深度模型由多个层次组成，每一层都对输入进行一定的变换。通过多层变换，模型可以学习到复杂的特征表示。
- **梯度消失问题（Vanishing Gradient Problem）**：在深层网络中，梯度在反向传播过程中逐渐变小，导致深层网络的参数更新非常缓慢甚至停止。通过使用适当的激活函数和初始化方法，可以缓解梯度消失问题。

---

### Chapter 4.1 The Notion of Layer

#### 4.1 层的概念

We call layers standard complex compounded tensor operations that have been designed and empirically identified as being generic and efficient. They often incorporate trainable parameters and correspond to a convenient level of granularity for designing and describing large deep models. The term is inherited from simple multi-layer neural networks, even though modern models may take the form of a complex graph of such modules, incorporating multiple parallel pathways.

我们将层称为标准复杂的复合张量操作，这些操作经过设计并通过经验验证为通用且高效。它们通常包含可训练参数，并对应于设计和描述大型深度模型的方便粒度级别。这个术语继承自简单的多层神经网络，尽管现代模型可能采用这种模块的复杂图形式，包含多个并行路径。

In the following pages, I try to stick to the convention for model depiction illustrated above:

在接下来的几页中，我尽量遵循上述模型描述的约定：

- operators / layers are depicted as boxes,
- darker coloring indicates that they embed trainable parameters,
- non-default valued hyper-parameters are specified in the box,
- a dashed outer frame with a multiplicative factor indicates that a group of layers is replicated in series, each with its own set of trainable parameters, if any, and
- in some cases, the dimension of their output is specified on the right when it differs from their input.

- 操作符/层被描绘为方框，
- 较深的颜色表示它们包含可训练参数，
- 非默认值的超参数在方框中指定，
- 带有乘法因子的虚线外框表示一组层被串联复制，每组层都有自己的可训练参数（如果有的话），
- 在某些情况下，当输出维度与输入维度不同时，输出维度在右侧指定。

Additionally, layers that have a complex internal structure are depicted with a greater height.

此外，具有复杂内部结构的层以更大的高度描绘。

#### 知识点讲解：
- **层的概念**：层是深度模型中的基本构建块，每个层都对输入进行一定的变换。层可以包含可训练参数，并且可以通过组合多个层来构建复杂的模型。
- **模型描述约定**：为了简化模型描述，通常使用方框表示层，较深的颜色表示包含可训练参数，虚线外框表示层的复制。

---

### Chapter 4.2 Linear Layers

#### 4.2 线性层

The most important modules in terms of computation and number of parameters are the linear layers. They benefit from decades of research and engineering in algorithmic and chip design for matrix operations.

在计算和参数数量方面，最重要的模块是线性层。它们受益于数十年来在矩阵操作的算法和芯片设计方面的研究和工程。

Note that the term "linear" in deep learning generally refers improperly to an affine operation, which is the sum of a linear expression and a constant bias.

请注意，深度学习中的“线性”一词通常不正确地指代仿射操作，即线性表达式与常数偏置的和。

#### 知识点讲解：
- **线性层（Linear Layers）**：线性层是深度模型中的基本构建块，通常指仿射变换（线性变换加上偏置）。线性层在计算上非常高效，并且在模型参数中占据主要部分。

#### Fully connected layers

The most basic linear layer is the fully connected layer, parameterized by a trainable weight matrix \( W \) of size \( D' \times D \) and bias vector \( b \) of dimension \( D' \). It implements an affine transformation generalized to arbitrary tensor shapes, where the supplementary dimensions are interpreted as vector indexes. Formally, given an input \( X \) of dimension \( D_1 \times \cdots \times D_K \times D \), it computes an output \( Y \) of dimension \( D_1 \times \cdots \times D_K \times D' \) with

最基本的线性层是全连接层，由大小为\( D' \times D \)的可训练权重矩阵\( W \)和维度为\( D' \)的偏置向量\( b \)参数化。它实现了推广到任意张量形状的仿射变换，其中附加维度被解释为向量索引。形式上，给定维度为\( D_1 \times \cdots \times D_K \times D \)的输入\( X \)，它计算维度为\( D_1 \times \cdots \times D_K \times D' \)的输出\( Y \)，其中

\[\forall d_1, \ldots, d_K,\]
\[Y[d_1, \ldots, d_K] = WX[d_1, \ldots, d_K] + b.\]

While at first sight such an affine operation may seem limited to simple transformations such as rotations, symmetries, and translations, it can in fact do more than that. In particular, projections for dimension reduction or signal filtering, but also, from the perspective of the dot product being a measure of similarity, a matrix-vector product can be interpreted as computing matching scores between the queries, as encoded by the input vectors, and keys, as encoded by the matrix rows.

虽然乍一看这种仿射操作似乎仅限于简单的变换，如旋转、对称和平移，但实际上它可以做得更多。特别是，用于降维或信号滤波的投影，而且从点积作为相似性度量的角度来看，矩阵-向量乘积可以解释为计算查询（由输入向量编码）和键（由矩阵行编码）之间的匹配分数。

#### 知识点讲解：
- **全连接层（Fully Connected Layers）**：全连接层是线性层的一种，它对输入进行仿射变换（线性变换加上偏置）。全连接层可以用于降维、信号滤波等任务。
- **点积（Dot Product）**：点积是衡量两个向量相似性的一种方法。在全连接层中，矩阵-向量乘积可以解释为计算查询和键之间的匹配分数。

As we saw in § 3.3, the gradient descent starts with the parameters’ random initialization. If this is done too naively, as seen in § 3.4, the network may suffer from exploding or vanishing activations and gradients [Glorot and Bengio, 2010]. Deep learning frameworks implement initialization methods that in particular scale the random parameters according to the dimension of the input to keep the variance of the activations constant and prevent pathological behaviors.

正如我们在§3.3中看到的，梯度下降从参数的随机初始化开始。如果这个过程过于简单，如§3.4中所示，网络可能会遭受激活值和梯度的爆炸或消失问题[Glorot and Bengio, 2010]。深度学习框架实现了初始化方法，特别是根据输入的维度缩放随机参数，以保持激活值的方差恒定并防止病态行为。

#### 知识点讲解：
- **参数初始化（Parameter Initialization）**：参数初始化是深度学习中的一个重要步骤。适当的初始化方法可以防止激活值和梯度的爆炸或消失问题，从而确保训练的稳定性。
- **激活值的方差（Variance of Activations）**：在深度网络中，保持激活值的方差恒定有助于防止梯度消失和爆炸问题。通过适当的初始化方法，可以确保激活值的方差在网络的每一层都保持稳定。

---

### Chapter 4.3 Activation Functions

#### 4.3 激活函数

If a network were combining only linear components, it would itself be a linear operator, so it is essential to have non-linear operations. These are implemented in particular with activation functions, which are layers that transform each component of the input tensor individually through a mapping, resulting in a tensor of the same shape.

如果一个网络仅组合线性组件，那么它本身将是一个线性操作符，因此必须引入非线性操作。这些操作特别是通过激活函数实现的，激活函数是通过映射逐个变换输入张量的每个分量的层，生成相同形状的张量。

There are many different activation functions, but the most used is the \textit{Rectified Linear Unit (ReLU)} [Glorot et al., 2011], which sets negative values to zero and keeps positive values unchanged (see Figure 4.5, top right):

有许多不同的激活函数，但最常用的是\textit{修正线性单元（ReLU）} [Glorot et al., 2011]，它将负值设为零并保持正值不变（见图4.5，右上）：

\[\text{relu}(x) = 
\begin{cases} 
0 & \text{if } x < 0, \\ 
x & \text{otherwise}. 
\end{cases}\]

Given that the core training strategy of deep-learning relies on the gradient, it may seem problematic to have a mapping that is not differentiable at zero and constant on half the real line. However, the main property gradient descent requires is that the gradient is informative on average. Parameter initialization and data normalization make half of the activations positive when the training starts, ensuring that this is the case.

鉴于深度学习的核心训练策略依赖于梯度，使用在零点不可导且在一半实数线上为常数的映射可能看起来有问题。然而，梯度下降所需的主要属性是梯度在平均情况下是有信息的。参数初始化和数据归一化使得训练开始时一半的激活值为正，确保了这一点。

#### 知识点讲解：
- **激活函数（Activation Functions）**：激活函数引入非线性，使得神经网络能够学习复杂的模式。ReLU是最常用的激活函数之一，它将负输入设为零，保持正输入不变。
- **ReLU的优点**：ReLU计算简单且在实践中表现良好，尽管在零点不可导，但在大多数情况下不会影响训练。

Before the generalization of ReLU, the standard activation function was the hyperbolic tangent (Tanh, see Figure 4.5, top left) which saturates exponentially fast on both the negative and positive sides, aggravating the vanishing gradient.

在ReLU普及之前，标准的激活函数是双曲正切函数（Tanh，见图4.5，左上），它在负侧和正侧都快速饱和，加剧了梯度消失问题。

#### 知识点讲解：
- **双曲正切函数（Tanh）**：Tanh激活函数将输入映射到[-1, 1]之间，但在输入较大或较小时会饱和，导致梯度消失问题。

Other popular activation functions follow the same idea of keeping positive values unchanged and squashing the negative values. Leaky ReLU [Maas et al., 2013] applies a small positive multiplier to negative values (see Figure 4.5, bottom left):

其他流行的激活函数遵循相同的思路，保持正值不变并压缩负值。Leaky ReLU [Maas et al., 2013]对负值应用一个小的正乘数（见图4.5，左下）：

\[ \text{leaky relu}(x) = \begin{cases} 
ax \text{ if } x < 0, \\ 
x \text{ otherwise.}
\end{cases} \]

And GELU [Hendrycks and Gimpel, 2016] is defined using the cumulative distribution function of the Gaussian distribution, that is:

而GELU [Hendrycks and Gimpel, 2016]使用高斯分布的累积分布函数定义，即：

\[ \text{gelu}(x) = xP(Z \leq x), \]

where \( Z \sim \mathcal{N}(0,1) \). It roughly behaves like a smooth ReLU (see Figure 4.5, bottom right).

其中\( Z \sim \mathcal{N}(0,1) \)。它的行为大致类似于平滑的ReLU（见图4.5，右下）。

#### 知识点讲解：
- **Leaky ReLU**：Leaky ReLU通过引入一个小的斜率来缓解ReLU的“死亡”问题（即某些神经元可能永远不被激活）。
- **GELU**：GELU是一种平滑的激活函数，它在输入较小时接近于零，在输入较大时接近于线性函数。GELU在某些任务中表现优于ReLU。

The choice of an activation function, in particular among the variants of ReLU, is generally driven by empirical performance.

激活函数的选择，特别是在ReLU的变体中，通常由经验性能驱动。

#### 知识点讲解：
- **激活函数的选择**：不同的激活函数在不同的任务中表现不同，通常需要通过实验来选择最适合的激活函数。

---

### 总结：
- **线性层**是深度模型中的基本构建块，通常指仿射变换（线性变换加上偏置）。
- **激活函数**引入非线性，使得神经网络能够学习复杂的模式。ReLU是最常用的激活函数之一，Leaky ReLU和GELU是其变体。
- **参数初始化**是深度学习中的一个重要步骤，适当的初始化方法可以防止激活值和梯度的爆炸或消失问题。

这些知识点是理解深度学习模型训练和优化过程的基础，掌握它们有助于更好地设计和优化深度学习模型。

---


好的，我将继续翻译接下来的章节。以下是Chapter 4.4到Chapter 4.6的翻译和知识点讲解：

---

### Chapter 4.4 Pooling

#### 4.4 池化

A classical strategy to reduce the signal size is to use a pooling operation that combines multiple activations into one that ideally summarizes the information. The most standard operation of this class is the max pooling layer, which, similarly to convolution, can operate in 1D and 2D and is defined by a kernel size.

减少信号大小的经典策略是使用池化操作，将多个激活值组合成一个理想情况下能概括信息的激活值。这类操作中最标准的是最大池化层，它与卷积类似，可以在1D和2D上操作，并由核大小定义。

In its standard form, this layer computes the maximum activation per channel, over non-overlapping sub-tensors of spatial size equal to the kernel size. These values are stored in a result tensor with the same number of channels as the input, and whose spatial size is divided by the kernel size. As with the convolution, this operator has three hyper-parameters: padding, stride, and dilation, with the stride being equal to the kernel size by default. A smaller stride results in a larger resulting tensor, following the same formula as for convolutions (see § 4.2).

在其标准形式中，该层计算每个通道的最大激活值，覆盖空间大小等于核大小的非重叠子张量。这些值存储在结果张量中，结果张量具有与输入相同的通道数，并且其空间大小除以核大小。与卷积一样，该操作符有三个超参数：填充（padding）、步幅（stride）和扩张（dilation），默认情况下步幅等于核大小。较小的步幅会导致较大的结果张量，遵循与卷积相同的公式（见§4.2）。

#### 知识点讲解：
- **池化（Pooling）**：池化操作通过将多个激活值组合成一个值来减少信号的大小。最大池化是最常用的池化操作，它选择每个区域中的最大值。
- **最大池化（Max Pooling）**：最大池化层通过选择每个区域中的最大值来减少信号的大小，同时保留最重要的特征。

The max operation can be intuitively interpreted as a logical disjunction, or, when it follows a series of convolutional layers that compute local scores for the presence of parts, as a way of encoding that at least one instance of a part is present. It loses precise location, making it invariant to local deformations.

最大操作可以直观地解释为逻辑析取，或者当它跟随一系列卷积层时，这些卷积层计算部分存在的局部分数，最大池化可以编码至少存在一个部分实例。它失去了精确的位置信息，使其对局部变形具有不变性。

#### 知识点讲解：
- **池化的不变性（Invariance of Pooling）**：池化操作通过选择区域中的最大值，使得模型对局部变形具有不变性。这种不变性有助于模型在处理图像等数据时具有更好的鲁棒性。

A standard alternative is the \underline{average pooling} layer that computes the average instead of the maximum over the sub-tensors. This is a linear operation, whereas max pooling is not.

一个标准的替代方案是\underline{平均池化}层，它计算子张量的平均值而不是最大值。这是一个线性操作，而最大池化不是。

#### 知识点讲解：
- **平均池化（Average Pooling）**：平均池化层通过计算每个区域的平均值来减少信号的大小。与最大池化不同，平均池化是一个线性操作，适用于某些需要平滑特征的任务。

---

### Chapter 4.5 Dropout

#### 4.5 Dropout

Some layers have been designed to explicitly facilitate training or improve the learned representations.

一些层被设计用来显式地促进训练或改进学习到的表示。

One of the main contributions of that sort was dropout [Srivastava et al., 2014]. Such a layer has no trainable parameters, but one hyperparameter, \( p \), and takes as input a tensor of arbitrary shape.

这类层的主要贡献之一是dropout [Srivastava et al., 2014]。这样的层没有可训练参数，但有一个超参数\( p \)，并接受任意形状的张量作为输入。

It is usually switched off during testing, in which case its output is equal to its input. When it is active, it has a probability \( p \) of setting to zero each activation of the input tensor independently, and it re-scales all the activations by a factor of \(\frac{1}{1-p}\) to maintain the expected value unchanged (see Figure 4.7).

在测试期间通常关闭它，在这种情况下，其输出等于输入。当它激活时，它以概率\( p \)独立地将输入张量的每个激活值设为零，并通过乘以\(\frac{1}{1-p}\)来重新缩放所有激活值，以保持期望值不变（见图4.7）。

#### 知识点讲解：
- **Dropout**：Dropout是一种正则化技术，通过在训练过程中随机丢弃一部分神经元来防止过拟合。在测试时，Dropout层被关闭，所有神经元都参与计算。
- **Dropout的作用**：Dropout通过随机丢弃神经元，使得模型不能依赖于某些特定的神经元，从而提高了模型的泛化能力。

The motivation behind dropout is to favor meaningful individual activation and discourage group representation. Since the probability that a group of \( k \) activations remains intact through a dropout layer is \((1 - p)^k\), joint representations become unreliable, making the training procedure avoid them. It can also be seen as a noise injection that makes the training more robust.

Dropout的动机是鼓励有意义的个体激活并抑制群体表示。由于一组\( k \)个激活值通过Dropout层保持完整的概率是\((1 - p)^k\)，联合表示变得不可靠，使得训练过程避免它们。它也可以被视为一种噪声注入，使训练更加鲁棒。

#### 知识点讲解：
- **Dropout的动机**：Dropout通过随机丢弃神经元，鼓励模型学习到更加鲁棒的特征表示，避免过度依赖某些特定的神经元组合。

When dealing with images and 2D tensors, the short-term correlation of the signals and the resulting redundancy negate the effect of dropout, since activations set to zero can be inferred from their neighbors. Hence, dropout for \( 2D \) tensors sets entire channels to zero instead of individual activations (see Figure 4.8).

在处理图像和2D张量时，信号的短期相关性和由此产生的冗余抵消了Dropout的效果，因为被设为零的激活值可以从其邻居推断出来。因此，对于\( 2D \)张量的Dropout将整个通道设为零，而不是单个激活值（见图4.8）。

#### 知识点讲解：
- **2D Dropout**：在处理图像等2D数据时，Dropout通常作用于整个通道，而不是单个像素。这是因为图像中的像素之间存在较强的相关性，单个像素的Dropout效果不明显。

Although dropout is generally used to improve training and is inactive during inference, it can be used in certain setups as a randomization strategy, for instance, to estimate empirically confidence scores [Gal and Ghahramani, 2015].

尽管Dropout通常用于改进训练并在推理期间不活动，但它可以在某些设置中用作随机化策略，例如，用于经验估计置信度分数[Gal and Ghahramani, 2015]。

#### 知识点讲解：
- **Dropout的其他用途**：除了防止过拟合，Dropout还可以用于估计模型的不确定性或生成多样化的输出。

---

### Chapter 4.6 Normalizing Layers

#### 4.6 归一化层

An important class of operators to facilitate the training of deep architectures are the normalizing layers, which force the empirical mean and variance of groups of activations.

促进深度架构训练的一类重要操作符是归一化层，它们强制激活值组的经验均值和方差。

The main layer in that family is batch normalization [Ioffe and Szegedy, 2015], which is the only standard layer to process batches instead of individual samples. It is parameterized by a hyper-parameter \( D \) and two series of trainable scalar parameters \(\beta_1, \ldots, \beta_D\) and \(\gamma_1, \ldots, \gamma_D\).

该家族中的主要层是批归一化（Batch Normalization）[Ioffe and Szegedy, 2015]，它是唯一处理批次而不是单个样本的标准层。它由一个超参数\( D \)和两个可训练标量参数系列\(\beta_1, \ldots, \beta_D\)和\(\gamma_1, \ldots, \gamma_D\)参数化。

Given a batch of \( B \) samples \( x_1, \ldots, x_B \) of dimension \( D \), it first computes for each of the \( D \) components an empirical mean \(\hat{m}_d\) and variance \(\hat{v}_d\) across the batch:

给定一批维度为\( D \)的\( B \)个样本\( x_1, \ldots, x_B \)，它首先为每个\( D \)分量计算批次上的经验均值\(\hat{m}_d\)和方差\(\hat{v}_d\)：

\[\hat{m}_d = \frac{1}{B} \sum_{b=1}^{B} x_b,d\]

\[\hat{v}_d = \frac{1}{B} \sum_{b=1}^{B} (x_{b,d} - \hat{m}_d)^2,\]

from which it computes for every component \( x_{b,d} \) a normalized value \( z_{b,d} \), with empirical mean 0 and variance 1, and from it the final result value \( y_{b,d} \) with mean \(\beta_d\) and standard de-

然后它为每个分量\( x_{b,d} \)计算一个归一化值\( z_{b,d} \)，其经验均值为0，方差为1，并从中计算最终结果值\( y_{b,d} \)，其均值为\(\beta_d\)，标准差为\(\gamma_d\)：

\[\forall b, \quad z_{b,d} = \frac{x_{b,d} - \hat{m}_d}{\sqrt{\hat{v}_d + \epsilon}}\]

\[y_{b,d} = \gamma_d z_{b,d} + \beta_d.\]

Because this normalization is defined across a batch, it is done only during training. During testing, the layer transforms individual samples according to the \(\hat{m}_d s\) and \(\hat{v}_d s\) estimated with a moving average over the full training set, which boils down to a fixed affine transformation per component.

由于这种归一化是在批次上定义的，因此仅在训练期间进行。在测试期间，该层根据在整个训练集上估计的\(\hat{m}_d s\)和\(\hat{v}_d s\)对单个样本进行变换，这归结为每个分量的固定仿射变换。

#### 知识点讲解：
- **批归一化（Batch Normalization）**：批归一化通过在训练期间对每个批次进行归一化，使得每一层的输入分布更加稳定，从而加速训练并提高模型的性能。
- **归一化的作用**：归一化层通过强制激活值的均值和方差，使得每一层的输入分布更加稳定，从而缓解梯度消失和梯度爆炸问题。

The motivation behind batch normalization was to avoid that a change in scaling in an early layer of the network during training impacts all the layers that follow, which then have to adapt their trainable parameters accordingly. Although the actual mode of action may be more complicated than this initial motivation, this layer considerably facilitates the training of deep models.

批归一化的动机是避免训练期间网络早期层的缩放变化影响所有后续层，这些层随后必须相应地调整其可训练参数。尽管实际作用模式可能比这个初始动机更复杂，但该层大大促进了深度模型的训练。

#### 知识点讲解：
- **批归一化的动机**：批归一化通过稳定每一层的输入分布，使得模型在训练过程中更加稳定，从而加速收敛并提高性能。

In the case of \(2D\) tensors, to follow the principle of convolutional layers of processing all locations similarly, the normalization is done per-channel across all \(2D\) positions, and \(\beta\) and \(\gamma\) remain vectors of dimension \(D\) so that the scaling/shift does not depend on the \(2D\) position. Hence, if the tensor to be processed is of shape \( B \times D \times H \times W \), the layer computes \((m_d, v_d)\), for \( d = 1, \ldots, D \) from the corresponding \( B \times H \times W \) slice, normalizes it accordingly, and finally scales and shifts its components with the trainable parameters \(\beta_d\) and \(\gamma_d\).

在处理\(2D\)张量时，为了遵循卷积层对所有位置进行类似处理的原则，归一化是在所有\(2D\)位置上按通道进行的，\(\beta\)和\(\gamma\)保持为维度\(D\)的向量，因此缩放/平移不依赖于\(2D\)位置。因此，如果要处理的张量形状为\( B \times D \times H \times W \)，则该层从相应的\( B \times H \times W \)切片计算\((m_d, v_d)\)，对其进行归一化，并最终使用可训练参数\(\beta_d\)和\(\gamma_d\)缩放和平移其分量。

#### 知识点讲解：
- **2D批归一化**：在处理图像等2D数据时，批归一化通常按通道进行，即对每个通道的所有位置进行归一化。

So, given a \( B \times D \) tensor, batch normalization normalizes it across \( b \) and scales/shifts it according to \( d \), which can be implemented as a component-wise product by \(\gamma\) and a sum with \(\beta\). Given a \( B \times D \times H \times W \) tensor, it normalizes across \( b, h, w \) and scales/shifts according to \( d \) (see Figure 4.9, left).

因此，给定一个\( B \times D \)张量，批归一化在\( b \)上进行归一化，并根据\( d \)进行缩放/平移，这可以通过\(\gamma\)的逐分量乘积和与\(\beta\)的和来实现。给定一个\( B \times D \times H \times W \)张量，它在\( b, h, w \)上进行归一化，并根据\( d \)进行缩放/平移（见图4.9，左）。

This can be generalized depending on these dimensions. For instance, layer normalization [Ba et al., 2016] computes moments and normalizes across all components of individual samples, and scales and shifts components individually (see Figure 4.9, right). So, given a \( B \times D \) tensor, it normalizes across \( d \) and scales/shifts also according to the same. Given a \( B \times D \times H \times W \) tensor, it normalizes it across \( d, h, w \) and scales/shifts according to the same.

这可以根据这些维度进行推广。例如，层归一化（Layer Normalization）[Ba et al., 2016]计算矩并在单个样本的所有分量上进行归一化，并单独缩放和平移分量（见图4.9，右）。因此，给定一个\( B \times D \)张量，它在\( d \)上进行归一化，并根据相同的\( d \)进行缩放/平移。给定一个\( B \times D \times H \times W \)张量，它在\( d, h, w \)上进行归一化，并根据相同的\( d, h, w \)进行缩放/平移。

#### 知识点讲解：
- **层归一化（Layer Normalization）**：层归一化在单个样本的所有分量上进行归一化，适用于处理序列数据（如文本）或小批量数据。

Contrary to batch normalization, since it processes samples individually, layer normalization behaves the same during training and testing.

与批归一化不同，由于层归一化逐个处理样本，因此在训练和测试期间行为相同。

#### 知识点讲解：
- **层归一化的优点**：层归一化不依赖于批次大小，因此在处理小批量数据或序列数据时表现更好。

---

### 总结：
- **池化**通过将多个激活值组合成一个值来减少信号的大小，最大池化和平均池化是常用的池化操作。
- **Dropout**通过在训练过程中随机丢弃一部分神经元来防止过拟合，提高模型的泛化能力。
- **归一化层**通过强制激活值的均值和方差，使得每一层的输入分布更加稳定，从而加速训练并提高模型的性能。批归一化和层归一化是常用的归一化方法。

这些知识点是理解深度学习模型训练和优化过程的基础，掌握它们有助于更好地设计和优化深度学习模型。

---

好的，我将继续翻译接下来的章节。以下是Chapter 4.7到Chapter 4.10的翻译和知识点讲解：

---

### Chapter 4.7 Skip Connections

#### 4.7 跳跃连接

Another technique that mitigates the vanishing gradient and allows the training of deep architectures are skip connections [Long et al., 2014; Ronneberger et al., 2015]. They are not layers per se, but an architectural design in which outputs of some layers are transported as-is to other layers further in the model, bypassing processing in between. This unmodified signal can be concatenated or added to the input of the layer the connection branches into (see Figure 4.10). A particular type of skip connections are the residual connections which combine the signal with a sum, and usually skip only a few layers (see Figure 4.10, right).

另一种缓解梯度消失并允许训练深度架构的技术是跳跃连接（Skip Connections）[Long et al., 2014; Ronneberger et al., 2015]。它们本身不是层，而是一种架构设计，其中某些层的输出被原封不动地传输到模型中更远的层，绕过中间的处理。这个未修改的信号可以连接到分支进入的层的输入（见图4.10）。一种特殊类型的跳跃连接是残差连接（Residual Connections），它们通过求和将信号组合在一起，通常只跳过几层（见图4.10，右）。

The most desirable property of this design is to ensure that, even in the case of gradient-killing processing at a certain stage, the gradient will still propagate through the skip connections. Residual connections, in particular, allow for the building of deep models with up to several hundred layers, and key models, such as the residual networks [He et al., 2015] in computer vision (see § 5.2), and the Transformers [Vaswani et al., 2017] in natural language processing (see § 5.3), are entirely composed of blocks of layers with residual connections.

这种设计的最理想特性是确保即使在某个阶段存在梯度消失的处理，梯度仍然可以通过跳跃连接传播。特别是残差连接，允许构建具有多达数百层的深度模型，关键模型如计算机视觉中的残差网络（Residual Networks）[He et al., 2015]（见§5.2）和自然语言处理中的Transformer [Vaswani et al., 2017]（见§5.3），完全由带有残差连接的层块组成。

#### 知识点讲解：
- **跳跃连接（Skip Connections）**：跳跃连接通过将某些层的输出直接传递到更远的层，绕过中间的处理，从而缓解梯度消失问题。
- **残差连接（Residual Connections）**：残差连接是一种特殊的跳跃连接，通过将输入与输出相加，使得梯度可以直接传播，从而允许训练非常深的网络。

Their role can also be to facilitate multi-scale reasoning in models that reduce the signal size before re-expanding it, by connecting layers with compatible sizes, for instance for semantic segmentation (see § 6.4). In the case of residual connections, they may also facilitate learning by simplifying the task to finding a differential improvement instead of a full update.

它们的作用还可以通过在模型中将信号大小缩小后再重新扩展时连接具有兼容大小的层，来促进多尺度推理，例如用于语义分割（见§6.4）。在残差连接的情况下，它们还可以通过将任务简化为寻找差分改进而不是完全更新来促进学习。

#### 知识点讲解：
- **多尺度推理（Multi-scale Reasoning）**：跳跃连接可以用于多尺度推理，特别是在图像分割等任务中，通过连接不同尺度的特征图，模型可以更好地捕捉全局和局部信息。

---

### Chapter 4.8 Attention Layers

#### 4.8 注意力层

In many applications, there is a need for an operation able to combine local information at locations far apart in a tensor. For instance, this could be distant details for coherent and realistic image synthesis, or words at different positions in a paragraph to make a grammatical or semantic decision in Natural Language Processing.

在许多应用中，需要一种操作能够组合张量中相距较远的局部信息。例如，这可能是用于连贯和逼真的图像合成的远处细节，或者是自然语言处理中用于做出语法或语义决策的段落中不同位置的单词。

Fully connected layers cannot process large-dimension signals, nor signals of variable size, and \underline{convolutional} layers are not able to propagate information quickly. Strategies that aggregate the results of convolutions, for instance, by averaging them over large spatial areas, suffer from mixing multiple signals into a limited number of dimensions.

全连接层无法处理大维度信号，也无法处理可变大小的信号，而\underline{卷积}层无法快速传播信息。通过在大空间区域上平均卷积结果等策略，会将多个信号混合到有限数量的维度中。

Attention layers specifically address this problem by computing an attention score for each component of the resulting tensor to each component of the input tensor, without locality constraints, and averaging the features across the full tensor accordingly [Vaswani et al., 2017].

注意力层通过为结果张量的每个分量计算与输入张量每个分量的注意力分数，专门解决了这个问题，没有局部性约束，并相应地平均整个张量上的特征[Vaswani et al., 2017]。

#### 知识点讲解：
- **注意力机制（Attention Mechanism）**：注意力机制通过计算输入张量中每个分量与输出张量中每个分量的相关性，使得模型能够捕捉长距离依赖关系。
- **注意力层的应用**：注意力层广泛应用于自然语言处理和图像生成等任务中，特别是在Transformer模型中。

Even though they are substantially more complicated than other layers, they have become a standard element in many recent models. They are, in particular, the key building block of \underline{Transformers}, the dominant architecture for Large Language Models. See § 5.3 and § 7.1.

尽管它们比其他层复杂得多，但它们已成为许多最新模型中的标准元素。特别是，它们是\underline{Transformer}的关键构建块，Transformer是大型语言模型的主导架构。参见§5.3和§7.1。

#### 知识点讲解：
- **Transformer模型**：Transformer模型通过自注意力机制（Self-Attention）和多头注意力机制（Multi-Head Attention）实现了强大的序列建模能力，广泛应用于自然语言处理任务中。

---

### Chapter 4.9 Token Embedding

#### 4.9 词嵌入

In many situations, we need to convert discrete tokens into vectors. This can be done with an embedding layer, which consists of a lookup table that directly maps integers to vectors.

在许多情况下，我们需要将离散的标记（tokens）转换为向量。这可以通过嵌入层（Embedding Layer）实现，嵌入层由一个查找表组成，直接将整数映射到向量。

Such a layer is defined by two hyper-parameters: the number \( N \) of possible token values, and the dimension \( D \) of the output vectors, and one trainable \( N \times D \) weight matrix \( M \).

这样的层由两个超参数定义：可能的标记值数量\( N \)和输出向量的维度\( D \)，以及一个可训练的\( N \times D \)权重矩阵\( M \)。

Given as input an integer tensor \( X \) of dimension \( D_1 \times \cdots \times D_K \) and values in \(\{0, \ldots, N - 1\}\) such a layer returns a real-valued tensor \( Y \) of dimension \( D_1 \times \cdots \times D_K \times D \) with

给定一个维度为\( D_1 \times \cdots \times D_K \)且值在\(\{0, \ldots, N - 1\}\)中的整数张量\( X \)，这样的层返回一个维度为\( D_1 \times \cdots \times D_K \times D \)的实值张量\( Y \)，其中

\[\forall d_1, \ldots, d_K,\]
\[Y[d_1, \ldots, d_K] = M[X[d_1, \ldots, d_K]].\]

#### 知识点讲解：
- **词嵌入（Token Embedding）**：词嵌入层将离散的标记（如单词或字符）映射到连续的向量空间中，使得模型能够处理文本数据。
- **嵌入层的应用**：嵌入层广泛应用于自然语言处理任务中，特别是在处理文本数据时，将单词或字符转换为向量表示。

---

### Chapter 4.10 Positional Encoding

#### 4.10 位置编码

While the processing of a fully connected layer is specific to both the positions of the features in the input tensor and to the positions of the resulting activations in the output tensor, convolutional layers and Multi-Head Attention layers are oblivious to the absolute position in the tensor. This is key to their strong invariance and inductive bias, which is beneficial for dealing with a stationary signal.

虽然全连接层的处理特定于输入张量中特征的位置和输出张量中结果激活值的位置，但卷积层和多头注意力层对张量中的绝对位置不敏感。这是它们强不变性和归纳偏差的关键，这对于处理平稳信号是有益的。

However, this can be an issue in certain situations where proper processing has to access the absolute positioning. This is the case, for instance, for image synthesis, where the statistics of a scene are not totally stationary, or in natural language processing, where the relative positions of words strongly modulate the meaning of a sentence.

然而，在某些情况下，这可能是一个问题，因为适当的处理必须访问绝对位置。例如，在图像合成中，场景的统计特性并不完全平稳，或者在自然语言处理中，单词的相对位置强烈调节句子的含义。

The standard way of coping with this problem is to add or concatenate to the feature representation, at every position, a positional encoding, which is a feature vector that depends on the position in the tensor. This positional encoding can be learned as other layer parameters, or defined analytically.

解决这个问题的标准方法是在每个位置的特征表示中添加或连接位置编码，位置编码是一个依赖于张量中位置的特征向量。这个位置编码可以像其他层参数一样学习，或者通过分析定义。

For instance, in the original Transformer model, for a series of vectors of dimension \( D \), Vaswani et al. [2017] add an encoding of the sequence index as a series of sines and cosines at various frequencies:

例如，在原始的Transformer模型中，对于维度为\( D \)的一系列向量，Vaswani et al. [2017]添加了序列索引的编码，作为一系列不同频率的正弦和余弦：

pos-enc[\( t, d \)]

\[\begin{cases}
\sin\left(\frac{t}{T^d/D}\right) & \text{if } d \in 2\mathbb{N} \\
\cos\left(\frac{t}{T^{(d-1)/D}}\right) & \text{otherwise},
\end{cases}\]

with \( T = 10^4 \).

其中\( T = 10^4 \)。

#### 知识点讲解：
- **位置编码（Positional Encoding）**：位置编码用于在模型中引入序列的位置信息，特别是在处理序列数据（如文本或时间序列）时。Transformer模型通过正弦和余弦函数生成位置编码。
- **位置编码的作用**：位置编码使得模型能够捕捉序列中元素的相对位置信息，从而更好地处理序列数据。

---

### 总结：
- **跳跃连接**通过将某些层的输出直接传递到更远的层，绕过中间的处理，从而缓解梯度消失问题。残差连接是一种特殊的跳跃连接，广泛应用于深度模型中。
- **注意力层**通过计算输入张量中每个分量与输出张量中每个分量的相关性，使得模型能够捕捉长距离依赖关系。注意力机制是Transformer模型的核心组件。
- **词嵌入层**将离散的标记映射到连续的向量空间中，使得模型能够处理文本数据。
- **位置编码**用于在模型中引入序列的位置信息，特别是在处理序列数据时。Transformer模型通过正弦和余弦函数生成位置编码。

这些知识点是理解深度学习模型训练和优化过程的基础，掌握它们有助于更好地设计和优化深度学习模型。

---

好的，我将继续翻译接下来的章节。以下是Chapter 5的翻译和知识点讲解：

---

### Chapter 5 Architectures

#### 5 架构

The field of deep learning has developed over the years for each application domain multiple deep architectures that exhibit good trade-offs with respect to multiple criteria of interest: e.g. ease of training, accuracy of prediction, memory footprint, computational cost, scalability.

多年来，深度学习领域为每个应用领域开发了多种深度架构，这些架构在多个感兴趣的标准之间表现出良好的权衡：例如，易于训练、预测准确性、内存占用、计算成本、可扩展性。

---

### Chapter 5.1 Multi-Layer Perceptrons

#### 5.1 多层感知器

The simplest deep architecture is the Multi-Layer Perceptron (MLP), which takes the form of a succession of fully connected layers separated by activation functions. See an example in Figure 5.1. For historical reasons, in such a model, the number of hidden layers refers to the number of linear layers, excluding the last one.

最简单的深度架构是多层感知器（MLP），它由一系列全连接层组成，中间由激活函数分隔。见图5.1中的示例。由于历史原因，在这种模型中，隐藏层的数量指的是线性层的数量，不包括最后一层。

A key theoretical result is the universal approximation theorem [Cybenko, 1989] which states that, if the activation function \(\sigma\) is continuous

一个关键的理论结果是通用逼近定理 [Cybenko, 1989]，它指出，如果激活函数\(\sigma\)是连续的

\[\begin{array}{c}
Y \\
\uparrow \\
\text{fully-conn} \quad 2 \\
\downarrow \\
\text{relu} \\
\downarrow \\
\text{fully-conn} \quad 10 \\
\downarrow \\
\text{relu} \\
\downarrow \\
\text{fully-conn} \quad 25 \\
\downarrow \\
X \quad 50
\end{array}\]

Hidden layers

Figure 5.1: This multi-layer perceptron takes as input a one-dimensional tensor of size 50, is composed of three fully connected layers with outputs of dimensions respectively 25, 10, and 2, the two first followed by ReLU layers.

图5.1：这个多层感知器接受大小为50的一维张量作为输入，由三个全连接层组成，输出维度分别为25、10和2，前两层后面跟着ReLU层。

#### 知识点讲解：
- **多层感知器（MLP）**：MLP是最简单的深度神经网络架构，由多个全连接层和激活函数组成。它能够逼近任意复杂的函数。
- **通用逼近定理（Universal Approximation Theorem）**：该定理指出，只要激活函数是连续的，具有一个隐藏层的MLP可以逼近任何连续函数。

In spite of their simplicity, MLPs remain an important tool when the dimension of the signal to be processed is not too large.

尽管MLP结构简单，但在处理维度不大的信号时，它仍然是一个重要的工具。

#### 知识点讲解：
- **MLP的应用**：MLP适用于处理低维数据，如图像分类、回归任务等。然而，对于高维数据（如图像），卷积神经网络（CNN）通常更为有效。

---

### Chapter 5.2 Convolutional Networks

#### 5.2 卷积网络

The standard architecture for processing images is a convolutional network, or \underline{convnet}, that combines multiple convolutional layers, either to reduce the signal size before it can be processed by fully connected layers, or to output a 2D signal also of large size.

处理图像的标准架构是卷积网络（ConvNet），它结合了多个卷积层，要么在信号被全连接层处理之前减少信号大小，要么输出同样具有大尺寸的2D信号。

#### LeNet-like

The original LeNet model for image classification [LeCun et al., 1998] combines a series of 2D convolutional layers and max pooling layers that play the role of feature extractor, with a series of fully connected layers which act as a MLP and perform the classification per se (see Figure 5.2).

原始的LeNet模型用于图像分类 [LeCun et al., 1998]，它结合了一系列2D卷积层和最大池化层，这些层充当特征提取器，以及一系列全连接层，这些层充当MLP并执行分类本身（见图5.2）。

This architecture was the blueprint for many models that share its structure and are simply larger, such as AlexNet [Krizhevsky et al., 2012] or the VGG family [Simonyan and Zisserman, 2014].

这种架构是许多模型的蓝图，这些模型共享其结构并且规模更大，例如AlexNet [Krizhevsky et al., 2012] 或 VGG系列 [Simonyan and Zisserman, 2014]。

#### 知识点讲解：
- **LeNet**：LeNet是最早的卷积神经网络之一，用于手写数字识别。它通过卷积层和池化层提取特征，并通过全连接层进行分类。
- **AlexNet和VGG**：AlexNet和VGG是LeNet的扩展版本，具有更多的卷积层和更大的规模，显著提升了图像分类的性能。

#### Residual networks

Standard convolutional neural networks that follow the architecture of the LeNet family are not easily extended to deep architectures and suffer from the vanishing gradient problem. The residual networks, or ResNets, proposed by He et al. [2015] explicitly address the issue of the vanishing gradient with residual connections, which allow hundreds of layers. They have become standard architectures for computer vision applications, and exist in multiple versions depending on the number of layers. We are going to look in detail at the architecture of the ResNet-50 for classification.

遵循LeNet系列架构的标准卷积神经网络不容易扩展到深度架构，并且存在梯度消失问题。残差网络（ResNets）由He et al. [2015]提出，通过残差连接明确解决了梯度消失问题，允许数百层的深度。它们已成为计算机视觉应用的标准架构，并根据层数存在多个版本。我们将详细查看用于分类的ResNet-50架构。

#### 知识点讲解：
- **残差网络（ResNets）**：残差网络通过引入跳跃连接（skip connections）解决了深度网络中的梯度消失问题，使得训练非常深的网络成为可能。
- **ResNet-50**：ResNet-50是一个具有50层的残差网络，广泛应用于图像分类任务中。

---

### Chapter 5.3 Attention Models

#### 5.3 注意力模型

As stated in § 4.8, many applications, particularly from natural language processing, benefit greatly from models that include attention mechanisms. The architecture of choice for such tasks, which has been instrumental in recent advances in deep learning, is the \textit{Transformer} proposed by Vaswani et al. [2017].

正如在§4.8中所述，许多应用，特别是自然语言处理中的任务，极大地受益于包含注意力机制的模型。这类任务的首选架构是Vaswani et al. [2017]提出的\textit{Transformer}，它在深度学习的最新进展中起到了关键作用。

#### Transformer

The original Transformer, pictured in Figure 5.7, was designed for sequence-to-sequence translation. It combines an encoder that processes the input sequence to get a refined representation, and an autoregressive decoder that generates each token of the result sequence, given the encoder’s representation of the input sequence and the output tokens generated so far.

原始的Transformer（见图5.7）是为序列到序列翻译设计的。它结合了一个编码器，用于处理输入序列以获得精细的表示，以及一个自回归解码器，根据编码器对输入序列的表示和迄今为止生成的输出标记，生成结果序列的每个标记。

As the residual convolutional networks of § 5.2, both the encoder and the decoder of the Transformer are sequences of compounded blocks built with residual connections.

正如§5.2中的残差卷积网络，Transformer的编码器和解码器都是由残差连接构建的复合块序列。

#### 知识点讲解：
- **Transformer**：Transformer是一种基于自注意力机制的模型，广泛应用于自然语言处理任务中，如机器翻译、文本生成等。
- **编码器-解码器架构**：Transformer由编码器和解码器组成，编码器处理输入序列，解码器生成输出序列。

#### Generative Pre-trained Transformer

The \textit{Generative Pre-trained Transformer (GPT)} [Radford et al., 2018, 2019], pictured in Figure 5.8 is a pure autoregressive model that consists of a succession of causal self-attention blocks, hence a causal version of the original Transformer encoder.

\textit{生成式预训练Transformer（GPT）} [Radford et al., 2018, 2019]（见图5.8）是一个纯粹的自回归模型，由一系列因果自注意力块组成，因此是原始Transformer编码器的因果版本。

This class of models scales extremely well, up to hundreds of billions of trainable parameters [Brown et al., 2020]. We will come back to their use for text generation in § 7.1.

这类模型扩展性极好，可达到数千亿个可训练参数 [Brown et al., 2020]。我们将在§7.1中回到它们在文本生成中的应用。

#### 知识点讲解：
- **GPT**：GPT是一种基于Transformer的自回归模型，广泛用于文本生成任务。GPT-3是当前最大的语言模型之一，具有1750亿个参数。

#### Vision Transformer

Transformers have been put to use for image classification with the Vision Transformer (ViT) model [Dosovitskiy et al., 2020] (see Figure 5.9).

Transformer已被用于图像分类，通过Vision Transformer（ViT）模型 [Dosovitskiy et al., 2020]（见图5.9）。

It splits the three-channel input image into \( M \) patches of resolution \( P \times P \), which are then flattened to create a sequence of vectors \( X_1, \ldots, X_M \) of shape \( M \times 3P^2 \). This sequence is multiplied by a trainable matrix \( W^e \) of shape \( 3P^2 \times D \) to map it to an \( M \times D \) sequence, to which is concatenated one trainable vector \( E_0 \). The resulting \((M+1) \times D\) sequence \( E_0, \ldots, E_M \) is then processed through multiple self-attention blocks. See § 5.3 and Figure 5.6.

它将三通道输入图像分割为\( M \)个分辨率为\( P \times P \)的补丁，然后将它们展平以创建形状为\( M \times 3P^2 \)的向量序列\( X_1, \ldots, X_M \)。该序列乘以形状为\( 3P^2 \times D \)的可训练矩阵\( W^e \)，将其映射到\( M \times D \)序列，并连接一个可训练向量\( E_0 \)。生成的\((M+1) \times D\)序列\( E_0, \ldots, E_M \)然后通过多个自注意力块进行处理。参见§5.3和图5.6。

#### 知识点讲解：
- **Vision Transformer（ViT）**：ViT将图像分割为多个补丁，并将这些补丁作为序列输入Transformer模型，从而应用于图像分类任务。

---

### 总结：
- **多层感知器（MLP）**是最简单的深度神经网络架构，适用于低维数据的处理。
- **卷积网络（ConvNet）**通过卷积层和池化层提取图像特征，广泛应用于图像分类任务。LeNet、AlexNet和VGG是经典的卷积网络架构。
- **残差网络（ResNets）**通过引入跳跃连接解决了深度网络中的梯度消失问题，允许训练非常深的网络。
- **Transformer**是一种基于自注意力机制的模型，广泛应用于自然语言处理和图像分类任务中。GPT和ViT是Transformer的变体，分别用于文本生成和图像分类。



好的，我将继续翻译接下来的章节。以下是Chapter 6的翻译和知识点讲解：

---

### Chapter 6 Prediction

#### 6 预测

A first category of applications, such as face recognition, sentiment analysis, object detection, or speech recognition, requires predicting an unknown value from an available signal.

第一类应用，如人脸识别、情感分析、目标检测或语音识别，需要从可用信号中预测未知值。

---

### Chapter 6.1 Image Denoising

#### 6.1 图像去噪

A direct application of deep models to image processing is to recover from degradation by utilizing the redundancy in the statistical structure of images. The petals of a sunflower in a grayscale picture can be colored with high confidence, and the texture of a geometric shape such as a table on a low-light, grainy picture can be corrected by averaging it over a large area likely to be uniform.

深度模型在图像处理中的直接应用是通过利用图像统计结构中的冗余来恢复退化。例如，灰度图像中的向日葵花瓣可以高置信度地着色，而低光、颗粒状图片中的几何形状（如桌子）的纹理可以通过在可能均匀的大区域上平均来校正。

A denoising autoencoder is a model that takes a degraded signal \( \widetilde{X} \) as input and computes an estimate of the original signal \( X \). For images, it is a convolutional network that may integrate skip-connections, in particular to combine representations at the same resolution obtained early and late in the model, as well as attention layers to facilitate taking into account elements that are far away from each other.

去噪自编码器是一种模型，它以退化信号\( \widetilde{X} \)作为输入，并计算原始信号\( X \)的估计值。对于图像，它是一个卷积网络，可能集成跳跃连接，特别是为了结合模型早期和后期获得的相同分辨率的表示，以及注意力层，以便于考虑彼此相距较远的元素。

Such a model is trained by collecting a large number of clean samples paired with their degraded inputs. The latter can be captured in degraded conditions, such as low-light or inadequate focus, or generated algorithmically, for instance, by converting the clean sample to grayscale, reducing its size, or aggressively compressing it with a lossy compression method.

这种模型通过收集大量干净样本及其退化输入进行训练。后者可以在退化条件下捕获，例如低光或对焦不当，或者通过算法生成，例如，将干净样本转换为灰度、缩小其尺寸或使用有损压缩方法进行压缩。

The standard training procedure for denoising autoencoders uses the MSE loss summed across all pixels, in which case the model aims at computing the best average clean picture, given the degraded one, that is \( \mathbb{E}[X | \bar{X}] \). This quantity may be problematic when \( X \) is not completely determined by \( \bar{X} \), in which case some parts of the generated signal may be an unrealistic, blurry average.

去噪自编码器的标准训练过程使用所有像素上的MSE损失，在这种情况下，模型旨在计算给定退化图像的最佳平均干净图像，即\( \mathbb{E}[X | \bar{X}] \)。当\( X \)不完全由\( \bar{X} \)决定时，这个量可能会有问题，在这种情况下，生成信号的某些部分可能是不现实的模糊平均值。

#### 知识点讲解：
- **图像去噪（Image Denoising）**：图像去噪是通过深度学习模型从退化图像中恢复原始图像的过程。去噪自编码器是一种常用的去噪模型。
- **去噪自编码器（Denoising Autoencoder）**：去噪自编码器通过训练模型从退化图像中恢复原始图像，通常使用卷积网络和跳跃连接来提高性能。

---

### Chapter 6.2 Image Classification

#### 6.2 图像分类

Image classification is the simplest strategy for extracting semantics from an image and consists of predicting a class from a finite, predefined number of classes, given an input image.

图像分类是从图像中提取语义的最简单策略，它涉及从有限的预定义类别中预测一个类别，给定输入图像。

The standard models for this task are convolutional networks, such as ResNets (see § 5.2), and attention-based models such as ViT (see § 5.3). These models generate a vector of logits with as many dimensions as there are classes.

此任务的标准模型是卷积网络，如ResNets（见§5.2），以及基于注意力的模型，如ViT（见§5.3）。这些模型生成一个具有与类别数量相同维度的logits向量。

The training procedure simply minimizes the cross-entropy loss (see § 3.1). Usually, performance can be improved with data augmentation, which consists of modifying the training samples with hand-designed random transformations that do not change the semantic content of the image, such as cropping, scaling, mirroring, or color changes.

训练过程简单地最小化交叉熵损失（见§3.1）。通常，可以通过数据增强来提高性能，数据增强包括使用手工设计的随机变换修改训练样本，这些变换不会改变图像的语义内容，例如裁剪、缩放、镜像或颜色变化。

#### 知识点讲解：
- **图像分类（Image Classification）**：图像分类是将输入图像分配到预定义类别中的任务。卷积网络和基于注意力的模型是常用的图像分类模型。
- **数据增强（Data Augmentation）**：数据增强通过对训练图像进行随机变换（如裁剪、缩放、镜像等）来增加训练数据的多样性，从而提高模型的泛化能力。

---

### Chapter 6.3 Object Detection

#### 6.3 目标检测

A more complex task for image understanding is object detection, in which the objective is, given an input image, to predict the classes and positions of objects of interest.

图像理解中更复杂的任务是目标检测，其目标是在给定输入图像的情况下，预测感兴趣对象的类别和位置。

An object position is formalized as the four coordinates \((x_1, y_1, x_2, y_2)\) of a rectangular bounding box, and the ground truth associated with each training image is a list of such bounding boxes, each labeled with the class of the object contained therein.

对象位置被形式化为矩形边界框的四个坐标\((x_1, y_1, x_2, y_2)\)，每个训练图像的ground truth是此类边界框的列表，每个边界框都标有其中包含的对象的类别。

The standard approach to solve this task, for instance, by the Single Shot Detector (SSD) [Liu et al., 2015]), is to use a convolutional neural network that produces a sequence of image representations \(Z_s\) of size \(D_s \times H_s \times W_s\), \(s = 1, \ldots, S\), with decreasing spatial resolution \(H_s \times W_s\) down to \(1 \times 1\) for \(s = S\) (see Figure 6.1). Each of these tensors covers the input image in full, so the \(h, w\) indices correspond to a partitioning of the image lattice into regular squares that gets coarser when \(s\) increases.

解决此任务的标准方法，例如通过Single Shot Detector（SSD）[Liu et al., 2015]，是使用卷积神经网络生成一系列图像表示\(Z_s\)，大小为\(D_s \times H_s \times W_s\)，\(s = 1, \ldots, S\)，空间分辨率\(H_s \times W_s\)随着\(s\)的增加而降低，直到\(s = S\)时为\(1 \times 1\)（见图6.1）。这些张量中的每一个都完全覆盖输入图像，因此\(h, w\)索引对应于将图像网格划分为规则的正方形，随着\(s\)的增加，这些正方形变得更粗糙。

#### 知识点讲解：
- **目标检测（Object Detection）**：目标检测是在图像中定位并分类多个对象的任务。SSD是一种常用的目标检测模型，通过卷积网络生成多尺度的特征图来检测对象。
- **边界框（Bounding Box）**：边界框用于表示图像中对象的位置，通常由四个坐标（左上角和右下角）定义。

---

### Chapter 6.4 Semantic Segmentation

#### 6.4 语义分割

The finest-grain prediction task for image understanding is semantic segmentation, which consists of predicting, for each pixel, the class of the object to which it belongs. This can be achieved with a standard convolutional neural network that outputs a convolutional map with as many channels as classes, carrying the estimated logits for every pixel.

图像理解中最细粒度的预测任务是语义分割，它涉及为每个像素预测其所属对象的类别。这可以通过标准的卷积神经网络实现，该网络输出具有与类别数量相同通道的卷积图，携带每个像素的估计logits。

While a standard residual network, for instance, can generate a dense output of the same resolution as its input, as for object detection, this task requires operating at multiple scales. This is necessary so that any object, or sufficiently informative sub-part, regardless of its size, is captured somewhere in the model by the feature representation at a single tensor position. Hence, standard architectures for this task downscale the image with a series of convolutional layers to increase the receptive field of the activations, and re-upscale it with a series of transposed convolutional layers, or other upscaling methods such as bilinear interpolation, to make the prediction at high resolution.

虽然标准的残差网络可以生成与输入分辨率相同的密集输出，如目标检测，但此任务需要在多个尺度上操作。这是必要的，以便任何对象或足够信息的子部分，无论其大小如何，都能在模型中的某个位置通过单个张量位置的特征表示捕获。因此，此任务的标准架构通过一系列卷积层缩小图像以增加激活的感受野，并通过一系列转置卷积层或其他上采样方法（如双线性插值）重新放大图像，以在高分辨率下进行预测。

#### 知识点讲解：
- **语义分割（Semantic Segmentation）**：语义分割是为图像中的每个像素分配类别标签的任务。与目标检测不同，语义分割不区分同一类别的不同实例。
- **多尺度操作（Multi-scale Operation）**：语义分割模型通常需要在多个尺度上操作，以捕捉不同大小的对象。通过下采样和上采样操作，模型可以在不同分辨率下进行预测。

---

### 总结：
- **图像去噪**是通过深度学习模型从退化图像中恢复原始图像的过程，去噪自编码器是常用的去噪模型。
- **图像分类**是将输入图像分配到预定义类别中的任务，卷积网络和基于注意力的模型是常用的图像分类模型。
- **目标检测**是在图像中定位并分类多个对象的任务，SSD是一种常用的目标检测模型。
- **语义分割**是为图像中的每个像素分配类别标签的任务，通常需要在多个尺度上操作以捕捉不同大小的对象。

这些知识点是理解深度学习在图像处理中应用的基础，掌握它们有助于更好地设计和优化图像处理模型。

---

好的，我将继续翻译接下来的章节。以下是Chapter 6.5到Chapter 6.7的翻译和知识点讲解：

---

### Chapter 6.5 Speech Recognition

#### 6.5 语音识别

Speech recognition consists of converting a sound sample into a sequence of words. There have been plenty of approaches to this problem historically, but a conceptually simple and recent one proposed by Radford et al. [2022] consists of casting it as a sequence-to-sequence translation and then solving it with a standard attention-based Transformer, as described in § 5.3.

语音识别涉及将声音样本转换为单词序列。历史上有很多方法来解决这个问题，但Radford et al. [2022]提出的一个概念上简单且最近的方法是将它视为序列到序列的翻译，然后用标准的基于注意力的Transformer来解决，如§5.3所述。

Their model first converts the sound signal into a spectrogram, which is a one-dimensional series \( T \times D \), that encodes at every time step a vector of energies in \( D \) frequency bands. The associated text is encoded with the BPE tokenizer (see § 3.2).

他们的模型首先将声音信号转换为频谱图，这是一个一维序列\( T \times D \)，在每个时间步编码\( D \)个频带中的能量向量。相关的文本使用BPE分词器进行编码（见§3.2）。

The spectrogram is processed through a few 1D convolutional layers, and the resulting representation is fed into the encoder of the Transformer. The decoder directly generates a discrete sequence of tokens, that correspond to one of the possible tasks considered during training. Multiple objectives are considered: transcription of English or non-English text, translation from any language to English, or detection of non-speech sequences, such as background music or ambient noise.

频谱图通过几个1D卷积层进行处理，生成的表示被输入到Transformer的编码器中。解码器直接生成离散的标记序列，对应于训练期间考虑的其中一个可能任务。考虑了多个目标：英语或非英语文本的转录、从任何语言到英语的翻译，或非语音序列的检测，如背景音乐或环境噪声。

This approach allows leveraging extremely large datasets that combine multiple types of sound sources with diverse ground truths.

这种方法允许利用结合了多种声音源和多样化ground truth的极大数据集。

It is noteworthy that even though the ultimate goal of this approach is to produce a translation as deterministic as possible given the input signal, it is formally the sampling of a text distribution conditioned on a sound sample, hence a synthesis process. The decoder is, in fact, extremely similar to the generative model of § 7.1.

值得注意的是，尽管这种方法的最终目标是尽可能确定性地生成翻译，但它在形式上是基于声音样本的文本分布的采样，因此是一个合成过程。解码器实际上与§7.1中的生成模型非常相似。

#### 知识点讲解：
- **语音识别（Speech Recognition）**：语音识别是将声音信号转换为文本的任务。Transformer模型通过将声音信号转换为频谱图，并使用编码器-解码器架构进行序列到序列的翻译。
- **频谱图（Spectrogram）**：频谱图是声音信号的时频表示，通常用于语音识别任务中。
- **BPE分词器（Byte Pair Encoding Tokenizer）**：BPE分词器是一种将文本分解为子词单元的方法，广泛应用于自然语言处理任务中。

---

### Chapter 6.6 Text-Image Representations

#### 6.6 文本-图像表示

A powerful approach to image understanding consists of learning consistent image and text representations, such that an image, or a textual description of it, would be mapped to the same feature vector.

图像理解的一种强大方法是学习一致的图像和文本表示，使得图像或其文本描述被映射到相同的特征向量。

The \textit{Contrastive Language-Image Pre-training (CLIP)} proposed by Radford et al. [2021] combines an image encoder \( f \), which is a ViT, and a text encoder \( g \), which is a GPT. See § 5.3 for both.

Radford et al. [2021]提出的\textit{对比语言-图像预训练（CLIP）}结合了一个图像编码器\( f \)（即ViT）和一个文本编码器\( g \)（即GPT）。参见§5.3。

To repurpose a GPT as a text encoder, instead of a standard autoregressive model, they add an "end of sentence" token to the input sequence, and use the representation of this token in the last layer as the embedding. Its dimension is between 512 and 1024, depending on the configuration.

为了将GPT重新用作文本编码器，而不是标准的自回归模型，他们在输入序列中添加了一个“句子结束”标记，并使用最后一层中该标记的表示作为嵌入。其维度在512到1024之间，具体取决于配置。

Those two models are trained from scratch using a dataset of 400 million image-text pairs \((i_k, t_k)\) collected from the internet. The training procedure follows the standard mini-batch stochastic gradient descent approach but relies on a contrastive loss. The embeddings are computed for every image and every text of the \( N \) pairs in the mini-batch, and a cosine similarity measure is computed not only between text and image embeddings from each pair, but also across pairs, resulting in an \( N \times N \) matrix of similarity scores:

这两个模型从头开始训练，使用从互联网收集的4亿个图像-文本对\((i_k, t_k)\)的数据集。训练过程遵循标准的小批量随机梯度下降方法，但依赖于对比损失。为小批量中的每对图像和文本计算嵌入，并计算文本和图像嵌入之间的余弦相似度，不仅在每个对之间，还在对之间计算，生成一个\( N \times N \)的相似度分数矩阵：

\[ l_{m,n} = f(t_m) \cdot g(t_n), \, m = 1, \ldots, N, n = 1, \ldots, N. \]

The model is trained with cross-entropy so that, \(\forall n\) the values \( l_1, n, \ldots, l_N, n \) interpreted as logit scores predict \( n \), and similarly for \( l_{n,1}, \ldots, l_{n,N} \). This means that \(\forall n, m, \, \text{s.t. } n \neq m \) the similarity \( l_{n,n} \) is unambiguously greater than both \( l_{n,m} \) and \( l_{m,n} \).

模型使用交叉熵进行训练，因此\(\forall n\)，值\( l_1, n, \ldots, l_N, n \)被解释为logit分数，预测\( n \)，同样适用于\( l_{n,1}, \ldots, l_{n,N} \)。这意味着\(\forall n, m, \, \text{s.t. } n \neq m \)，相似度\( l_{n,n} \)明确大于\( l_{n,m} \)和\( l_{m,n} \)。

When it has been trained, this model can be used to do zero-shot prediction, that is, classifying a signal in the absence of training examples by defining a series of candidate classes with text descriptions, and computing the similarity of the embedding of an image with the embedding of each of those descriptions (see Figure 6.4).

训练完成后，该模型可以用于零样本预测，即通过定义一系列带有文本描述的候选类别，并计算图像嵌入与每个描述嵌入的相似度，在没有训练样本的情况下对信号进行分类（见图6.4）。

Additionally, since the textual descriptions are often detailed, such a model has to capture a richer representation of images and pick up cues beyond what is necessary for instance for classification. This translates to excellent performance on challenging datasets such as ImageNet Adversarial [Hendrycks et al., 2019] which was specifically designed to degrade or erase cues on which standard predictors rely.

此外，由于文本描述通常很详细，这样的模型必须捕捉更丰富的图像表示，并提取超出分类所需的线索。这转化为在具有挑战性的数据集（如ImageNet Adversarial [Hendrycks et al., 2019]）上的出色性能，该数据集专门设计用于降解或消除标准预测器依赖的线索。

#### 知识点讲解：
- **对比语言-图像预训练（CLIP）**：CLIP通过对比学习将图像和文本映射到相同的嵌入空间，使得模型能够进行零样本预测。
- **零样本预测（Zero-shot Prediction）**：零样本预测是指在没有特定类别训练样本的情况下，通过文本描述对图像进行分类。

---

### Chapter 6.7 Reinforcement Learning

#### 6.7 强化学习

Many problems, such as strategy games or robotic control, can be formalized with a discrete-time state process \( S_t \) and reward process \( R_t \) that can be modulated by choosing actions \( A_t \). If \( S_t \) is Markovian, meaning that it carries alone as much information about the future as all the past states until that instant, such an object is a Markovian Decision Process (MDP).

许多问题，如策略游戏或机器人控制，可以用离散时间状态过程\( S_t \)和奖励过程\( R_t \)来形式化，这些过程可以通过选择动作\( A_t \)来调节。如果\( S_t \)是马尔可夫的，意味着它单独携带了关于未来的所有信息，直到该时刻的所有过去状态，这样的对象就是马尔可夫决策过程（MDP）。

Given an MDP, the objective is classically to find a policy \(\pi\) such that \( A_t = \pi(S_t) \) maximizes the expectation of the return, which is an accumulated discounted reward:

给定一个MDP，经典目标是找到一个策略\(\pi\)，使得\( A_t = \pi(S_t) \)最大化回报的期望，即累积折扣奖励：

\[\mathbb{E} \left[ \sum_{t \geq 0} \gamma^t R_t \right],\]

for a discount factor \( 0 < \gamma < 1 \).

其中折扣因子\( 0 < \gamma < 1 \)。

This is the standard setup of Reinforcement Learning (RL), and it can be worked out by introducing the optimal state-action value function \( Q(s, a) \) which is the expected return if we execute action \( a \) in state \( s \), and then follow the optimal policy. It provides a means to compute the optimal policy as \(\pi(s) = \arg\max_a Q(s, a)\), and, thanks to the Markovian assumption, it verifies the Bellman equation:

这是强化学习（RL）的标准设置，可以通过引入最优状态-动作值函数\( Q(s, a) \)来解决，该函数表示在状态\( s \)中执行动作\( a \)然后遵循最优策略的期望回报。它提供了一种计算最优策略的方法，即\(\pi(s) = \arg\max_a Q(s, a)\)，并且由于马尔可夫假设，它满足贝尔曼方程：

\[Q(s, a) = \tag{6.1}\]

\[\mathbb{E} \left[ R_t + \gamma \max_{a'} Q(S_{t+1}, a') \right] S_t = s, A_t = a \]

from which we can design a procedure to train a parametric model \( Q(\cdot, \cdot; w) \).

从中我们可以设计一个训练参数模型\( Q(\cdot, \cdot; w) \)的过程。

To apply this framework to play classical Atari video games, Mnih et al. [2015] use for \( S_t \) the concatenation of the frame at time \( t \) and the three that precede, so that the Markovian assumption is reasonable, and use for \( Q \) a model dubbed the Deep Q-Network (DQN), composed of two convolutional layers and one fully connected layer with one output value per action, following the classical structure of a LeNet (see § 5.2).

为了将这个框架应用于玩经典的Atari视频游戏，Mnih et al. [2015]使用\( S_t \)作为时间\( t \)的帧和前三个帧的连接，使得马尔可夫假设合理，并使用\( Q \)的模型称为深度Q网络（DQN），由两个卷积层和一个全连接层组成，每个动作有一个输出值，遵循LeNet的经典结构（见§5.2）。

Training is achieved by alternatively playing and recording episodes, and building mini-batches of tuples \((s_n, a_n, r_n, s'_n) \sim (S_t, A_t, R_t, S_{t+1})\) taken across stored episodes and time steps, and minimizing

训练通过交替玩游戏和记录剧集，并构建从存储的剧集和时间步骤中提取的元组\((s_n, a_n, r_n, s'_n) \sim (S_t, A_t, R_t, S_{t+1})\)的小批量，并最小化

\[\mathcal{L}(w) = \frac{1}{N} \sum_{n=1}^{N} \left( Q(s_n, a_n; w) - y_n \right)^2 \tag{6.2}\]

with one iteration of SGD, where \( y_n = r_n \) if this tuple is the end of the episode, and \( y_n = r_n + \gamma \max_a Q(s'_n, a; w) \) otherwise.

通过一次SGD迭代，其中如果该元组是剧集的结束，则\( y_n = r_n \)，否则\( y_n = r_n + \gamma \max_a Q(s'_n, a; w) \)。

#### 知识点讲解：
- **强化学习（Reinforcement Learning）**：强化学习是一种通过与环境交互来学习策略的机器学习方法。目标是最大化累积奖励。
- **深度Q网络（Deep Q-Network, DQN）**：DQN是一种基于Q学习的深度强化学习模型，通过卷积网络处理图像输入，并输出每个动作的Q值。

---

### 总结：
- **语音识别**是将声音信号转换为文本的任务，Transformer模型通过将声音信号转换为频谱图，并使用编码器-解码器架构进行序列到序列的翻译。
- **对比语言-图像预训练（CLIP）**通过对比学习将图像和文本映射到相同的嵌入空间，使得模型能够进行零样本预测。
- **强化学习**是一种通过与环境交互来学习策略的机器学习方法，深度Q网络（DQN）是一种常用的强化学习模型，用于处理图像输入的任务。

这些知识点是理解深度学习在语音识别、图像-文本表示和强化学习中应用的基础，掌握它们有助于更好地设计和优化相关模型。

---











































### Chapter 6.8: The Compute Schism

#### 6.8.1: Prompt Engineering

**English:**
The simplest strategy to specialize or improve a Large Language Model with a limited computational budget is to use prompt engineering, that is, to carefully craft the beginning of the text sequence to bias the autoregressive process [Sahoo et al., 2024]. This approach moves a part of the information traditionally encoded in the model’s parameters to the input.

**Chinese:**
在计算资源有限的情况下，专门化或改进大型语言模型的最简单策略是使用提示工程（prompt engineering），即精心设计文本序列的开头，以引导自回归过程 [Sahoo et al., 2024]。这种方法将传统上编码在模型参数中的部分信息转移到输入中。

**知识点讲解:**
- **Prompt Engineering（提示工程）**: 这是一种通过设计输入提示（prompt）来引导模型生成特定输出的技术。提示工程的核心思想是通过精心设计的输入，引导模型生成符合预期的结果，而不需要重新训练模型。
- **Autoregressive Process（自回归过程）**: 自回归模型是一种生成模型，它通过逐步生成序列中的每个元素来生成整个序列。每个元素的生成依赖于之前生成的元素。

**English:**
We saw in § 7.1 a simple example of few-shot prediction, to use an LLM for a text classification task without fine-tuning. A long and sophisticated prompt allows generalizing this strategy to complex tasks.

**Chinese:**
我们在§7.1中看到了一个简单的少样本预测（few-shot prediction）示例，即在不需要微调的情况下使用大型语言模型（LLM）进行文本分类任务。一个长而复杂的提示可以将这种策略推广到更复杂的任务中。

**知识点讲解:**
- **Few-shot Prediction（少样本预测）**: 少样本预测是指模型在只有少量样本的情况下进行预测。通过精心设计的提示，模型可以在没有大量训练数据的情况下完成任务。
- **Fine-tuning（微调）**: 微调是指在预训练模型的基础上，使用特定任务的数据对模型进行进一步训练，以适应特定任务的需求。

**English:**
Since the prompt’s role is to leverage the “good” biases that were present in the training set, it benefits from surprising strategies such as stating that the response is generated by a skilled professional [Xu et al., 2023].

**Chinese:**
由于提示的作用是利用训练集中存在的“良好”偏差，因此它可以从一些出人意料的策略中受益，例如声明响应是由熟练的专业人士生成的 [Xu et al., 2023]。

**知识点讲解:**
- **Bias（偏差）**: 在机器学习中，偏差指的是模型在训练过程中学到的偏好或倾向。提示工程通过利用这些偏差来引导模型生成更符合预期的输出。

**English:**
The context size of a language model, that is, the number of tokens it can operate on, directly modulates the quantity of information that can be provided in the prompt. This is mostly constrained by the computational cost of standard attention models, which is quadratic with the context size (see § 4.8).

**Chinese:**
语言模型的上下文大小，即它可以处理的令牌数量，直接决定了提示中可以提供的信息量。这主要受到标准注意力模型计算成本的限制，该成本与上下文大小成二次方关系（参见§4.8）。

**知识点讲解:**
- **Context Size（上下文大小）**: 上下文大小指的是模型在处理输入时能够考虑的令牌数量。较大的上下文大小允许模型处理更长的输入序列，但也会增加计算成本。
- **Attention Models（注意力模型）**: 注意力模型是一种用于处理序列数据的模型，它通过计算输入序列中每个元素的重要性来生成输出。标准的注意力模型的计算复杂度与输入序列长度的平方成正比。

**English:**
A remarkable type of prompting aims at making the model generate intermediate steps before generating the response itself.

**Chinese:**
一种显著的提示类型旨在让模型在生成响应之前生成中间步骤。

**知识点讲解:**
- **Intermediate Steps（中间步骤）**: 中间步骤是指模型在生成最终输出之前生成的中间结果。通过生成中间步骤，模型可以更好地分解复杂任务，从而提高生成结果的准确性。

**English:**
Such a chain-of-thought is composed of successive steps that are simpler, hence have been better modeled during training, and are predicted more deterministically [Wei et al., 2022; Kojima et al., 2022]. See Figure 8.1 for an example.

**Chinese:**
这种思维链（chain-of-thought）由一系列更简单的步骤组成，因此在训练过程中得到了更好的建模，并且可以更确定性地预测 [Wei et al., 2022; Kojima et al., 2022]。参见图8.1中的示例。

**知识点讲解:**
- **Chain-of-Thought（思维链）**: 思维链是一种提示技术，通过让模型生成中间推理步骤来引导模型生成更准确的最终答案。这种方法特别适用于需要复杂推理的任务。

**English:**
Prompt engineering can also be put to use to connect a language model to an external knowledge base. It plays the role of a smart interface that allows the end user to formulate questions in natural language and get back a response that combines information that is not encoded in the model’s parameters [Lewis et al., 2020].

**Chinese:**
提示工程还可以用于将语言模型连接到外部知识库。它充当一个智能接口，允许最终用户以自然语言提出问题，并返回一个结合了未编码在模型参数中的信息的响应 [Lewis et al., 2020]。

**知识点讲解:**
- **External Knowledge Base（外部知识库）**: 外部知识库是指模型外部的结构化数据源，如数据库或知识图谱。通过提示工程，模型可以利用这些外部知识来生成更准确的响应。

**English:**
For such Retrieval-Augmented Generation (RAG), an embedding model is used to retrieve documents whose embedding is correlated to that of the user’s query. Then, a prompt is constructed by joining these retrieved documents with instructions to combine them, and the generative model produces the response to the user’s query.

**Chinese:**
对于这种检索增强生成（Retrieval-Augmented Generation, RAG），使用嵌入模型来检索与用户查询嵌入相关的文档。然后，通过将这些检索到的文档与组合指令结合来构建提示，生成模型生成对用户查询的响应。

**知识点讲解:**
- **Retrieval-Augmented Generation (RAG)（检索增强生成）**: RAG是一种结合了检索和生成的技术，通过从外部知识库中检索相关信息，并将其与生成模型结合，生成更准确的响应。
- **Embedding Model（嵌入模型）**: 嵌入模型是一种将文本或其他数据转换为向量表示的模型。这些向量表示可以用于计算文本之间的相似性。

#### 6.8.2: Quantization

**English:**
Although training or generating multiple streams can benefit from high-end parallel computing devices, deployment of a Large Language Model for individual use requires generally single-stream inference, which is bounded by memory size and speed far more than by computation.

**Chinese:**
尽管训练或生成多个流可以从高端并行计算设备中受益，但为个人使用部署大型语言模型通常需要单流推理，这更多地受到内存大小和速度的限制，而不是计算能力的限制。

**知识点讲解:**
- **Single-stream Inference（单流推理）**: 单流推理是指模型在单个计算流上运行，通常用于个人设备或资源有限的环境中。与多流推理相比，单流推理更受内存和速度的限制。

**English:**
As stated in § 2.1, parameters, activations, and gradients are usually encoded with 32 or 16 bits. The precision it provides is necessary for training, to allow gradual changes to accumulate.

**Chinese:**
如§2.1所述，参数、激活值和梯度通常用32位或16位编码。这种精度对于训练是必要的，以便允许逐渐的变化积累。

**知识点讲解:**
- **Precision（精度）**: 精度指的是数值表示的位数，通常用32位或16位浮点数表示。高精度在训练过程中是必要的，以确保梯度下降的稳定性。

**English:**
However, since activations are the sums of many terms, quantization during inference is mitigated by an averaging effect. This is even more true with large architectures, and models quantized down to 6 or 4 bits per parameter exhibit remarkable performance. Additionally to reducing the memory footprint, quantization also improves inference speed significantly.

**Chinese:**
然而，由于激活值是许多项的总和，推理过程中的量化通过平均效应得到了缓解。对于大型架构来说，这一点尤其明显，量化到每个参数6或4位的模型表现出显著的性能。除了减少内存占用外，量化还显著提高了推理速度。

**知识点讲解:**
- **Quantization（量化）**: 量化是一种将高精度数值转换为低精度数值的技术，通常用于减少模型的内存占用和计算成本。量化可以在推理过程中显著提高效率，同时保持模型的性能。

**English:**
This has motivated the development of software to quantize existing models with Post-Training Quantization, and run them in single-stream inference on consumer hardware, such as llama.cpp [Llama.cpp, 2023]. This framework implements multiple formats, that apply specific quantization levels for the different weight matrices of a language model. For instance the quantization may use more bits for the \( W^v \) weights of the attention blocks, and for the weights of the feed-forward blocks.

**Chinese:**
这推动了开发用于量化现有模型的软件，如llama.cpp [Llama.cpp, 2023]，并在消费级硬件上以单流推理运行这些模型。该框架实现了多种格式，针对语言模型的不同权重矩阵应用特定的量化级别。例如，量化可能对注意力块的\( W^v \)权重和前馈块的权重使用更多的位数。

**知识点讲解:**
- **Post-Training Quantization（训练后量化）**: 训练后量化是指在模型训练完成后对模型进行量化，以减少模型的内存占用和计算成本。这种方法通常用于在资源有限的设备上部署模型。

**English:**
An example of llama.cpp’s quantization is Q4_1. It quantizes individually sub-blocks of 32 entries of the original weight matrix by storing for each a scaling factor \( d \) and a bias \( m \) in the original FP16 encoding, and encoding each entry \( x \) with 4 bits as a value \( q \in \{0, \ldots, 2^4 - 1\} \). The resulting de-quantized value being \( \bar{x} = dq + m \).

**Chinese:**
llama.cpp的量化示例是Q4_1。它通过对原始权重矩阵的每个32个条目的子块进行单独量化，为每个子块存储一个缩放因子\( d \)和一个偏置\( m \)，并使用4位将每个条目\( x \)编码为\( q \in \{0, \ldots, 2^4 - 1\} \)。反量化后的值为\( \bar{x} = dq + m \)。

**知识点讲解:**
- **Scaling Factor and Bias（缩放因子和偏置）**: 在量化过程中，缩放因子和偏置用于将低精度的量化值转换回高精度的数值。缩放因子用于调整量化值的范围，而偏置用于调整量化值的偏移。

**English:**
Such a block was encoded originally as 32 values in FP16, hence 64 bytes, while the quantized version needs 4 bytes for \( q \) and \( m \) and \( 32 \cdot 4 \) bits = 16 bytes for the entries, hence a total of 20 bytes.

**Chinese:**
这样的块原本编码为32个FP16值，因此需要64字节，而量化版本需要4字节用于\( q \)和\( m \)，以及\( 32 \cdot 4 \)位=16字节用于条目，因此总共需要20字节。

**知识点讲解:**
- **Memory Footprint（内存占用）**: 内存占用指的是模型在运行时所需的内存大小。量化可以显著减少模型的内存占用，从而使其能够在资源有限的设备上运行。

**English:**
Such an aggressive quantization surprisingly degrades only marginally the performance of the models, as illustrated on Figure 8.2.

**Chinese:**
如此激进的量化令人惊讶地仅略微降低了模型的性能，如图8.2所示。

**知识点讲解:**
- **Aggressive Quantization（激进量化）**: 激进量化是指将模型参数压缩到极低精度的量化方法。尽管量化精度较低，但通过适当的缩放因子和偏置调整，模型的性能损失可以控制在可接受的范围内。

**English:**
An alternative to Post-Training Quantization is Quantization-Aware Training that applies quantization during the forward pass but keeps high-precision encoding of parameters and gradients, and propagates the gradients during the backward pass as if there was no quantization [Ma et al., 2024].

**Chinese:**
训练后量化的另一种替代方法是量化感知训练（Quantization-Aware Training），它在前向传播过程中应用量化，但保持参数和梯度的高精度编码，并在反向传播过程中传播梯度，就像没有量化一样 [Ma et al., 2024]。

**知识点讲解:**
- **Quantization-Aware Training（量化感知训练）**: 量化感知训练是一种在训练过程中模拟量化的技术，通过在训练过程中应用量化来使模型适应低精度的计算环境。这种方法可以在训练过程中调整模型参数，以减少量化对模型性能的影响。

#### 6.8.3: Adapters

**English:**
As we saw in § 3.6, fine-tuning is a key strategy to reuse pre-trained models. Since it aims at making only minor changes to an existing model, techniques have been developed that add components with few parameters, referred to as adapters, to the pre-trained architecture, and freeze all the original parameters [Houlsby et al., 2019].

**Chinese:**
正如我们在§3.6中看到的，微调是重用预训练模型的关键策略。由于它旨在对现有模型进行少量修改，因此开发了一些技术，向预训练架构中添加少量参数的组件，称为适配器（adapters），并冻结所有原始参数 [Houlsby et al., 2019]。

**知识点讲解:**
- **Adapters（适配器）**: 适配器是一种在预训练模型中添加少量可训练参数的技术，用于在不改变原始模型参数的情况下适应新任务。适配器通常插入到模型的某些层中，以微调模型的行为。

**English:**
The current dominant method is the Low-Rank Adaptation (LoRA), which adds low-rank corrections to some of the model’s weight matrices [Hu et al., 2021].

**Chinese:**
当前的主流方法是低秩适应（Low-Rank Adaptation, LoRA），它向模型的某些权重矩阵添加低秩校正 [Hu et al., 2021]。

**知识点讲解:**
- **Low-Rank Adaptation (LoRA)（低秩适应）**: LoRA是一种通过向模型的权重矩阵添加低秩矩阵来进行微调的技术。这种方法通过引入少量额外的参数来调整模型的行为，而不需要重新训练整个模型。

**English:**
Formally, given a linear operation of the form \(XW^T\), where \(X\) is a \(N \times D\) tensor of activations for a batch of \(N\) samples, and \(W\) is a \(C \times D\) weight matrix, the LoRA adapter replaces this operation with \(X(W + BA)^T\), where \(A\) and \(B\) are two trainable matrices of size \(R \times D\) and \(C \times R\) respectively, with \(R \ll \min(C, D)\), and the matrix \(W\) is removed from the trainable parameters. The matrix \(A\) is initialized with random Gaussian values, and \(B\) is set to zero, so that the fine-tuning starts with a model that computes an output identical to that of the original one.

**Chinese:**
形式上，给定形式为\(XW^T\)的线性操作，其中\(X\)是大小为\(N \times D\)的激活张量，表示一批\(N\)个样本，\(W\)是大小为\(C \times D\)的权重矩阵，LoRA适配器将此操作替换为\(X(W + BA)^T\)，其中\(A\)和\(B\)分别是大小为\(R \times D\)和\(C \times R\)的两个可训练矩阵，且\(R \ll \min(C, D)\)，矩阵\(W\)从可训练参数中移除。矩阵\(A\)用随机高斯值初始化，\(B\)设置为零，因此微调开始时模型计算的输出与原始模型相同。

**知识点讲解:**
- **Low-Rank Matrix（低秩矩阵）**: 低秩矩阵是指秩远小于其行数和列数的矩阵。在LoRA中，低秩矩阵用于表示对原始权重矩阵的微小调整，从而减少微调所需的参数数量。

**English:**
The total number of parameters to optimize with this approach is generally a few percent of the number of parameters in the original model.

**Chinese:**
使用这种方法优化的参数总数通常是原始模型参数数量的百分之几。

**知识点讲解:**
- **Parameter Efficiency（参数效率）**: 参数效率指的是在微调过程中引入的额外参数数量。LoRA通过引入少量低秩矩阵来保持较高的参数效率，从而减少微调的计算成本。

**English:**
The standard procedure to fine-tune a transformer with such adapters is to change only the weight matrices in the attention blocks, and to keep the MLP of the feed-forward blocks unchanged. The same strategy has been used successfully to tune diffusion denoising models by fine-tuning the attention blocks responsible for the text-based conditioning.

**Chinese:**
使用这种适配器微调Transformer的标准程序是仅更改注意力块中的权重矩阵，并保持前馈块的多层感知机（MLP）不变。同样的策略已成功用于通过微调负责基于文本条件的注意力块来调整扩散去噪模型。

**知识点讲解:**
- **Attention Blocks（注意力块）**: 注意力块是Transformer模型中的关键组件，负责计算输入序列中每个元素的重要性。通过微调注意力块，可以调整模型对不同输入的关注程度。

**English:**
Since fine-tuning with LoRA adapters drastically reduces the number of trainable parameters, it reduces the memory footprint required by optimizers such as Adam, which generally store two running average per parameter to optimize. Also, it reduces slightly the computation during the backward pass.

**Chinese:**
由于使用LoRA适配器进行微调大大减少了可训练参数的数量，因此它减少了优化器（如Adam）所需的内存占用，优化器通常为每个参数存储两个运行平均值。此外，它还略微减少了反向传播过程中的计算量。

**知识点讲解:**
- **Optimizer Memory Footprint（优化器内存占用）**: 优化器在训练过程中需要存储每个参数的运行平均值，以计算梯度更新。通过减少可训练参数的数量，LoRA可以显著降低优化器的内存需求。

**English:**
For commercial applications that require a large number of fine-tuned models, the \(AB\) pairs can be stored separately from the original model, which has to be stored only once. And finally, contrary to other type of adapters, the modifications can be integrated into the original architecture, simply by adding \(AB\) to \(W\), resulting in an architecture and parameter count for inference identical to that of the original model.

**Chinese:**
对于需要大量微调模型的商业应用，\(AB\)对可以与原始模型分开存储，原始模型只需存储一次。最后，与其他类型的适配器不同，这些修改可以通过简单地将\(AB\)添加到\(W\)中集成到原始架构中，从而在推理时保持与原始模型相同的架构和参数数量。

**知识点讲解:**
- **Model Integration（模型集成）**: 模型集成是指将微调后的参数与原始模型参数结合，以生成最终的推理模型。LoRA通过简单的矩阵加法实现模型集成，从而保持推理时的模型结构不变。

**English:**
We saw that quantization degrade models’ accuracy only marginally. However, gradient descent requires high precision in both the gradient and the trained parameters, to allow the accumulation of small changes. The QLoRA approach combines a quantized base model and unquantized Low-Rank Adaptation to reduce the memory requirement even more [Dettmers et al., 2023].

**Chinese:**
我们看到量化仅略微降低了模型的准确性。然而，梯度下降需要在梯度和训练参数中保持高精度，以允许小变化的积累。QLoRA方法结合了量化基础模型和未量化的低秩适应，以进一步减少内存需求 [Dettmers et al., 2023]。

**知识点讲解:**
- **QLoRA（量化低秩适应）**: QLoRA是一种结合了量化和低秩适应的技术，通过在量化基础模型上添加未量化的低秩矩阵来进行微调。这种方法在保持模型性能的同时，进一步减少了内存需求。

#### 6.8.4: Model Merging

**English:**
An alternative to the fine-tuning and prompting methods seen in the previous sections consists of combining multiple models with diverse capabilities into a single one, without additional training.

**Chinese:**
与前面章节中看到的微调和提示方法不同，另一种方法是将具有不同能力的多个模型组合成一个模型，而无需额外的训练。

**知识点讲解:**
- **Model Merging（模型合并）**: 模型合并是一种将多个模型的参数组合成一个模型的技术，通常用于结合不同模型的优势，而无需重新训练。

**English:**
Model merging relies on the compatibility between multiple fine-tuned versions of a base model.

**Chinese:**
模型合并依赖于基础模型的多个微调版本之间的兼容性。

**知识点讲解:**
- **Compatibility（兼容性）**: 兼容性指的是不同模型参数之间的相似性或一致性。模型合并要求不同微调版本的模型参数在某种程度上是兼容的，以便能够有效地组合。

**English:**
Ilharco et al. [2022] showed that models obtained by fine-tuning a CLIP base model on several image classification data-sets can be combined in the parameter space, where they exhibit Task Arithmetic properties.

**Chinese:**
Ilharco等人[2022]表明，通过在多个图像分类数据集上微调CLIP基础模型获得的模型可以在参数空间中组合，这些模型表现出任务算术（Task Arithmetic）特性。

**知识点讲解:**
- **Task Arithmetic（任务算术）**: 任务算术是一种模型合并技术，通过在参数空间中组合不同任务的微调模型，生成一个能够处理多个任务的单一模型。

**English:**
Formally, let \(\theta\) be the parameter vector of a pretrained model, and for \(t = 1, \ldots, T\), let \(\theta_t\) and \(\tau_t = \theta_t - \theta\) be respectively the parameters after fine-tuning on task \(t\) and the corresponding residual. Experiments show that the model with parameters \(\theta + \tau_1 + \cdots + \tau_T\) exhibits multi-task capabilities. Similarly, subtracting a \(\tau_t\) degrades the performance on the corresponding task.

**Chinese:**
形式上，设\(\theta\)为预训练模型的参数向量，对于\(t = 1, \ldots, T\)，设\(\theta_t\)和\(\tau_t = \theta_t - \theta\)分别为在任务\(t\)上微调后的参数和相应的残差。实验表明，具有参数\(\theta + \tau_1 + \cdots + \tau_T\)的模型表现出多任务能力。类似地，减去\(\tau_t\)会降低相应任务的性能。

**知识点讲解:**
- **Residual Parameters（残差参数）**: 残差参数是指微调后的模型参数与原始模型参数之间的差异。通过组合这些残差参数，可以生成一个能够处理多个任务的单一模型。

**English:**
Methods have been developed to reduce the interference between the different residuals and improve the performance when the number of tasks is large.

**Chinese:**
已经开发了一些方法来减少不同残差之间的干扰，并在任务数量较多时提高性能。

**知识点讲解:**
- **Interference Reduction（干扰减少）**: 干扰减少是指通过调整模型合并策略，减少不同任务残差之间的冲突，从而提高多任务模型的性能。

**English:**
An alternative to merging models in parameter space is to recombine their layers. Akiba et al. [2024] combine merging the parameters and re-combining layers, and rely on a stochastic optimization to deal with the combinatorial explosion. Experiments with three fine-tuned versions of Mistral-7B [Jiang et al., 2023] show that combining these two merging strategies outperforms both of them.

**Chinese:**
在参数空间中合并模型的另一种方法是重新组合它们的层。Akiba等人[2024]结合了参数合并和层重新组合，并依赖随机优化来处理组合爆炸。使用Mistral-7B [Jiang et al., 2023]的三个微调版本进行的实验表明，结合这两种合并策略优于单独使用其中任何一种。

**知识点讲解:**
- **Layer Recombination（层重新组合）**: 层重新组合是一种模型合并技术，通过重新组合不同模型的层来生成一个新的模型。这种方法可以结合不同模型的优势，生成更强大的多任务模型。
- **Stochastic Optimization（随机优化）**: 随机优化是一种通过随机搜索来寻找最优解的方法。在模型合并中，随机优化可以用于处理组合爆炸问题，找到最佳的模型组合策略。

### 总结

本章介绍了在计算资源有限的情况下，如何通过提示工程、量化和适配器等技术来优化大型语言模型的推理和微调。提示工程通过精心设计的输入提示引导模型生成特定输出；量化通过减少模型参数的精度来降低内存占用和计算成本；适配器则通过在预训练模型中添加少量可训练参数来适应新任务。这些技术使得大型语言模型能够在资源有限的设备上高效运行，同时保持较高的性能。

### Chapter 6.8.4: Model Merging (Continued)

#### 6.8.4.1: Model Merging in Parameter Space

**English:**
Model merging in parameter space involves combining the parameters of multiple models trained on different tasks to create a single model capable of performing multiple tasks. This approach leverages the idea that the differences between models (residuals) can be additive, allowing for the combination of task-specific knowledge without retraining.

**Chinese:**
参数空间中的模型合并涉及将多个在不同任务上训练的模型的参数组合起来，创建一个能够执行多个任务的单一模型。这种方法利用了模型之间的差异（残差）可以相加的思想，从而在不重新训练的情况下结合特定任务的知识。

**知识点讲解:**
- **Parameter Space（参数空间）**: 参数空间是指模型参数的所有可能组合。通过在这个空间中组合不同模型的参数，可以生成一个能够处理多个任务的单一模型。
- **Residuals（残差）**: 残差是指微调后的模型参数与原始模型参数之间的差异。这些差异可以用于组合不同任务的知识。

**English:**
The key insight is that the residuals from fine-tuning on different tasks can be combined linearly to create a model that performs well on all tasks. This is particularly useful when the tasks are related, as the residuals will share some common structure.

**Chinese:**
关键见解是，在不同任务上微调的残差可以线性组合，从而创建一个在所有任务上表现良好的模型。这在任务相关时尤其有用，因为残差将共享一些共同的结构。

**知识点讲解:**
- **Linear Combination（线性组合）**: 线性组合是指将多个向量或矩阵通过加权求和的方式组合起来。在模型合并中，线性组合用于将不同任务的残差结合起来。
- **Task Relatedness（任务相关性）**: 任务相关性指的是不同任务之间的相似性或关联性。任务相关性越高，模型合并的效果通常越好。

#### 6.8.4.2: Layer Recombination

**English:**
Layer recombination is an alternative approach to model merging that involves recombining the layers of different models to create a new model. This method is particularly useful when the models have different architectures or when the tasks require different levels of abstraction.

**Chinese:**
层重新组合是模型合并的另一种方法，涉及重新组合不同模型的层以创建一个新模型。这种方法在模型具有不同架构或任务需要不同抽象级别时特别有用。

**知识点讲解:**
- **Layer Recombination（层重新组合）**: 层重新组合是指将不同模型的层重新组合以生成一个新模型。这种方法可以结合不同模型的优势，生成更强大的多任务模型。
- **Abstraction Levels（抽象级别）**: 抽象级别指的是模型在处理数据时所使用的抽象程度。不同任务可能需要不同抽象级别的模型层。

**English:**
For example, one might combine the lower layers of a model trained on image classification with the higher layers of a model trained on object detection to create a model that can perform both tasks effectively.

**Chinese:**
例如，可以将一个在图像分类上训练的模型的较低层与一个在目标检测上训练的模型的较高层结合起来，创建一个能够有效执行这两个任务的模型。

**知识点讲解:**
- **Lower Layers（较低层）**: 较低层通常负责提取输入数据的基本特征，如边缘和纹理。
- **Higher Layers（较高层）**: 较高层通常负责处理更抽象的特征，如物体的形状和类别。

#### 6.8.4.3: Stochastic Optimization for Model Merging

**English:**
Stochastic optimization is a technique used to handle the combinatorial explosion that arises when merging multiple models. By randomly sampling different combinations of model parameters or layers, one can find a combination that performs well across multiple tasks.

**Chinese:**
随机优化是一种用于处理合并多个模型时出现的组合爆炸问题的技术。通过随机采样模型参数或层的不同组合，可以找到一个在多个任务上表现良好的组合。

**知识点讲解:**
- **Combinatorial Explosion（组合爆炸）**: 组合爆炸指的是当组合数量随着元素数量的增加而急剧增加的现象。在模型合并中，组合爆炸使得寻找最佳组合变得困难。
- **Random Sampling（随机采样）**: 随机采样是指从所有可能的组合中随机选择一部分进行测试。这种方法可以有效地减少计算量，同时找到较好的组合。

**English:**
Akiba et al. [2024] demonstrated that combining parameter merging and layer recombination with stochastic optimization can lead to better performance than using either method alone. Their experiments with Mistral-7B showed significant improvements in multi-task performance.

**Chinese:**
Akiba等人[2024]证明了将参数合并和层重新组合与随机优化结合起来，可以比单独使用任何一种方法获得更好的性能。他们在Mistral-7B上的实验显示了多任务性能的显著提升。

**知识点讲解:**
- **Multi-task Performance（多任务性能）**: 多任务性能指的是模型在多个任务上的表现。通过结合参数合并和层重新组合，可以生成一个在多个任务上表现良好的模型。

### Chapter 6.9: The Missing Bits

#### 6.9.1: Recurrent Neural Networks

**English:**
Before attention models showed greater performance, Recurrent Neural Networks (RNN) were the standard approach for dealing with temporal sequences such as text or sound samples. These architectures possess an internal hidden state that gets updated each time a component of the sequence is processed. Their main components are layers such as LSTM [Hochreiter and Schmidhuber, 1997] or GRU [Cho et al., 2014].

**Chinese:**
在注意力模型表现出更高性能之前，循环神经网络（RNN）是处理文本或声音样本等时间序列的标准方法。这些架构具有一个内部隐藏状态，每次处理序列的一个组件时都会更新。它们的主要组件是LSTM [Hochreiter and Schmidhuber, 1997] 或 GRU [Cho et al., 2014] 等层。

**知识点讲解:**
- **Recurrent Neural Networks (RNN)（循环神经网络）**: RNN是一种用于处理序列数据的神经网络，具有一个内部隐藏状态，可以在处理序列时保持信息。
- **LSTM（长短期记忆网络）**: LSTM是一种特殊的RNN，通过引入门控机制来解决长序列中的梯度消失问题。
- **GRU（门控循环单元）**: GRU是另一种RNN变体，通过简化LSTM的结构来提高计算效率。

**English:**
Training a recurrent architecture amounts to unfolding it in time, which results in a long composition of operators. This has historically prompted the design of key techniques now used for deep architectures such as rectifiers and gating, a form of skip connections which are modulated by the input.

**Chinese:**
训练循环架构相当于在时间上展开它，这会导致一系列长操作符的组合。这在历史上促使了现在用于深度架构的关键技术的设计，如整流器和门控机制，这是一种由输入调制的跳跃连接形式。

**知识点讲解:**
- **Unfolding in Time（时间展开）**: 时间展开是指将循环神经网络在时间步骤上展开，形成一个前馈网络。这种方法使得RNN可以通过反向传播进行训练。
- **Rectifiers（整流器）**: 整流器是一种激活函数，如ReLU，用于在神经网络中引入非线性。
- **Gating（门控机制）**: 门控机制是一种通过输入信号控制信息流动的技术，常用于LSTM和GRU中。

**English:**
One of the key drawbacks of traditional recurrent architectures is that the structure of the computation \( x_{t+1} = f(x_t) \) imposes to process the input sequence serially, which takes a time proportional to \( T \). In contrast, transformers, for instance, can take advantage of parallel computation, resulting in a constant time if enough computing units are available.

**Chinese:**
传统循环架构的一个关键缺点是计算结构 \( x_{t+1} = f(x_t) \) 要求按顺序处理输入序列，这需要与 \( T \) 成正比的时间。相比之下，例如Transformer可以利用并行计算，如果有足够的计算单元，则可以在恒定时间内完成。

**知识点讲解:**
- **Serial Processing（顺序处理）**: 顺序处理是指按顺序逐个处理输入序列中的元素。这种方法在处理长序列时效率较低。
- **Parallel Computation（并行计算）**: 并行计算是指同时处理多个输入元素。Transformer通过自注意力机制实现了并行计算，从而提高了处理效率。

**English:**
This is addressed by architectures such as QRNN [Bradbury et al., 2016], S4 [Gu et al., 2021], or Mamba [Gu and Dao, 2023], whose recurrent operations are affine so that the \( f^t \) themselves, and consequently the \( x_t = f^t(x_0) \), can be computed in parallel, resulting in a constant time if \( f \) does not depend on \( t \) and \(\log T\) otherwise, again if enough parallel computing units are available.

**Chinese:**
这个问题通过QRNN [Bradbury et al., 2016]、S4 [Gu et al., 2021] 或 Mamba [Gu and Dao, 2023] 等架构得到解决，这些架构的循环操作是仿射的，因此 \( f^t \) 本身以及 \( x_t = f^t(x_0) \) 可以并行计算，如果 \( f \) 不依赖于 \( t \)，则可以在恒定时间内完成，否则在 \(\log T\) 时间内完成，前提是有足够的并行计算单元。

**知识点讲解:**
- **Affine Operations（仿射操作）**: 仿射操作是指线性变换加上一个偏置项。仿射操作可以并行计算，从而提高计算效率。
- **Parallel Computing Units（并行计算单元）**: 并行计算单元是指能够同时执行多个计算任务的硬件资源，如GPU或TPU。

#### 6.9.2: Autoencoder

**English:**
An autoencoder is a model that maps an input signal, possibly of high dimension, to a low-dimension latent representation, and then maps it back to the original signal, ensuring that information has been preserved. We saw it in § 6.1 for denoising, but it can also be used to automatically discover a meaningful low-dimension representation of the data.

**Chinese:**
自编码器是一种将输入信号（可能是高维的）映射到低维潜在表示，然后再将其映射回原始信号的模型，确保信息得以保留。我们在§6.1中看到了它用于去噪，但它也可以用于自动发现数据的有意义的低维表示。

**知识点讲解:**
- **Autoencoder（自编码器）**: 自编码器是一种无监督学习模型，通过将输入数据压缩到低维表示并重建输入数据来学习数据的特征。
- **Latent Representation（潜在表示）**: 潜在表示是指自编码器将输入数据压缩到的低维表示。这种表示通常包含数据的关键特征。

**English:**
The Variational Autoencoder (VAE) proposed by Kingma and Welling [2013] is a generative model with a similar structure. It imposes, through the loss, a pre-defined distribution on the latent representation. This allows, after training, the generation of new samples by sampling the latent representation according to this imposed distribution and then mapping back through the decoder.

**Chinese:**
Kingma和Welling [2013] 提出的变分自编码器（VAE）是一种具有类似结构的生成模型。它通过损失函数在潜在表示上施加预定义的分布。这使得在训练后，可以通过按照这种施加的分布对潜在表示进行采样，然后通过解码器映射回来生成新样本。

**知识点讲解:**
- **Variational Autoencoder (VAE)（变分自编码器）**: VAE是一种生成模型，通过在潜在表示上施加概率分布来生成新样本。
- **Pre-defined Distribution（预定义分布）**: 预定义分布是指在训练过程中施加在潜在表示上的概率分布，通常选择高斯分布。

#### 6.9.3: Generative Adversarial Networks

**English:**
Another approach to density modeling is the Generative Adversarial Networks (GAN) introduced by Goodfellow et al. [2014]. This method combines a generator, which takes a random input following a fixed distribution as input and produces a structured signal such as an image, and a discriminator, which takes a sample as input and predicts whether it comes from the training set or if it was generated by the generator.

**Chinese:**
另一种密度建模方法是Goodfellow等人[2014] 提出的生成对抗网络（GAN）。这种方法结合了一个生成器，它接受一个遵循固定分布的随机输入并生成一个结构化信号（如图像），以及一个判别器，它接受一个样本作为输入并预测它是来自训练集还是由生成器生成的。

**知识点讲解:**
- **Generative Adversarial Networks (GAN)（生成对抗网络）**: GAN是一种生成模型，通过生成器和判别器之间的对抗训练来生成逼真的样本。
- **Generator（生成器）**: 生成器是GAN的一部分，负责生成逼真的样本。
- **Discriminator（判别器）**: 判别器是GAN的另一部分，负责区分真实样本和生成样本。

**English:**
Training optimizes the discriminator to minimize a standard cross-entropy loss, and the generator to maximize the discriminator’s loss. It results in a generator that produces samples that are indistinguishable from real data.

**Chinese:**
训练优化判别器以最小化标准交叉熵损失，并优化生成器以最大化判别器的损失。这导致生成器生成与真实数据无法区分的样本。

**知识点讲解:**
- **Cross-entropy Loss（交叉熵损失）**: 交叉熵损失是一种用于衡量预测分布与真实分布之间差异的损失函数。在GAN中，判别器使用交叉熵损失来区分真实样本和生成样本。
- **Adversarial Training（对抗训练）**: 对抗训练是指生成器和判别器之间的对抗过程，生成器试图生成逼真的样本，而判别器试图区分真实样本和生成样本。

#### 6.9.4: Graph Neural Networks

**English:**
Many applications require processing signals which are not organized regularly on a grid. For instance, proteins, 3D meshes, geographic locations, or social interactions are more naturally structured as graphs. Standard convolutional networks or even attention models are poorly adapted to process such data, and the tool of choice for such a task is Graph Neural Networks (GNN) [Scarselli et al., 2009].

**Chinese:**
许多应用需要处理未在网格上规则组织的信号。例如，蛋白质、3D网格、地理位置或社交互动更自然地结构化为图。标准卷积网络甚至注意力模型都不太适合处理此类数据，而图神经网络（GNN）[Scarselli et al., 2009] 是处理此类任务的首选工具。

**知识点讲解:**
- **Graph Neural Networks (GNN)（图神经网络）**: GNN是一种用于处理图结构数据的神经网络，通过在图中的节点之间传递信息来学习图的特征。
- **Graph Structure（图结构）**: 图结构是指由节点和边组成的数据结构，常用于表示复杂的关系和交互。

**English:**
These models are composed of layers that compute activations at each vertex by combining linearly the activations located at its immediate neighboring vertices. This operation is very similar to a standard convolution, except that the data structure does not reflect any geometrical information associated with the feature vectors they carry.

**Chinese:**
这些模型由层组成，这些层通过线性组合位于其直接相邻顶点的激活来计算每个顶点的激活。这种操作与标准卷积非常相似，只是数据结构不反映它们所携带的特征向量的任何几何信息。

**知识点讲解:**
- **Vertex Activation（顶点激活）**: 顶点激活是指图神经网络中每个节点的激活值，通过与其邻居节点的激活值线性组合来计算。
- **Geometrical Information（几何信息）**: 几何信息是指数据在空间中的位置和形状信息。在图神经网络中，数据结构不反映几何信息，而是反映节点之间的关系。

#### 6.9.5: Self-Supervised Learning

**English:**
As stated in § 7.1, even though they are trained only to predict the next word, Large Language Models trained on large unlabeled datasets such as GPT (see § 5.3) are able to solve various tasks, such as identifying the grammatical role of a word, answering questions, or even translating from one language to another [Radford et al., 2019].

**Chinese:**
如§7.1所述，尽管它们仅被训练来预测下一个单词，但在大型未标记数据集（如GPT，参见§5.3）上训练的大型语言模型能够解决各种任务，例如识别单词的语法角色、回答问题，甚至从一种语言翻译到另一种语言 [Radford et al., 2019]。

**知识点讲解:**
- **Self-Supervised Learning（自监督学习）**: 自监督学习是一种无监督学习方法，通过设计预训练任务来学习数据的特征表示，而无需人工标注的标签。
- **Large Language Models（大型语言模型）**: 大型语言模型是指在大规模文本数据上训练的模型，能够生成和理解自然语言。

**English:**
Such models constitute one category of a larger class of methods that fall under the name of self-supervised learning, and try to take advantage of unlabeled datasets [Balestriero et al., 2023].

**Chinese:**
这些模型构成了自监督学习这一大类方法中的一类，试图利用未标记的数据集 [Balestriero et al., 2023]。

**知识点讲解:**
- **Unlabeled Datasets（未标记数据集）**: 未标记数据集是指没有人工标注的数据集。自监督学习通过设计预训练任务来利用这些数据集进行训练。

**English:**
The key principle of these methods is to define a task that does not require labels but necessitates feature representations which are useful for the real task of interest, for which a small labeled dataset exists. In computer vision, for instance, image features can be optimized so that they are invariant to data transformations that do not change the semantic content of the image, while being statistically uncorrelated [Zbontar et al., 2021].

**Chinese:**
这些方法的关键原则是定义一个不需要标签但需要特征表示的任务，这些特征表示对于感兴趣的真实任务是有用的，而真实任务存在一个小型标记数据集。例如，在计算机视觉中，可以优化图像特征，使其对不改变图像语义内容的数据变换保持不变，同时在统计上不相关 [Zbontar et al., 2021]。

**知识点讲解:**
- **Feature Representations（特征表示）**: 特征表示是指数据在模型中的表示形式。自监督学习通过设计预训练任务来学习有用的特征表示。
- **Data Transformations（数据变换）**: 数据变换是指对数据进行旋转、缩放、裁剪等操作。自监督学习通过使特征表示对这些变换保持不变来学习鲁棒的特征。

**English:**
In both NLP and computer vision, a powerful generic strategy is to train a model to recover a masked part of the input, or to predict the relative position of patches, or to reconstruct the input from a corrupted version [Devlin et al., 2018; Zhou et al., 2021].

**Chinese:**
在自然语言处理和计算机视觉中，一个强大的通用策略是训练模型来恢复输入的掩码部分，或预测补丁的相对位置，或从损坏的版本中重建输入 [Devlin et al., 2018; Zhou et al., 2021]。

**知识点讲解:**
- **Masked Part（掩码部分）**: 掩码部分是指输入中被隐藏或遮挡的部分。通过恢复掩码部分，模型可以学习到输入数据的完整表示。
- **Relative Position（相对位置）**: 相对位置是指输入中不同部分之间的位置关系。通过预测相对位置，模型可以学习到输入数据的空间结构。
- **Reconstruction（重建）**: 重建是指从损坏的输入中恢复原始输入。通过重建任务，模型可以学习到输入数据的关键特征。

### 总结

本章介绍了在计算资源有限的情况下，如何通过提示工程、量化和适配器等技术来优化大型语言模型的推理和微调。提示工程通过精心设计的输入提示引导模型生成特定输出；量化通过减少模型参数的精度来降低内存占用和计算成本；适配器则通过在预训练模型中添加少量可训练参数来适应新任务。这些技术使得大型语言模型能够在资源有限的设备上高效运行，同时保持较高的性能。


### Chapter 6.9.5: Self-Supervised Learning (Continued)

#### 6.9.5.1: Masked Language Modeling

**English:**
Masked Language Modeling (MLM) is a common self-supervised learning task in natural language processing. In this task, certain words in a sentence are masked, and the model is trained to predict the masked words based on the surrounding context. This approach allows the model to learn rich representations of language without requiring labeled data.

**Chinese:**
掩码语言建模（Masked Language Modeling, MLM）是自然语言处理中常见的自监督学习任务。在这个任务中，句子中的某些词被掩码，模型被训练为根据上下文预测被掩码的词。这种方法使模型能够在不需要标注数据的情况下学习丰富的语言表示。

**知识点讲解:**
- **Masked Language Modeling (MLM)（掩码语言建模）**: MLM是一种自监督学习任务，通过预测被掩码的词来训练模型。这种方法广泛应用于BERT等预训练语言模型中。
- **Context（上下文）**: 上下文是指句子中围绕某个词的词语或句子结构。通过利用上下文信息，模型可以更好地理解语言的含义。

**English:**
For example, in the sentence "The cat sat on the [MASK]", the model might be trained to predict that the masked word is "mat" based on the context provided by the other words in the sentence.

**Chinese:**
例如，在句子“The cat sat on the [MASK]”中，模型可能会被训练为根据句子中其他词提供的上下文预测被掩码的词是“mat”。

**知识点讲解:**
- **Prediction（预测）**: 预测是指模型根据输入数据推断出缺失或未来的信息。在MLM中，模型通过预测被掩码的词来学习语言表示。

#### 6.9.5.2: Contrastive Learning

**English:**
Contrastive learning is another self-supervised learning technique that involves training a model to distinguish between similar and dissimilar pairs of data points. This is often done by maximizing the similarity between positive pairs (e.g., two different views of the same image) and minimizing the similarity between negative pairs (e.g., views of different images).

**Chinese:**
对比学习是另一种自监督学习技术，涉及训练模型以区分相似和不相似的数据点对。这通常通过最大化正对（例如，同一图像的两个不同视图）之间的相似性并最小化负对（例如，不同图像的视图）之间的相似性来实现。

**知识点讲解:**
- **Contrastive Learning（对比学习）**: 对比学习是一种通过比较数据点对来学习特征表示的技术。这种方法在计算机视觉和自然语言处理中都有广泛应用。
- **Positive Pairs（正对）**: 正对是指来自同一数据源的不同视图或样本。通过最大化正对之间的相似性，模型可以学习到数据的共同特征。
- **Negative Pairs（负对）**: 负对是指来自不同数据源的样本。通过最小化负对之间的相似性，模型可以学习到数据的区分性特征。

**English:**
In computer vision, contrastive learning has been used to train models on large unlabeled datasets by creating positive pairs through data augmentation techniques such as cropping, rotating, or color jittering.

**Chinese:**
在计算机视觉中，对比学习已通过数据增强技术（如裁剪、旋转或颜色抖动）创建正对，从而在大型未标记数据集上训练模型。

**知识点讲解:**
- **Data Augmentation（数据增强）**: 数据增强是指通过对原始数据进行变换（如裁剪、旋转、颜色抖动等）来生成新的训练样本。这种方法可以增加数据的多样性，提高模型的泛化能力。

#### 6.9.5.3: Reconstruction-Based Methods

**English:**
Reconstruction-based methods involve training a model to reconstruct the input data from a corrupted or partial version of it. This can be done using autoencoders, where the model learns to encode the input into a lower-dimensional representation and then decode it back to the original input.

**Chinese:**
基于重建的方法涉及训练模型从损坏或部分版本的输入数据中重建原始输入。这可以使用自编码器来实现，模型学习将输入编码为低维表示，然后将其解码回原始输入。

**知识点讲解:**
- **Reconstruction-Based Methods（基于重建的方法）**: 基于重建的方法是一种自监督学习技术，通过重建输入数据来学习特征表示。这种方法常用于图像和文本数据的特征学习。
- **Autoencoders（自编码器）**: 自编码器是一种神经网络，通过将输入数据压缩到低维表示并重建输入数据来学习特征。

**English:**
For example, in image denoising, the model is trained to reconstruct a clean image from a noisy version of it. This forces the model to learn robust features that are invariant to noise.

**Chinese:**
例如，在图像去噪中，模型被训练为从噪声版本中重建干净的图像。这迫使模型学习对噪声不变的鲁棒特征。

**知识点讲解:**
- **Image Denoising（图像去噪）**: 图像去噪是指从噪声图像中恢复原始图像的过程。通过训练模型进行图像去噪，可以学习到对噪声不变的图像特征。

### Chapter 6.10: Conclusion

**English:**
Self-supervised learning has emerged as a powerful paradigm for training models on large unlabeled datasets. By designing tasks that do not require labeled data, such as masked language modeling, contrastive learning, and reconstruction-based methods, researchers can leverage the vast amounts of unlabeled data available to train more robust and generalizable models.

**Chinese:**
自监督学习已成为在大型未标记数据集上训练模型的有力范式。通过设计不需要标注数据的任务，如掩码语言建模、对比学习和基于重建的方法，研究人员可以利用大量未标记数据来训练更鲁棒和可泛化的模型。

**知识点讲解:**
- **Self-Supervised Learning（自监督学习）**: 自监督学习是一种利用未标记数据训练模型的方法，通过设计预训练任务来学习数据的特征表示。
- **Robust Models（鲁棒模型）**: 鲁棒模型是指对输入数据中的噪声和变化具有较强适应能力的模型。自监督学习可以帮助模型学习到对噪声和变化不变的特征。
- **Generalizable Models（可泛化模型）**: 可泛化模型是指在新数据上表现良好的模型。自监督学习通过利用大量未标记数据，可以提高模型的泛化能力。

**English:**
These techniques have been particularly successful in natural language processing and computer vision, where large amounts of unlabeled data are readily available. As the field continues to evolve, we can expect to see even more innovative applications of self-supervised learning in other domains.

**Chinese:**
这些技术在自然语言处理和计算机视觉中特别成功，因为这些领域中有大量未标记数据可用。随着该领域的不断发展，我们可以预期在其他领域看到更多自监督学习的创新应用。

**知识点讲解:**
- **Natural Language Processing（自然语言处理）**: 自然语言处理是指利用计算机处理和理解人类语言的技术。自监督学习在自然语言处理中取得了显著的成功。
- **Computer Vision（计算机视觉）**: 计算机视觉是指利用计算机处理和理解图像和视频的技术。自监督学习在计算机视觉中也取得了显著的成功。

### 总结

本章介绍了自监督学习的几种主要技术，包括掩码语言建模、对比学习和基于重建的方法。这些技术通过设计不需要标注数据的任务，利用大量未标记数据来训练模型，从而提高了模型的鲁棒性和泛化能力。自监督学习在自然语言处理和计算机视觉中取得了显著的成功，并有望在其他领域得到更广泛的应用。

#### 6.10.1: Future Directions in Self-Supervised Learning

**English:**
As self-supervised learning continues to advance, several promising directions are emerging. One area of focus is the development of more sophisticated pretext tasks that can better capture the underlying structure of the data. Another direction is the integration of self-supervised learning with other learning paradigms, such as reinforcement learning and meta-learning, to create more versatile and adaptive models.

**Chinese:**
随着自监督学习的不断发展，一些有前景的方向正在涌现。一个重点领域是开发更复杂的预训练任务，以更好地捕捉数据的底层结构。另一个方向是将自监督学习与其他学习范式（如强化学习和元学习）结合起来，以创建更通用和自适应的模型。

**知识点讲解:**
- **Pretext Tasks（预训练任务）**: 预训练任务是指自监督学习中设计的任务，用于在没有标注数据的情况下训练模型。更复杂的预训练任务可以帮助模型更好地理解数据的结构。
- **Reinforcement Learning（强化学习）**: 强化学习是一种通过与环境交互来学习策略的学习范式。将自监督学习与强化学习结合，可以提高模型在动态环境中的适应能力。
- **Meta-Learning（元学习）**: 元学习是指学习如何学习的技术。通过将自监督学习与元学习结合，可以创建能够快速适应新任务的模型。

**English:**
Additionally, there is growing interest in applying self-supervised learning to multimodal data, where the goal is to learn representations that are consistent across different types of data, such as text, images, and audio. This can enable more powerful and flexible models that can understand and generate content across multiple modalities.

**Chinese:**
此外，将自监督学习应用于多模态数据的兴趣日益增长，其目标是学习跨不同类型数据（如文本、图像和音频）的一致表示。这可以创建更强大和灵活的模型，能够理解和生成跨多种模态的内容。

**知识点讲解:**
- **Multimodal Data（多模态数据）**: 多模态数据是指包含多种类型数据（如文本、图像、音频等）的数据集。通过自监督学习，可以学习到跨模态的一致表示。
- **Consistent Representations（一致表示）**: 一致表示是指在不同模态之间共享的特征表示。通过学习一致表示，模型可以更好地理解和生成跨模态的内容。

#### 6.10.2: Challenges and Limitations

**English:**
Despite its successes, self-supervised learning still faces several challenges. One major challenge is the design of pretext tasks that can effectively capture the complexity of real-world data. Another challenge is the scalability of self-supervised learning methods, particularly when dealing with extremely large datasets and models.

**Chinese:**
尽管取得了成功，自监督学习仍然面临一些挑战。一个主要挑战是设计能够有效捕捉现实世界数据复杂性的预训练任务。另一个挑战是自监督学习方法的可扩展性，特别是在处理极大数据集和模型时。

**知识点讲解:**
- **Complexity of Real-World Data（现实世界数据的复杂性）**: 现实世界数据通常具有复杂的结构和噪声，设计能够有效捕捉这些复杂性的预训练任务是一个挑战。
- **Scalability（可扩展性）**: 可扩展性是指方法在处理大规模数据时的效率和性能。自监督学习方法在处理极大数据集和模型时，需要解决计算资源和时间成本的问题。

**English:**
Furthermore, while self-supervised learning can reduce the need for labeled data, it does not eliminate it entirely. In many cases, fine-tuning on a small amount of labeled data is still necessary to achieve optimal performance on specific tasks.

**Chinese:**
此外，尽管自监督学习可以减少对标注数据的需求，但它并不能完全消除这种需求。在许多情况下，仍然需要在少量标注数据上进行微调，以在特定任务上实现最佳性能。

**知识点讲解:**
- **Fine-Tuning（微调）**: 微调是指在预训练模型的基础上，使用特定任务的数据对模型进行进一步训练，以适应特定任务的需求。尽管自监督学习可以减少对标注数据的需求，但在某些任务上，微调仍然是必要的。

#### 6.10.3: Ethical Considerations

**English:**
As with any powerful technology, self-supervised learning raises important ethical considerations. One concern is the potential for bias in the learned representations, particularly when the training data contains biases. Ensuring that self-supervised learning models are fair and unbiased is an ongoing area of research.

**Chinese:**
与任何强大的技术一样，自监督学习也引发了重要的伦理考虑。一个担忧是学习到的表示可能存在偏见，特别是在训练数据包含偏见的情况下。确保自监督学习模型公平且无偏见是一个持续的研究领域。

**知识点讲解:**
- **Bias in Representations（表示中的偏见）**: 表示中的偏见是指模型在学习过程中可能继承训练数据中的偏见，导致不公平的结果。消除表示中的偏见是自监督学习中的一个重要挑战。
- **Fairness（公平性）**: 公平性是指模型在不同群体之间表现一致，不因性别、种族等因素而产生偏见。确保自监督学习模型的公平性是一个重要的伦理问题。

**English:**
Another ethical consideration is the potential for misuse of self-supervised learning models, such as in the creation of deepfakes or other forms of synthetic media. It is important for researchers and practitioners to consider the societal impact of their work and to develop guidelines for responsible use.

**Chinese:**
另一个伦理考虑是自监督学习模型可能被滥用的风险，例如用于创建深度伪造或其他形式的合成媒体。研究人员和实践者需要考虑其工作的社会影响，并制定负责任使用的指南。

**知识点讲解:**
- **Misuse of Models（模型的滥用）**: 模型的滥用是指将模型用于不道德或非法的目的，如生成虚假信息或侵犯隐私。防止模型滥用是自监督学习中的一个重要伦理问题。
- **Responsible Use（负责任的使用）**: 负责任的使用是指在使用技术时考虑其潜在的社会影响，并采取措施防止滥用。制定负责任使用的指南是确保技术造福社会的重要步骤。

### 总结

本章总结了自监督学习的未来方向、挑战和伦理考虑。随着自监督学习的不断发展，开发更复杂的预训练任务、结合其他学习范式以及应用于多模态数据是未来的重要方向。然而，自监督学习仍然面临设计有效预训练任务、提高可扩展性以及确保公平性和防止滥用等挑战。研究人员和实践者需要在这些领域继续努力，以确保自监督学习技术的健康发展和社会效益。

### Chapter 6.11: Final Thoughts

#### 6.11.1: The Impact of Self-Supervised Learning

**English:**
Self-supervised learning has already had a profound impact on the field of machine learning, enabling the training of powerful models on vast amounts of unlabeled data. This has led to significant advancements in natural language processing, computer vision, and other domains, where labeled data is often scarce or expensive to obtain.

**Chinese:**
自监督学习已经对机器学习领域产生了深远的影响，使得能够在大量未标记数据上训练强大的模型。这导致了自然语言处理、计算机视觉和其他领域的显著进步，在这些领域中，标注数据通常稀缺或获取成本高昂。

**知识点讲解:**
- **Impact on Machine Learning（对机器学习的影响）**: 自监督学习通过利用未标记数据，显著扩展了机器学习的应用范围，特别是在数据标注成本高的领域。
- **Advancements in NLP and CV（自然语言处理和计算机视觉的进步）**: 自监督学习在自然语言处理和计算机视觉中的应用，如BERT和对比学习，极大地推动了这些领域的发展。

**English:**
As the field continues to evolve, self-supervised learning is likely to play an increasingly important role in the development of artificial intelligence. By reducing the reliance on labeled data, it opens up new possibilities for training models in domains where data annotation is challenging or impractical.

**Chinese:**
随着该领域的不断发展，自监督学习很可能在人工智能的发展中扮演越来越重要的角色。通过减少对标注数据的依赖，它为在数据标注具有挑战性或不可行的领域中训练模型开辟了新的可能性。

**知识点讲解:**
- **Reduced Reliance on Labeled Data（减少对标注数据的依赖）**: 自监督学习通过利用未标记数据，减少了对昂贵且耗时的数据标注的依赖，从而扩展了机器学习的应用范围。
- **New Possibilities（新的可能性）**: 自监督学习为在数据标注困难的领域（如医学影像分析）中训练模型提供了新的可能性。

#### 6.11.2: The Role of the Research Community

**English:**
The research community plays a crucial role in advancing self-supervised learning. By developing new techniques, sharing datasets, and fostering collaboration, researchers can accelerate progress and ensure that the benefits of self-supervised learning are widely accessible.

**Chinese:**
研究社区在推动自监督学习方面发挥着关键作用。通过开发新技术、共享数据集和促进合作，研究人员可以加速进展，并确保自监督学习的好处能够广泛普及。

**知识点讲解:**
- **Research Community（研究社区）**: 研究社区是指从事科学研究的人员和组织，通过合作和知识共享推动技术进步。
- **Collaboration（合作）**: 合作是指研究人员之间的协同工作，通过共享资源和知识，加速技术的发展和普及。

**English:**
Open-source initiatives and public datasets have been instrumental in the rapid adoption of self-supervised learning techniques. Continued support for these initiatives will be essential for maintaining the momentum of progress in the field.

**Chinese:**
开源项目和公共数据集在自监督学习技术的快速采用中发挥了重要作用。继续支持这些项目对于保持该领域的进展势头至关重要。

**知识点讲解:**
- **Open-Source Initiatives（开源项目）**: 开源项目是指公开源代码的软件项目，允许任何人使用、修改和分发。开源项目在推动技术普及和创新方面发挥了重要作用。
- **Public Datasets（公共数据集）**: 公共数据集是指公开可用的数据集，供研究人员使用。公共数据集为自监督学习的研究和应用提供了重要的资源。

#### 6.11.3: The Future of Self-Supervised Learning

**English:**
Looking ahead, the future of self-supervised learning is bright. As techniques continue to improve and new applications are discovered, self-supervised learning will likely become a cornerstone of artificial intelligence, enabling the development of more intelligent, adaptable, and efficient models.

**Chinese:**
展望未来，自监督学习的前景光明。随着技术的不断改进和新应用的发现，自监督学习很可能成为人工智能的基石，推动更智能、更适应性强和更高效的模型的发展。

**知识点讲解:**
- **Future of Self-Supervised Learning（自监督学习的未来）**: 自监督学习的未来充满希望，随着技术的进步，它将在人工智能中发挥越来越重要的作用。
- **Intelligent and Adaptable Models（智能和适应性强的模型）**: 自监督学习有助于开发更智能和适应性强的模型，这些模型能够更好地理解和处理复杂的数据。

**English:**
Ultimately, the goal of self-supervised learning is to create models that can learn from the world in a way that is more akin to human learning—by observing, interacting, and making sense of the environment without the need for explicit supervision. Achieving this goal will bring us closer to realizing the full potential of artificial intelligence.

**Chinese:**
最终，自监督学习的目标是创建能够以更类似于人类学习的方式从世界中学习的模型——通过观察、互动和理解环境，而不需要明确的监督。实现这一目标将使我们更接近实现人工智能的全部潜力。

**知识点讲解:**
- **Human-Like Learning（类人学习）**: 类人学习是指模型能够像人类一样通过观察和互动来学习，而不需要明确的指导。自监督学习是实现类人学习的重要一步。
- **Full Potential of AI（人工智能的全部潜力）**: 人工智能的全部潜力是指AI能够在各种复杂任务中表现出与人类相当甚至超越人类的能力。自监督学习是实现这一目标的关键技术之一。

### 总结

本章总结了自监督学习的影响、研究社区的作用以及未来的发展方向。自监督学习通过减少对标注数据的依赖，推动了自然语言处理、计算机视觉等领域的进步。研究社区通过开发新技术、共享资源和促进合作，加速了自监督学习的发展。未来，自监督学习有望成为人工智能的基石，推动更智能、更适应性强的模型的发展，最终实现类人学习和人工智能的全部潜力。

### Chapter 6.12: Final Words

#### 6.12.1: The Journey Ahead

**English:**
The journey of self-supervised learning is still in its early stages, and there is much to explore and discover. As researchers continue to push the boundaries of what is possible, we can expect to see even more innovative applications and breakthroughs in the coming years.

**Chinese:**
自监督学习的旅程仍处于早期阶段，还有许多需要探索和发现的地方。随着研究人员不断突破可能的界限，我们可以预期在未来几年内看到更多创新的应用和突破。

**知识点讲解:**
- **Early Stages（早期阶段）**: 自监督学习作为一个研究领域，仍处于快速发展阶段，未来有许多潜在的研究方向和应用场景。
- **Innovative Applications（创新应用）**: 自监督学习的创新应用包括跨模态学习、多任务学习等，这些应用将进一步扩展人工智能的能力。

**English:**
The potential of self-supervised learning to transform industries and improve our daily lives is immense. From healthcare to education, from autonomous vehicles to personalized recommendations, the possibilities are endless.

**Chinese:**
自监督学习在改变行业和改善我们日常生活方面的潜力是巨大的。从医疗保健到教育，从自动驾驶到个性化推荐，可能性是无限的。

**知识点讲解:**
- **Transform Industries（改变行业）**: 自监督学习可以通过提高模型的效率和性能，推动各个行业的数字化转型。
- **Improve Daily Lives（改善日常生活）**: 自监督学习可以应用于智能家居、个性化推荐等领域，直接改善人们的日常生活体验。

#### 6.12.2: A Call to Action

**English:**
As we look to the future, it is important for researchers, practitioners, and policymakers to work together to ensure that the benefits of self-supervised learning are realized in a way that is ethical, fair, and beneficial to all. This includes addressing challenges such as bias, scalability, and misuse, while also fostering innovation and collaboration.

**Chinese:**
展望未来，研究人员、从业者和政策制定者需要共同努力，确保自监督学习的好处能够以道德、公平和有益于所有人的方式实现。这包括解决偏见、可扩展性和滥用等挑战，同时促进创新和合作。

**知识点讲解:**
- **Ethical and Fair（道德和公平）**: 在开发和应用自监督学习技术时，必须确保其符合道德标准，并且对所有群体公平。
- **Addressing Challenges（解决挑战）**: 解决自监督学习中的挑战，如数据偏见、模型可扩展性和技术滥用，是实现其潜力的关键。

**English:**
By embracing the principles of openness, transparency, and inclusivity, we can create a future where self-supervised learning not only advances technology but also contributes to a more equitable and sustainable world.

**Chinese:**
通过拥抱开放、透明和包容的原则，我们可以创造一个未来，在这个未来中，自监督学习不仅推动技术进步，还为更公平和可持续的世界做出贡献。

**知识点讲解:**
- **Openness and Transparency（开放和透明）**: 开放和透明是指研究和技术开发过程中的信息公开和共享，这有助于促进合作和创新。
- **Inclusivity（包容性）**: 包容性是指确保技术开发和应用的受益者包括所有群体，特别是那些传统上被边缘化的群体。

#### 6.12.3: Final Thoughts

**English:**
Self-supervised learning represents a significant step forward in the quest to create machines that can learn and reason like humans. While there are still many challenges to overcome, the progress made so far is a testament to the power of innovation and collaboration.

**Chinese:**
自监督学习代表了在创造能够像人类一样学习和推理的机器的道路上迈出的重要一步。尽管仍有许多挑战需要克服，但迄今为止的进展证明了创新和合作的力量。

**知识点讲解:**
- **Human-Like Learning and Reasoning（类人学习和推理）**: 自监督学习是实现类人学习和推理的关键技术之一，它使机器能够从数据中自主学习和理解。
- **Power of Innovation and Collaboration（创新和合作的力量）**: 创新和合作是推动技术进步的核心动力，自监督学习的成功离不开研究社区的共同努力。

**English:**
As we continue to explore the possibilities of self-supervised learning, let us remain committed to the principles of ethical research and responsible innovation. Together, we can unlock the full potential of artificial intelligence and create a better future for all.

**Chinese:**
在我们继续探索自监督学习的可能性时，让我们始终坚持道德研究和负责任创新的原则。共同努力，我们可以释放人工智能的全部潜力，为所有人创造一个更美好的未来。

**知识点讲解:**
- **Ethical Research（道德研究）**: 道德研究是指在研究过程中遵循道德规范，确保研究结果对社会有益。
- **Responsible Innovation（负责任创新）**: 负责任创新是指在技术开发和应用过程中考虑其社会影响，确保技术造福全人类。

### 总结

本章总结了自监督学习的未来前景、研究社区的作用以及实现其潜力的关键原则。自监督学习通过减少对标注数据的依赖，推动了人工智能的进步，并在多个领域展现了巨大的应用潜力。研究人员、从业者和政策制定者需要共同努力，解决技术挑战，确保自监督学习的发展符合道德和公平原则。通过开放、透明和包容的创新，我们可以释放人工智能的全部潜力，为所有人创造一个更美好的未来。




### Chapter 7: Synthesis

#### 7.1 Text Generation

**Text Generation (文本生成)** is the process of creating coherent and contextually relevant text using machine learning models. The standard approach to text synthesis is to use an **attention-based, autoregressive model (基于注意力的自回归模型)**. A very successful model proposed by Radford et al. [2018] is the **GPT (Generative Pre-trained Transformer, 生成式预训练变压器)**, which we described in § 5.3.

**知识点讲解:**
- **Autoregressive Model (自回归模型):** 自回归模型是一种生成模型，它通过逐步生成序列中的每个元素来生成整个序列。在文本生成中，模型会根据前面生成的词来预测下一个词。
- **Attention Mechanism (注意力机制):** 注意力机制允许模型在处理序列时关注序列中的不同部分，从而更好地捕捉长距离依赖关系。

This architecture has been used for very large models, such as OpenAI’s 175-billion-parameter **GPT-3** [Brown et al., 2020]. It is composed of 96 **self-attention blocks (自注意力块)**, each with 96 heads, and processes tokens of dimension 12,288, with a hidden dimension of 49,512 in the **MLPs (多层感知器)** of the attention blocks.

**知识点讲解:**
- **Self-Attention Block (自注意力块):** 自注意力块是Transformer模型的核心组件，它通过计算输入序列中每个元素与其他元素的相关性来捕捉序列中的依赖关系。
- **MLP (多层感知器):** 多层感知器是一种前馈神经网络，通常用于处理非线性关系。

When such a model is trained on a very large dataset, it results in a **Large Language Model (LLM, 大语言模型)**, which exhibits extremely powerful properties. Besides the syntactic and grammatical structure of the language, it has to integrate very diverse knowledge, e.g., to predict the word following “The capital of Japan is”, “if water is heated to 100 Celsius degrees it turns into”, or “because her puppy was sick, Jane was”.

**知识点讲解:**
- **Large Language Model (大语言模型):** 大语言模型是通过在大规模文本数据上进行预训练的模型，能够生成连贯的文本并执行多种语言任务。

This results in particular in the ability to solve **few-shot prediction (少样本预测)**, where only a handful of training examples are available, as illustrated in Figure 7.1. More surprisingly, when given a carefully crafted **prompt (提示)**, it can exhibit abilities for **question answering (问答)**, **problem solving (问题解决)**, and **chain-of-thought (思维链)** that appear eerily close to high-level reasoning [Chowdhery et al., 2022; Bubeck et al., 2023].

**知识点讲解:**
- **Few-Shot Prediction (少样本预测):** 少样本预测是指模型在只有少量训练样本的情况下，能够进行有效的预测。
- **Chain-of-Thought (思维链):** 思维链是一种提示方法，通过让模型生成中间步骤来引导其生成更准确的答案。

Due to these remarkable capabilities, these models are sometimes called **foundation models (基础模型)** [Bommasani et al., 2021].

**知识点讲解:**
- **Foundation Models (基础模型):** 基础模型是指在大规模数据上预训练的模型，能够适应多种下游任务。

However, even though it integrates a very large body of knowledge, such a model may be inadequate for interacting with human users. In many situations, one needs responses that follow the statistics of a helpful dialog with an assistant. This differs from the statistics of available large training sets, which combine novels, encyclopedias, forum messages, and blog posts.

This discrepancy is addressed by **fine-tuning (微调)** such a language model (see § 3.6). The current dominant strategy is **Reinforcement Learning from Human Feedback (RLHF, 基于人类反馈的强化学习)** [Ouyang et al., 2022], which consists of creating small labeled training sets by asking users to either write responses or provide ratings of generated responses. The former can be used as-is to fine-tune the language model, and the latter can be used to train a **reward network (奖励网络)** that predicts the rating and use it as a target to fine-tune the language model with a standard Reinforcement Learning approach.

**知识点讲解:**
- **Fine-Tuning (微调):** 微调是指在预训练模型的基础上，使用特定任务的数据进行进一步训练，以使模型更好地适应特定任务。
- **Reinforcement Learning from Human Feedback (RLHF, 基于人类反馈的强化学习):** RLHF是一种通过人类反馈来指导模型训练的强化学习方法，通常用于生成更符合人类期望的文本。

#### 7.2 Image Generation

**Image Generation (图像生成)** is the process of creating new images using machine learning models. Multiple deep methods have been developed to model and sample from a high-dimensional density. A powerful approach for image synthesis relies on inverting a **diffusion process (扩散过程)**. Such a generative model is referred to, somewhat incorrectly, as a **diffusion model (扩散模型)**.

**知识点讲解:**
- **Diffusion Model (扩散模型):** 扩散模型是一种生成模型，它通过逐步添加噪声来破坏数据，然后学习如何逆转这个过程来生成新的数据。

The principle consists of defining analytically a process that gradually degrades any sample, and consequently transforms the complex and unknown density of the data into a simple and well-known density such as a normal, and training a deep architecture to invert this degradation process [Ho et al., 2020].

**知识点讲解:**
- **Degradation Process (退化过程):** 退化过程是指通过逐步添加噪声或其他形式的干扰来破坏数据的过程。
- **Inversion (逆转):** 逆转是指通过学习如何从噪声数据中恢复原始数据的过程。

Given a fixed \( T \), the diffusion process defines a probability distribution over series of \( T + 1 \) images as follows: sample \( x_0 \) uniformly from the dataset, and then sequentially sample \( x_{t+1} \sim p(x_{t+1} | x_t) \), \( t = 0, \ldots, T - 1 \), where the conditional distribution \( p \) is defined analytically and such that it gradually erases the structure that was in \( x_0 \). The setup should degrade the signal so much that the distribution \( p(x_T) \) has a known analytical form which can be sampled.

**知识点讲解:**
- **Conditional Distribution (条件分布):** 条件分布是指在给定某些条件下，随机变量的分布。

For instance, Ho et al. [2020] normalize the data to have a mean of 0 and a variance of 1, and their diffusion process consists of adding a bit of white noise and re-normalizing the variance to 1. This process exponentially reduces the importance of \( x_0 \), and \( x_t \)’s density can rapidly be approximated with a normal.

**知识点讲解:**
- **White Noise (白噪声):** 白噪声是一种具有均匀功率谱的随机信号，通常用于模拟随机干扰。

The **denoiser (去噪器)** \( f \) is a deep architecture that should model and allow sampling from \( f(x_{t-1}, x_t, t; w) \simeq p(x_{t-1} | x_t) \). It can be shown, thanks to a **variational bound (变分界)**, that if this one-step reverse process is accurate enough, sampling \( x_T \sim p(x_T) \) and denoising \( T \) steps with \( f \) results in \( x_0 \) that follows \( p(x_0) \).

**知识点讲解:**
- **Denoiser (去噪器):** 去噪器是指通过学习如何从噪声数据中恢复原始数据的模型。
- **Variational Bound (变分界):** 变分界是一种数学工具，用于估计复杂分布的性质。

Training \( f \) can be achieved by generating a large number of sequences \( x_0^{(n)}, \ldots, x_T^{(n)} \), picking a \( t_n \) in each, and maximizing

\[
\sum_n \log f \left( x_{t_n-1}^{(n)}, x_{t_n}^{(n)}, t_n; w \right).
\]

Given their diffusion process, Ho et al. [2020] have a denoising of the form:

\[
x_{t-1} | x_t \sim \mathcal{N}(x_t + f(x_t, t; w); \sigma_t), \tag{7.1}
\]

where \( \sigma_t \) is defined analytically.

**知识点讲解:**
- **Normal Distribution (正态分布):** 正态分布是一种常见的连续概率分布，通常用于模拟随机变量的分布。

In practice, such a model initially hallucinates structures by pure luck in the random noise, and then gradually refines the image by reinforcing the most likely continuation of the image obtained thus far.

This approach can be extended to **text-conditioned synthesis (文本条件合成)**, to generate images that match a description. For instance, Nichol et al. [2021] add to the mean of the denoising distribution of Equation 7.1 a bias that goes in the direction of increasing the **CLIP matching score (CLIP匹配分数)** (see § 6.6) between the produced image and the conditioning text description.

**知识点讲解:**
- **Text-Conditioned Synthesis (文本条件合成):** 文本条件合成是指根据文本描述生成与之匹配的图像。
- **CLIP Matching Score (CLIP匹配分数):** CLIP匹配分数是指通过CLIP模型计算的图像和文本之间的相似度分数。

### Chapter 8: The Compute Schism

#### 8.1 Prompt Engineering

**Prompt Engineering (提示工程)** is the process of carefully crafting the input to a machine learning model to elicit the desired output. The simplest strategy to specialize or improve a **Large Language Model (LLM, 大语言模型)** with a limited computational budget is to use prompt engineering, that is, to carefully craft the beginning of the text sequence to bias the autoregressive process [Sahoo et al., 2024]. This approach moves a part of the information traditionally encoded in the model’s parameters to the input.

**知识点讲解:**
- **Prompt Engineering (提示工程):** 提示工程是指通过设计特定的输入提示来引导模型生成期望的输出。
- **Autoregressive Process (自回归过程):** 自回归过程是指模型根据前面生成的元素来预测下一个元素的过程。

We saw in § 7.1 a simple example of **few-shot prediction (少样本预测)**, to use an LLM for a text classification task without fine-tuning. A long and sophisticated prompt allows generalizing this strategy to complex tasks.

Since the prompt’s role is to leverage the “good” biases that were present in the training set, it benefits from surprising strategies such as stating that the response is generated by a skilled professional [Xu et al., 2023].

**知识点讲解:**
- **Few-Shot Prediction (少样本预测):** 少样本预测是指模型在只有少量训练样本的情况下，能够进行有效的预测。

The **context size (上下文大小)** of a language model, that is, the number of tokens it can operate on, directly modulates the quantity of information that can be provided in the prompt. This is mostly constrained by the computational cost of standard attention models, which is quadratic with the context size (see § 4.8).

**知识点讲解:**
- **Context Size (上下文大小):** 上下文大小是指模型能够处理的输入序列的长度。
- **Quadratic Cost (二次成本):** 二次成本是指计算成本随着输入规模的平方增长。

A remarkable type of prompting aims at making the model generate intermediate steps before generating the response itself.

Such a **chain-of-thought (思维链)** is composed of successive steps that are simpler, hence have been better modeled during training, and are predicted more deterministically [Wei et al., 2022; Kojima et al., 2022]. See Figure 8.1 for an example.

**知识点讲解:**
- **Chain-of-Thought (思维链):** 思维链是一种提示方法，通过让模型生成中间步骤来引导其生成更准确的答案。

**Retrieval-Augmented Generation (检索增强生成)**

Prompt engineering can also be put to use to connect a language model to an external knowledge base. It plays the role of a smart interface that allows the end user to formulate questions in natural language and get back a response that combines information that is not encoded in the model’s parameters [Lewis et al., 2020].

**知识点讲解:**
- **Retrieval-Augmented Generation (检索增强生成):** 检索增强生成是指通过检索外部知识库来增强语言模型的生成能力。

For such **Retrieval-Augmented Generation (RAG)**, an **embedding model (嵌入模型)** is used to retrieve documents whose embedding is correlated to that of the user’s query. Then, a prompt is constructed by joining these retrieved documents with instructions to combine them, and the generative model produces the response to the user’s query.

**知识点讲解:**
- **Embedding Model (嵌入模型):** 嵌入模型是指将文本或其他数据映射到低维向量空间的模型。

#### 8.2 Quantization

**Quantization (量化)** is the process of reducing the precision of the numbers used in a model to save memory and computational resources. Although training or generating multiple streams can benefit from high-end parallel computing devices, deployment of a **Large Language Model (LLM, 大语言模型)** for individual use requires generally single-stream inference, which is bounded by memory size and speed far more than by computation.

**知识点讲解:**
- **Quantization (量化):** 量化是指将模型中的浮点数转换为低精度的整数，以减少内存占用和计算成本。
- **Single-Stream Inference (单流推理):** 单流推理是指模型在单个计算设备上进行的推理过程。

As stated in § 2.1, parameters, activations, and gradients are usually encoded with 32 or 16 bits. The precision it provides is necessary for training, to allow gradual changes to accumulate.

However, since activations are the sums of many terms, quantization during inference is mitigated by an averaging effect. This is even more true with large architectures, and models quantized down to 6 or 4 bits per parameter exhibit remarkable performance. Additionally to reducing the memory footprint, quantization also improves inference speed significantly.

**知识点讲解:**
- **Memory Footprint (内存占用):** 内存占用是指模型在运行时所需要的内存空间。
- **Inference Speed (推理速度):** 推理速度是指模型生成输出所需的时间。

This has motivated the development of software to quantize existing models with **Post-Training Quantization (训练后量化)**, and run them in single-stream inference on consumer hardware, such as **llama.cpp** [Llama.cpp, 2023]. This framework implements multiple formats, that apply specific quantization levels for the different weight matrices of a language model. For instance, the quantization may use more bits for the \( W^v \) weights of the attention blocks, and for the weights of the feed-forward blocks.

**知识点讲解:**
- **Post-Training Quantization (训练后量化):** 训练后量化是指在模型训练完成后，对模型进行量化以减少内存占用和计算成本。

An example of llama.cpp’s quantization is **Q4_1**. It quantizes individually sub-blocks of 32 entries of the original weight matrix by storing for each a scaling factor \( d \) and a bias \( m \) in the original FP16 encoding, and encoding each entry \( x \) with 4 bits as a value \( q \in \{0, \ldots, 2^4 - 1\} \). The resulting de-quantized value being \( \bar{x} = dq + m \).

**知识点讲解:**
- **Scaling Factor (缩放因子):** 缩放因子是指用于调整量化后数值范围的参数。
- **Bias (偏置):** 偏置是指用于调整量化后数值的偏移量。

Such a block was encoded originally as 32 values in FP16, hence 64 bytes, while the quantized version needs 4 bytes for \( q \) and \( m \) and \( 32 \cdot 4 \) bits = 16 bytes for the entries, hence a total of 20 bytes.

Such an aggressive quantization surprisingly degrades only marginally the performance of the models, as illustrated on Figure 8.2.

**知识点讲解:**
- **Aggressive Quantization (激进量化):** 激进量化是指将模型中的数值精度大幅降低，以减少内存占用和计算成本。

An alternative to **Post-Training Quantization (训练后量化)** is **Quantization-Aware Training (量化感知训练)** that applies quantization during the forward pass but keeps high-precision encoding of parameters and gradients, and propagates the gradients during the backward pass as if there was no quantization [Ma et al., 2024].

**知识点讲解:**
- **Quantization-Aware Training (量化感知训练):** 量化感知训练是指在训练过程中模拟量化的效果，以使模型在量化后仍能保持较高的性能。

#### 8.3 Adapters

**Adapters (适配器)** are small modules added to a pre-trained model to adapt it to a specific task without modifying the original model’s parameters. As we saw in § 3.6, **fine-tuning (微调)** is a key strategy to reuse pre-trained models. Since it aims at making only minor changes to an existing model, techniques have been developed that add components with few parameters, referred to as **adapters (适配器)**, to the pre-trained architecture, and freeze all the original parameters [Houlsby et al., 2019].

**知识点讲解:**
- **Adapters (适配器):** 适配器是指在不修改原始模型参数的情况下，通过添加少量参数来适应特定任务的小模块。
- **Fine-Tuning (微调):** 微调是指在预训练模型的基础上，使用特定任务的数据进行进一步训练，以使模型更好地适应特定任务。

The current dominant method is the **Low-Rank Adaptation (LoRA, 低秩适配)**, which adds low-rank corrections to some of the model’s weight matrices [Hu et al., 2021].

**知识点讲解:**
- **Low-Rank Adaptation (LoRA, 低秩适配):** 低秩适配是一种通过添加低秩矩阵来调整模型权重的方法，以减少参数数量。

Formally, given a linear operation of the form \( XW^T \), where \( X \) is a \( N \times D \) tensor of activations for a batch of \( N \) samples, and \( W \) is a \( C \times D \) weight matrix, the LoRA adapter replaces this operation with \( X(W + BA)^T \), where \( A \) and \( B \) are two trainable matrices of size \( R \times D \) and \( C \times R \) respectively, with \( R \ll \min(C, D) \), and the matrix \( W \) is removed from the trainable parameters. The matrix \( A \) is initialized with random Gaussian values, and \( B \) is set to zero, so that the fine-tuning starts with a model that computes an output identical to that of the original one.

**知识点讲解:**
- **Low-Rank Matrix (低秩矩阵):** 低秩矩阵是指秩远小于其行数和列数的矩阵，通常用于近似表示高维数据。

The total number of parameters to optimize with this approach is generally a few percent of the number of parameters in the original model.

The standard procedure to fine-tune a transformer with such adapters is to change only the weight matrices in the attention blocks, and to keep the **MLP (多层感知器)** of the feed-forward blocks unchanged. The same strategy has been used successfully to tune **diffusion denoising models (扩散去噪模型)** by fine-tuning the attention blocks responsible for the text-based conditioning.

**知识点讲解:**
- **Attention Blocks (注意力块):** 注意力块是Transformer模型的核心组件，用于捕捉序列中的依赖关系。
- **Diffusion Denoising Models (扩散去噪模型):** 扩散去噪模型是一种生成模型，通过学习如何从噪声数据中恢复原始数据来生成新的数据。

Since fine-tuning with LoRA adapters drastically reduces the number of trainable parameters, it reduces the memory footprint required by optimizers such as **Adam**, which generally store two running averages per parameter to optimize. Also, it reduces slightly the computation during the backward pass.

**知识点讲解:**
- **Adam Optimizer (Adam优化器):** Adam优化器是一种自适应学习率优化算法，通常用于训练深度学习模型。

For commercial applications that require a large number of fine-tuned models, the \( AB \) pairs can be stored separately from the original model, which has to be stored only once. And finally, contrary to other types of adapters, the modifications can be integrated into the original architecture, simply by adding \( AB \) to \( W \), resulting in an architecture and parameter count for inference that is identical to the original one.

**知识点讲解:**
- **Commercial Applications (商业应用):** 商业应用是指将模型应用于实际业务场景中，通常需要高效的推理和部署。

We saw that quantization degrades models’ accuracy only marginally. However, gradient descent requires high precision in both the gradient and the trained parameters, to allow the accumulation of small changes. The **QLoRA (量化低秩适配)** approach combines a quantized base model and unquantized Low-Rank Adaptation to reduce the memory requirement even more [Dettmers et al., 2023].

**知识点讲解:**
- **QLoRA (量化低秩适配):** QLoRA是一种结合量化和低秩适配的方法，用于进一步减少模型的内存占用。

#### 8.4 Model Merging

**Model Merging (模型合并)** is the process of combining multiple models into a single model to leverage their combined capabilities. An alternative to the fine-tuning and prompting methods seen in the previous sections consists of combining multiple models with diverse capabilities into a single one, without additional training.

**知识点讲解:**
- **Model Merging (模型合并):** 模型合并是指将多个模型合并为一个模型，以利用它们的综合能力。

Model merging relies on the compatibility between multiple fine-tuned versions of a base model.

Ilharco et al. [2022] showed that models obtained by fine-tuning a **CLIP (Contrastive Language-Image Pre-training, 对比语言-图像预训练)** base model on several image classification datasets can be combined in the parameter space, where they exhibit **Task Arithmetic (任务算术)** properties.

**知识点讲解:**
- **Task Arithmetic (任务算术):** 任务算术是指通过将多个任务模型的参数进行算术操作来合并模型的方法。

Formally, let \( \theta \) be the parameter vector of a pre-trained model, and for \( t = 1, \ldots, T \), let \( \theta_t \) and \( \tau_t = \theta_t - \theta \) be respectively the parameters after fine-tuning on task \( t \) and the corresponding residual. Experiments show that the model with parameters \( \theta + \tau_1 + \cdots + \tau_T \) exhibits multi-task capabilities. Similarly, subtracting a \( \tau_t \) degrades the performance on the corresponding task.

**知识点讲解:**
- **Residual (残差):** 残差是指模型在微调后参数与原始参数之间的差异。

Methods have been developed to reduce the interference between the different residuals and improve the performance when the number of tasks is large [Yadav et al., 2023].

**知识点讲解:**
- **Interference (干扰):** 干扰是指多个任务模型在合并时可能产生的性能下降。

An alternative to merging models in parameter space is to recombine their layers. Akiba et al. [2024] combine merging the parameters and re-combining layers, and rely on a stochastic optimization to deal with the combinatorial explosion. Experiments with three fine-tuned versions of **Mistral-7B** [Jiang et al., 2023] show that combining these two merging strategies outperforms both of them.

**知识点讲解:**
- **Stochastic Optimization (随机优化):** 随机优化是指通过随机搜索来优化模型参数的方法。
- **Combinatorial Explosion (组合爆炸):** 组合爆炸是指随着任务数量的增加，模型合并的可能性呈指数增长。

### The Missing Bits

For the sake of concision, this volume skips many important topics, in particular:

#### Recurrent Neural Networks

Before attention models showed greater performance, **Recurrent Neural Networks (RNN, 循环神经网络)** were the standard approach for dealing with temporal sequences such as text or sound samples. These architectures possess an internal **hidden state (隐藏状态)** that gets updated each time a component of the sequence is processed. Their main components are layers such as **LSTM (Long Short-Term Memory, 长短期记忆)** [Hochreiter and Schmidhuber, 1997] or **GRU (Gated Recurrent Unit, 门控循环单元)** [Cho et al., 2014].

**知识点讲解:**
- **Recurrent Neural Networks (RNN, 循环神经网络):** 循环神经网络是一种用于处理序列数据的神经网络，具有记忆能力。
- **Hidden State (隐藏状态):** 隐藏状态是指RNN在处理序列时保留的内部状态，用于捕捉序列中的依赖关系。

Training a recurrent architecture amounts to unfolding it in time, which results in a long composition of operators. This has historically prompted the design of key techniques now used for deep architectures such as **rectifiers (整流器)** and **gating (门控)**, a form of skip connections which are modulated by the hidden state.

**知识点讲解:**
- **Rectifiers (整流器):** 整流器是指用于处理非线性激活函数的组件，如ReLU。
- **Gating (门控):** 门控是指通过控制信息的流动来增强模型的记忆能力。

One of the key drawbacks of traditional recurrent architectures is that the structure of the computation \( x_{t+1} = f(x_t) \) imposes to process the input sequence serially, which takes a time proportional to \( T \). In contrast, transformers, for instance, can take advantage of parallel computation, resulting in a constant time if enough computing units are available.

**知识点讲解:**
- **Serial Processing (串行处理):** 串行处理是指按顺序处理输入序列，通常需要较长的时间。
- **Parallel Computation (并行计算):** 并行计算是指同时处理多个输入，以提高计算效率。

This is addressed by architectures such as **QRNN (Quasi-Recurrent Neural Networks, 准循环神经网络)** [Bradbury et al., 2016], **S4 (Structured State Spaces, 结构化状态空间)** [Gu et al., 2021], or **Mamba (Mamba模型)** [Gu and Dao, 2023], whose recurrent operations are affine so that the \( f^t \) themselves, and consequently the \( x_t = f^t(x_0) \), can be computed in parallel, resulting in a constant time if \( f \) does not depend on \( t \) and \( \log T \) otherwise, again if enough parallel computing units are available.

**知识点讲解:**
- **Affine Operation (仿射操作):** 仿射操作是指线性变换加上一个偏置项的操作。
- **Parallel Computing Units (并行计算单元):** 并行计算单元是指能够同时执行多个计算任务的硬件设备。

#### Autoencoder

An **autoencoder (自编码器)** is a model that maps an input signal, possibly of high dimension, to a low-dimension **latent representation (潜在表示)**, and then maps it back to the original signal, ensuring that information has been preserved. We saw it in § 6.1 for denoising, but it can also be used to automatically discover a meaningful low-dimension representation.

**知识点讲解:**
- **Autoencoder (自编码器):** 自编码器是一种通过将输入数据映射到低维空间并重建原始数据的模型，通常用于降维和特征提取。
- **Latent Representation (潜在表示):** 潜在表示是指数据在低维空间中的表示，通常用于捕捉数据的主要特征。

The **Variational Autoencoder (VAE, 变分自编码器)** proposed by Kingma and Welling [2013] is a generative model with a similar structure. It imposes, through the loss, a pre-defined distribution on the latent representation. This allows, after training, the generation of new samples by sampling the latent representation according to this imposed distribution and then mapping back through the decoder.

**知识点讲解:**
- **Variational Autoencoder (VAE, 变分自编码器):** 变分自编码器是一种生成模型，通过学习潜在表示的分布来生成新的数据。
- **Pre-Defined Distribution (预定义分布):** 预定义分布是指在模型训练前设定的潜在表示的分布，通常为正态分布。

#### Generative Adversarial Networks

Another approach to density modeling is the **Generative Adversarial Networks (GAN, 生成对抗网络)** introduced by Goodfellow et al. [2014]. This method combines a **generator (生成器)**, which takes a random input following a fixed distribution as input and produces a structured signal such as an image, and a **discriminator (判别器)**, which takes a sample as input and predicts whether it comes from the training set or if it was generated by the generator.

**知识点讲解:**
- **Generative Adversarial Networks (GAN, 生成对抗网络):** 生成对抗网络是一种通过生成器和判别器之间的对抗来生成新数据的模型。
- **Generator (生成器):** 生成器是指通过学习如何从随机噪声中生成新数据的模型。
- **Discriminator (判别器):** 判别器是指通过学习如何区分真实数据和生成数据的模型。

Training optimizes the discriminator to minimize a standard cross-entropy loss, and the generator to maximize the discriminator’s loss. It results in a generator that produces samples that are indistinguishable from real data. In practice, when the gradient flows through the discriminator to the generator, it informs the latter about the cues that the discriminator uses that need to be addressed.

**知识点讲解:**
- **Cross-Entropy Loss (交叉熵损失):** 交叉熵损失是指用于衡量模型预测与真实标签之间差异的损失函数。
- **Gradient Flow (梯度流):** 梯度流是指梯度在模型中的传播过程，用于更新模型参数。

#### Graph Neural Networks

Many applications require processing signals which are not organized regularly on a grid. For instance, proteins, 3D meshes, geographic locations, or social interactions are more naturally structured as **graphs (图)**. Standard convolutional networks or even attention models are poorly adapted to process such data, and the tool of choice for such a task is **Graph Neural Networks (GNN, 图神经网络)** [Scarselli et al., 2009].

**知识点讲解:**
- **Graph Neural Networks (GNN, 图神经网络):** 图神经网络是一种用于处理图结构数据的神经网络，能够捕捉节点之间的关系。
- **Graph (图):** 图是指由节点和边组成的数据结构，通常用于表示复杂的关系。

These models are composed of layers that compute activations at each vertex by combining linearly the activations located at its immediate neighboring vertices. This operation is very similar to a standard convolution, except that the data structure does not reflect any geometrical information associated with the feature vectors they carry.

**知识点讲解:**
- **Vertex (顶点):** 顶点是指图中的节点，通常用于表示实体。
- **Neighboring Vertices (相邻顶点):** 相邻顶点是指与某个顶点直接相连的其他顶点。

#### Self-Supervised Learning

As stated in § 7.1, even though they are trained only to predict the next word, **Large Language Models (LLM, 大语言模型)** trained on large unlabeled datasets such as **GPT (Generative Pre-trained Transformer, 生成式预训练变压器)** (see § 5.3) are able to solve various tasks, such as identifying the grammatical role of a word, answering questions, or even translating from one language to another [Radford et al., 2019].

**知识点讲解:**
- **Self-Supervised Learning (自监督学习):** 自监督学习是指通过无标签数据来训练模型，使其能够自动学习有用的特征表示。
- **Grammatical Role (语法角色):** 语法角色是指单词在句子中的语法功能，如主语、宾语等。

Such models constitute one category of a larger class of methods that fall under the name of **self-supervised learning (自监督学习)**, and try to take advantage of unlabeled datasets [Balestriero et al., 2023].

**知识点讲解:**
- **Unlabeled Datasets (无标签数据集):** 无标签数据集是指没有标注的数据集，通常用于自监督学习。

The key principle of these methods is to define a task that does not require labels but necessitates feature representations which are useful for the real task of interest, for which a small labeled dataset exists. In computer vision, for instance, image features can be optimized so that they are invariant to data transformations that do not change the semantic content of the image, while being statistically uncorrelated [Zbontar et al., 2021].

**知识点讲解:**
- **Feature Representations (特征表示):** 特征表示是指数据在模型中的表示，通常用于捕捉数据的主要特征。
- **Invariant (不变性):** 不变性是指特征表示在数据变换下保持不变的性质。

In both NLP and computer vision, a powerful generic strategy is to train a model to recover a corrupted version of the input, for instance, by masking some of its components, or by predicting the missing parts of a sequence [Devlin et al., 2018; Zhou et al., 2021].

**知识点讲解:**
- **Corrupted Version (损坏版本):** 损坏版本是指通过添加噪声或删除部分数据来破坏原始数据的版本。
- **Masking (掩码):** 掩码是指通过隐藏部分数据来训练模型的方法。

### Bibliography

The bibliography section lists all the references cited in the book, providing the necessary citations for further reading and research.

### Index

The index section provides a quick reference to the key terms and concepts discussed in the book, allowing readers to easily locate specific topics.

---

This concludes the translation and explanation of the remaining chapters from Chapter 7 onwards. The content covers advanced topics in deep learning, including text generation, image generation, model optimization techniques like quantization and adapters, and the merging of models. Each section is accompanied by detailed explanations of the key concepts to enhance understanding.


