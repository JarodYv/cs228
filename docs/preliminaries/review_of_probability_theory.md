# 概率论回顾

首先让我们回顾一下概率论的一些概念，这部分所有的材料都改编自[CS229 - 概率论课程笔记](http://cs229.stanford.edu/section/cs229-prob.pdf)。

## 1. 概率基础

为了定义集合上的概率，我们需要一些基本元素。

**样本空间 $ \Omega $**: 随机实验所有结果的集合。其中，每个结果 $\omega \in \Omega$ 都可以被认为是实验结束时真实世界状态的完整描述。

**事件集（或事件空间）$ F $**: 元素 $A \in F$（称之为事件）是 $\Omega$ 的子集的集合。（例如，$A \subseteq \Omega$ 是实验可能结果的集合）。

**概率测度**: 一个满足如下性质的函数 $P : F \to \Reals$

* 对于任意 $A \in F$, $P(A) \geq 0$
* 如果 $A_1, A_2, \dotsc$ 是不相交事件（即 $A_i \cap A_j = \emptyset$，其中 $i \neq j$），那么 $P(\bigcup_i A_i) = \sum_i P(A_i)$
* $P(\Omega) = 1$

这三个性质被称为概率公理。

**例题：** 考虑抛六面骰子的事件。样本空间为 $\Omega = \{1, 2, 3, 4, 5, 6\}$。 我们可以在此样本空间上定义不同的事件空间。
例如，最简单的事件空间是平凡事件空间 $F = \{\emptyset, \Omega\}$。另一个事件空间是 $ \Omega $ 的所有子集的集合。
对于第一个事件空间，满足上述要求的唯一概率测度为 $P(\emptyset) = 0, P(\Omega) = 1$
对于第二个事件空间，一个有效的概率度量是将事件空间中每个集合的概率分配为 $\frac{i}{6}$，其中 $i$ 是该集合的元素数；
例如：$P(\{1, 2, 3, 4\}) = \frac{4}{6}$ 和 $P(\{1, 2, 3\}) = \frac{3}{6}$。

#### **性质**

- $A \subseteq B \implies P(A) \leq P(B)$
- $P(A \cap B) \leq \min(P(A), P(B))$
- **联合约束:** $P(A \cup B) \leq P(A) + P(B)$
- $P(\Omega - A) = 1 - P(A)$
- **全概率法则:** 设 $A_1, \dotsc, A_k$ 是一组不相交的事件，如果 $\bigcup^k_{i=1} A_i = \Omega$, 那么 $\sum^k_{i=1} P(A_i) = 1$

### 1.1 条件概率

设 $B$ 是非零概率的事件。给定 $B$ 的情况下任意事件 $A$ 的条件概率为：

$$ P(A \mid B) = \frac {P(A \cap B)}{P(B)} $$

换句话说，$P(A \mid B)$ 是观察到事件 $B$ 发生后事件 $A$ 发生的概率度量。

### 1.2 链式法则

设 $S_1, \dotsc, S_k$ 为事件，$P(S_i) >0$，则链式法则描述如下：

$$
\begin{align*}
& P(S_1 \cap S_2 \cap \dotsb \cap S_k) \\
&= P(S_1) P(S_2 | S_1) P(S_3 | S_2 \cap S_1 ) \dotsb P(S_k | S_1 \cap S_2 \cap \dotsb \cap S_{k-1})
\end{align*}
$$

当 $k=2$ 时，上面的公式就是条件概率公式：

$$ P(S_1 \cap S_2) = P(S_1) P(S_2 | S_1) $$

一般来说，链式规则是通过多次应用条件概率的定义而导出的，例如：

$$
\begin{align*}
& P(S_1 \cap S_2 \cap S_3 \cap S_4) \\
&= P(S_1 \cap S_2 \cap S_3) P(S_4 \mid S_1 \cap S_2 \cap S_3) \\
&= P(S_1 \cap S_2) P(S_3 \mid S_1 \cap S_2) P(S_4 \mid S_1 \cap S_2 \cap S_3) \\
&= P(S_1) P(S_2 \mid S_1) P(S_3 \mid S_1 \cap S_2) P(S_4 \mid S_1 \cap S_2 \cap S_3)
\end{align*}
$$

### 1.3 独立性

如果 $P(A \cap B) = P(A)P(B)$，则称这两个事件独立，或等价于 $P(A \mid B) = P(A)$。直观感觉，如果事件 $A$ 和 $B$ 独立，
意味着对 $B$ 的观测不影响 $A$ 的概率。

## 2. 随机变量

考虑如下实验：我们翻转10个硬币，我们想知道出现正面的硬币数量。这里，样本空间 $\Omega$ 的元素是长度为10由正面和反面组成的序列。
例如 $\omega_0 = \langle H, H, T, H, T, H, H, T, T, T \rangle \in \Omega$。
然而在实践中，我们通常不关心获得任何特定序列的正面和反面的概率。相反，我们通常关心结果的实函数，例如10次抛投中出现的正面次数，或连续出现反面的次数。
在某些技术场景下，这些函数被称为 **随机变量**。

随机变量更严格的定义是：随机变量 $X$ 是一个函数 $X : \Omega \to E$，其中 $E$ 是某可度量空间。
通常，我们使用大写字母 $X(\omega)$ 表示随机变量，或简写为 $X$(其中隐含了对随机结果 $\omega$ 的依赖)。
我们使用小写字母 $x$ 表示随机变量的值。因此 $X = x$ 表示我们将值 $x \in E$ 复制给随机变量 $X$。

**例题：** 在我们上面的实验中，假设 $X(\omega)$ 是一系列抛投 $\omega$ 中正面出现的次数。
假定只抛投10个硬币，$X(\omega)$ 只能取有限个值，因此称为离散随机变量。 
此时，与随机变量 $X$ 关联的集合取某个特定值 $k$ 的概率为 $P(X = k) := P(\{\omega : X(\omega) = k\})$

**例题：** 假设 $X(\omega)$ 是表示放射性粒子衰变所需的时间的随机变量。在本例中，$X(\omega)$ 有无穷多个可能值，因此称其为连续随机变量。
我们记 $X$ 取两个实数 $a$ 和 $b$ ($a \lt b$)之间的概率为 $P(a \leq X \leq b) := P(\{\omega : a \leq X(\omega) \leq b\})$

当描述随机变量具有特定值的事件时，我们通常用**指示函数** $\mathbf{1}\{A\}$，其中当事件 $A$ 发生时取值为1，反之取值为0。例如，对随机变量 $X$

$$
    \mathbf{1}\{X > 3\} = \begin{cases}
    1, & \text{if }X > 3 \\
    0, & \text{otherwise}
    \end{cases}
$$

### 2.1 累积分布函数

为了指定处理随机变量时使用的概率度量，通常会指定替代函数（CDF、PDF和PMF），从这些函数中可以立即得出控制实验的概率度量。
本节和接下来的两节，我们将依次介绍这些类型的函数。**累积分布函数**（CDF）是一个函数 $F_X : \Reals \to [0, 1]$，它指定了如下概率度量：

$$ F_X(x) = P(X \leq x) $$

使用该函数，可以计算 $X$ 取任意两个实数 $a$ 和 $b$ ($a \lt b$) 之间的值的概率。

#### **性质**

- $0 \leq F_X(x) \leq 1$
- $\lim_{x \to -\infty} F_X(x) = 0$
- $\lim_{x \to +\infty} F_X(x) = 1$
- $x \leq y \implies F_X(x) \leq F_X(y)$

### 2.2 概率质量函数

当随机变量 $X$ 取有限组可能值时（即 $X$ 是离散随机变量），表示随机变量的概率度量的更简单方法是直接指定随机变量可能取到的每个值的概率。
特别地，概率质量函数（PMF）是一个函数 $p_X : \Reals \to [0, 1]$ 因此 $p_X(x) = P(X = x)$

在离散随机变量的情况下，我们使用符号 $Val(X)$ 表示随机变量 $X$ 的一组可能值。
例如：如果 $X(\omega)$ 是一个随机变量，表示10次硬币投掷中的正面数，那么 $Val(X) = \{0, 1, 2, \dotsc, 10\}$

#### **性质**

- $0 \leq p_X(x) \leq 1$
- $\sum_{x \in Val(X)} p_X(x) = 1$
- $\sum_{x \in A} p_X(x) = P(X \in A)$

### 2.3 概率密度函数

对于某些连续随机变量，累积分布函数 $F_X(x)$ 处处可微。在种情况下，我们将概率密度函数或PDF定义为CDF的导数，例如：

$$ f_X(x) = \frac{dF_X(x)}{dx} $$

注意，连续随机变量的PDF可能并不总是存在的。（例如 $F_X(x)$ 不是处处可微的）。

根据微分的性质，对于任意无穷小 $\delta x$，

$$ P(x \leq X \leq x + \delta x) \approx f_X(x) \delta x$$

CDF和PDF（如果存在）都可用于计算不同事件的概率。但需要强调的是，任意给定点 $x$ 处的PDF值不是该事件的概率，比如 $f_X(x) \neq P(X = x)$。
举个例子，$f_X(x)$ 可以去大于1的值，但 $f_X(x)$ 在 $\Reals$ 的任意子集上的积分最多为1。

#### **性质**

- $f_X(x) \geq 0$
- $\int^{\infty}_{-\infty} f_X(x) dx = 1$
- $\int_{x \in A} f_X(x) dx = P(X \in A)$

### 2.4 期望

假设 $X$ 是一个离散随机变量，具有PMF $p_X(x)$， $g : \Reals \to \Reals$ 是任意函数。这里 $g(X)$ 可以看作一个随机变量，
我们定义 $g(X)$ 的 **期望** 或 **期望值** 为

$$ \mathbb{E}[g(X)] = \sum_{x \in Val(X)} g(x)p_X(x)$$

如果 $X$ 是一个连续随机变量，具有PDF $f_X(x)$，则 $g(X)$ 的期望值定义为

$$ \mathbb{E}[g(X)] = \int^{\infty}_{-\infty} g(x)f_X(x)dx$$

直觉上，$g(X)$ 的期望可被视为 $x$ 取不同值时$g(x)$值的“加权平均值”，其中权重由 $p_X(x)$ 或 $f_X(x)$给出。
作为上述内容的一种特殊情况，注意，随机变量本身的期望值 $\mathbb{E}[X]$ 是通过 $g(x) = x$ 求出的。这也被称为随机变量 $X$ 的平均值。

#### **性质**

- $\mathbb{E}[a] = a$ 对任意常数 $a \in \Reals$
- $\mathbb{E}[af(X)] = a\mathbb{E}[f(X)]$ 对任意常数 $a \in \Reals$
- (线性期望) $\mathbb{E}[f(X) + g(X)] = \mathbb{E}[f(X)] + \mathbb{E}[g(X)]$
- 对于离散随机变量 $X$, $\mathbb{E}[\mathbf{1}\{X = k\}] = P(X = k)$

### 2.5 方差

随机变量 $X$ 的方差是随机变量 $X$ 在其均值附近的集中程度的度量。形式上，随机变量 $X$ 的方差定义为 $Var[X] = \mathbb{E}[(X - \mathbb{E}[X])^2]$

使用前面章节给出的性质，我们可以推到出方差的另一种表达形式：

$$
\begin{align*}
& \mathbb{E}[(X - \mathbb{E}[X])^2] \\
&= \mathbb{E}[X^2 - 2\mathbb{E}[X]X + \mathbb{E}[X]^2] \\
&= \mathbb{E}[X^2] - 2\mathbb{E}[X]\mathbb{E}[X] + \mathbb{E}[X]^2 \\
&= \mathbb{E}[X^2] - \mathbb{E}[X]^2
\end{align*}
$$

其中第二等式来自于线性期望，且事实上，相对于外部期望，$\mathbb{E}[X]$实际上是一常数。

#### **性质**

- $Var[a] = 0$ 对任意常数 $a \in \Reals$
- $Var[af(X)] = a^2 Var[f(X)]$ 对任意常数 $a \in \Reals$

**例题：** 计算均匀随机变量 $X$ 的均值和方差，$X$ 的 PDF为 $f_X(x) = 1, \forall x \in [0, 1]$，否则为0。

$$
\begin{align*}
\mathbb{E}[X] &= \int^{\infty}_{-\infty} x f_X(x) dx = \int^1_0 x dx = \frac{1}{2} \\
\mathbb{E}[X^2] &= \int^{\infty}_{-\infty} x^2 f_X(x)dx = \int^1_0 x^2 dx = \frac{1}{3} \\
Var[X] &= \mathbb{E}[X^2] - \mathbb{E}[X]^2 = \frac{1}{3} - \frac{1}{4} = \frac{1}{12}
\end{align*}
$$

**例题：** 设 $g(x) = \mathbf{1}\{x \in A\}$，其中 $A \subseteq \Omega$，求 $\mathbb{E}[g(X)]$

- 离散情况

$$
\mathbb{E}[g(X)] = \sum_{x \in Val(X)} \mathbf{1}\{x \in A \} P_X(x) = \sum_{x \in A} P_X(x) = P(X \in A)
$$

- 连续情况

$$
\mathbb{E}[g(X)] = \int_{-\infty}^\infty \mathbf{1}\{x \in A \} f_X(x) dx = \int_{x\in A} f_X(x) dx = P(X \in A)
$$

### 2.6 一些常见的随机变量

#### 离散随机变量

- **$X \sim \text{Bernoulli}(p)$** (其中 $0 \leq p \leq 1$): 掷硬币的结果（$H=1, T=0$）钟出现正面的概率为 $p$。

$$
p(x) = \begin{cases}
    p, & \text{if }x = 1 \\
    1-p, & \text{if }x = 0
\end{cases}
$$

- **$X \sim \text{Binomial}(n, p)$** (其中 $0 \leq p \leq 1$):

$$ p(x) = \binom{n}{x} \cdot p^x (1-p)^{n-x} $$

- **$X \sim \text{Geometric}(p)$** (其中 $p > 0$):

$$ p(x) = p(1 - p)^{x-1} $$

- **$X \sim \text{Poisson}(\lambda)$** (其中 $\lambda$ > 0): 用于模拟罕见事件频率的非负整数的概率分布。

$$ p(x) = e^{-\lambda} \frac{\lambda^x}{x!} $$

#### 连续随机变量

- **$X \sim \text{Uniform}(a, b)$** (其中 $a < b$): 实数轴上 $a$ 和 $b$ 之间的每个值的概率密度相等。

$$
f(x) = \begin{cases}
    \frac{1}{b-a}, & \text{if }a \leq x \leq b \\
    0, & \text{otherwise}
\end{cases}
$$

- **$X \sim \text{Exponential}(\lambda)$** (其中 $\lambda > 0$): 非负实数上的衰减概率密度。

$$
f(x) = \begin{cases}
    \lambda e^{-\lambda x}, & \text{if }x \geq 0 \\
    0, & \text{otherwise}
\end{cases}
$$

- **$X \sim \text{Normal}(\mu, \sigma^2)$**: 也叫高斯分布

$$ f(x) = \frac{1}{\sqrt{2\pi}\sigma}e^{-\frac{(x-\mu)^2}{2\sigma^2}} $$

## 3. 双随机变量

到目前为止，我们只考察了单个随机变量。然而，许多情况下，在随机实验中，我们感兴趣的随机变量可能不止一个。
例如，在掷硬币10次的实验中，我们可能即关注 $X(\omega) = $出现正面的次数也关注$Y(\omega) = $连续出现正面的次数。
本章我们就来考察两个随机变量的情况。

### 3.1 联合分布和边缘分布

设我们有2个随机变量 $X$ 和 $Y$。处理这2个随机变量的一种方法是分别考虑他们。如果是这样我们只需要 $F_X(x)$ 和 $F_Y(y)$。
但是如果我们想知道同时给 $X$ 和 $Y$ 赋值随机实验结果是什么，我们就需要一个更复杂的结构，称为 $X$ 和 $Y$ 的联合累积分布函数。定义如下：

$$ F_XY(x, y) = P(X \le x, Y \le y) $$

上面公式表明，如果知道联合累积分布函数，可以计算与 $X$ 和 $Y$ 相关的任何事件的概率。

联合累计分布函数CDF $F_XY(x, y)$ 与分开考虑的累计分布函数 $F_X(x)$ 和 $F_Y(y)$ 存在如下关系：

$$
\begin{align*}
F_X(x) &= \lim_{y \to \infty} F_{XY} (x, y) \\
F_Y(y) &= \lim_{x \to \infty} F_{XY} (x, y)
\end{align*}
$$

这里，我们称 $F_X(x)$ 和 $F_Y(y)$ 为 $F_XY(x, y)$ 的**边缘累计概率分布**。

#### **性质**

- $0 \leq F_{XY} (x, y) \leq 1$
- $\lim_{x,y\to \infty} F_{XY} (x, y) = 1$
- $\lim_{x,y\to -\infty} F_{XY} (x, y) = 0$
- $F_X(x) = \lim_{y \to \infty} F_{XY} (x, y)$

### 3.2 联合概率质量函数和边缘概率质量函数

如果 $X$ 和 $Y$ 是离散随机变量，则联合概率质量函数 $p_{XY} : Val(X) \times Val(Y) \to [0, 1]$ 定义如下：

$$ p_{XY}(x, y) = P(X = x, Y = y) $$

其中，$0 \leq P_{XY}(x, y) \leq 1$$ for all $$x, y,$$ and $$\sum_{x \in Val(X)} \sum_{y \in Val(Y)} P_{XY}(x, y) = 1$。

两个变量的联合概率质量函数PMF如何分别与每个变量的概率质量函数相关？可以证明：

$$ p_X(x) = \sum_y p_{XY} (x, y). $$

p_Y(y)也是如此。这种情况下我们称 $p_X(x)$ 为随机变量 $X$ 的**边缘概率密度函数**。
在统计学中，通过对另一个变量求和来得到一个变量的边缘分布的过程通常被称为“边缘化”。

### 3.3 联合概率密度函数和边缘概率密度函数

设两个连续随机变量 $X$ 和 $Y$ 存在联合分布函数 $F_{XY}$。在 $F_{XY}(x, y)$ 在 $x$ 和 $y$ 上处处可微的情况下，
那么我们可以定义联合概率密度函数：

$$ f_{XY}(x, y) = \frac{\partial^2F_{XY}(x, y)}{\partial x \partial y} $$

跟以为情况一样，$f_{XY} (x, y) \neq P(X = x, Y = y)$，但

$$ \int \int_{(x,y) \in A} f_{XY} (x, y) dx dy = P((X, Y) \in A)$$

注意，概率密度函数的值总是非负的，但它们可能大于1。尽管如此，必须满足 $\int^{\infty}_{-\infty} \int^{\infty}_{-\infty} f_{XY}(x,y) = 1$

与离散情况类似，我们定义

$$ f_X(x) = \int^{\infty}_{-\infty} f_{XY} (x, y)dy $$

为随机变量 $X$ 的**边缘概率密度函数**（或边缘密度），$f_Y(y)$也是如此。

### 3.4 条件概率分布

### 3.5 链式法则

### 3.6 贝叶斯法则

### 3.7 独立性

### 3.8 期望和协方差

