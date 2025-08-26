# Blossom

### Notations and Symbols

这些是开花函数和 Bézier 曲线相关的符号。

| Symbol | Description |
| :----: | :---------: |
| $n \in \mathbb{N}$ | Degree of Bernstein polynomial |
| $\mathbf{P}_{i} \in \mathbb{R}^{d}$ | $i$-th Control point |
| $a, b \in \mathbb{R}$ | Interval |
| $f(u_{1}, \dots, u_{n})$ | Blossom function of $n$ variables |

## Introduction

任意 $n$ 次多项式 $p(t)$ 都存在一个唯一对应的 $n$ 元开花函数 $f(u_{1}, \dots, u_{n})$。开花函数 $f(u_{1}, \dots, u_{n})$ 满足如下抽象的性质：

1. **对称性**：$f(u_{1}, \dots, u_{n}) = f(u_{\sigma(1)}, \dots, u_{\sigma(n)})$，其中 $\sigma$ 是 $\{1, \dots, n\}$ 的任意一个全排列。
2. **多线性**：$f(\dots, (1 - \alpha)u_{k} + \alpha v_{k}, \dots) = (1 - \alpha) f(\dots, u_{k}, \dots) + \alpha f(\dots, v_{k}, \dots)$。
3. **对角线特性**：$f(t, \dots, t) = p(t)$。

单纯从这些抽象的性质还看不出开花函数的作用。先举一个例子：

> 三次多项式的幂基函数 $\{1, t, t^{2}, t^{3}\}$ 对应的 3 元开花函数为：
> $$
> \begin{aligned}
> p_{0}(t) = 1 & \iff f_{0}(u_{1}, u_{2}, u_{3}) = 1 \\
> p_{1}(t) = t & \iff f_{1}(u_{1}, u_{2}, u_{3}) = \frac{u_{1} + u_{2} + u_{3}}{3} \\
> p_{2}(t) = t^{2} & \iff f_{2}(u_{1}, u_{2}, u_{3}) = \frac{u_{1}u_{2} + u_{1}u_{3} + u_{2}u_{3}}{3} \\
> p_{3}(t) = t^{3} & \iff f_{3}(u_{1}, u_{2}, u_{3}) = u_{1}u_{2}u_{3}
> \end{aligned}
> $$
> 因此任意三次多项式 $p(t) = a_{3} t^{3} + a_{2} t^{2} + a_{1} t + a_{0}$ 的 3 元开花函数的形式为：
> $$
> f(u_{1}, u_{2}, u_{3}) = a_{3} (u_{1}u_{2}u_{3}) + a_{2} \frac{u_{1}u_{2} + u_{1}u_{3} + u_{2}u_{3}}{3} + a_{1} \frac{u_{1} + u_{2} + u_{3}}{3} + a_{0}
> $$

由此可以推广，$n$ 次多项式的幂基函数 $p_k(t) = t^{k}$ 对应的 $n$ 元开花函数为：
$$
p_{k}(t) = t^{k} \iff f(u_{1}, \dots, u_{n}) = \frac{\displaystyle{\sum_{1 \le i_1 < \dots < i_k \le n} u_{i_{1}} \cdots u_{i_{k}}}}{\displaystyle{\binom{n}{k}}}
$$
其中，分子是对参数 $\{u_{1}, \dots, u_{n}\}$ 中所有可能的 **k 元组合** 进行求和，是 k 次基本对称多项式。

这个唯一形式是由“开花”的三个性质共同决定的：
1. **对称性**与**多线性**要求函数的结构必须是基本对称多项式的线性组合。对于 $t^k$ 这种单项，其“开花”形式必然为 $c \cdot (\sum u_{i_{1}} \cdots u_{i_{k}})$，其中 $c$ 为常数。
2. **对角线特性**决定了归一化常数 $c$。当把所有参数都设为 $t$ 时，分子共有 $\binom{n}{k}$ 项，每一项都等于 $t^k$，所以和为 $\binom{n}{k} t^k$。为了满足 $f(t, \dots, t) = t^k$，常数 $c$ 必须是 $1 / \binom{n}{k}$。

由于任意 $n$ 次多项式都可以表示为幂基函数的线性组合，且“开花”也满足线性关系，因此任意 $n$ 次多项式 $p(t)$ 都存在唯一的 $n$ 元开花函数 $f(u_{1}, \dots, u_{n})$。同理，以 Bernstein 基函数（其本身也是多项式）为基的多项式，自然也有其对应的、唯一的开花函数。

## De Casteljau Algorithm

De Casteljau 算法是计算开花函数值的一种高效的递归方法。设参数 $t$ 位于区间 $[a, b]$ 内，则 $t$ 可以表示为 $a$ 和 $b$ 的仿射组合：
$$
t = \frac{b - t}{b - a} \cdot a + \frac{t - a}{b - a} \cdot b
$$
利用开花函数的多线性，我们可以对其中一个参数 $t$ 进行分解：
$$
\begin{aligned}
f(t, \dots, t) &= f\left(\frac{b - t}{b - a} \cdot a + \frac{t - a}{b - a} \cdot b, t, \dots, t\right) \\
&= \frac{b - t}{b - a} \cdot f(a, t, \dots, t) + \frac{t - a}{b - a} \cdot f(b, t, \dots, t)
\end{aligned}
$$
这个过程可以递归地应用于每一个参数 $t$，直到所有参数都变成 $a$ 或 $b$ 为止。最终，我们只需要计算那些只包含 $a$ 和 $b$ 的开花函数值。根据对称性，这些初始值只有 $n + 1$ 种不同形式：$f(\underbrace{a, \dots, a}_{n - k}, \underbrace{b, \dots, b}_{k})$，其中 $k \in \{0, \dots, n\}$。

这个递归过程可以用一个三角形图示来表达：
$$
\begin{matrix}
f(a, \dots, a, a) \\
 & \searrow \\
 & & f(a, \dots, a, t) \\
 & \nearrow & & \searrow \\
f(a, \dots, a, b) & & & & f(a, \dots, t, t) \\
 & \searrow & & \nearrow & & \searrow \\
 & & f(a, \dots, b, t) & & \vdots & & f(t, \dots, t, t) \\
 & \nearrow & & \searrow & & \nearrow \\
\vdots & & \vdots & & f(b, \dots, t, t) \\
 & \searrow & & \nearrow \\
 & & f(b, \dots, b, t) \\
 & \nearrow \\
f(b, \dots, b, b)
\end{matrix}
$$
其中，从左上到右下的箭头 $\searrow$ 表示乘以权重 $\dfrac{b - t}{b - a}$，从左下到右上的箭头 $\nearrow$ 表示乘以权重 $\dfrac{t - a}{b - a}$。整个计算从左侧的初始值开始，逐列向右推进，最终得到最右侧顶点的 $f(t, \dots, t)$。

这与我们熟悉的 Bézier 曲线的计算过程完全相同。这种等价性并非巧合，因为 Bézier 曲线的控制点正是其开花函数在区间端点上的特定求值。

## Blossom of Bézier Curves

设 $\mathbf{C}(t; \{\mathbf{P}_{i}\}_{i = 0}^{n})$ 是一个定义在 $[0, 1]$ 上 $n$ 次 Bézier 曲线，其控制点为 $\{\mathbf{P}_{i}\}_{i = 0}^{n}$。$\mathbf{C}(t; \{\mathbf{P}_{i}\}_{i = 0}^{n})$ 可以按维度分解为 $d$ 个 $n$ 次多项式，每个多项式都对应了一个唯一的标量开花函数。这些开花函数的向量组合，就是 Bézier 曲线的**向量值开花函数**，记为 $\mathbf{f}(u_{1}, \dots, u_{n}; \{\mathbf{P}_{i}\}_{i = 0}^{n})$。很显然，它也是唯一的。

开花理论最美妙的结论之一，就是它揭示了控制点的几何意义：**控制点 $\mathbf{P}_k$ 就是其开花函数在区间端点 0 和 1 上的求值结果。**
$$
\mathbf{P}_k = \mathbf{f}(\underbrace{0, \dots, 0}_{n-k}, \underbrace{1, \dots, 1}_{k})
$$

### Subdivision of a Bézier Curve

将一条 Bézier 曲线 $\mathbf{C}$ 拆分为两条短的 Bézier 曲线 $\mathbf{C}_{1}$ 和 $\mathbf{C}_{2}$ 之后，曲线 $\mathbf{C}_{1}$ 和 $\mathbf{C}_{2}$ 的控制点由 $\mathbf{C}$ 的控制点和拆分点决定。具体来说，设在 $\mathbf{C}(u), u \in [0, 1]$ 处拆分，则 $\mathbf{C}_{1}$ 对应原区间 $[0, u]$； $\mathbf{C}_{2}$ 对应原区间 $[u, 1]$。它们各自的控制点可以直接通过原始曲线的开花函数 $\mathbf{f}$ 计算得出。


- 曲线 $\mathbf{C}_{1}$ 的控制点为 $\{\mathbf{Q}_{i} = \mathbf{f}(\underbrace{0, \dots, 0}_{n - i}, \underbrace{u, \dots, u}_{i}; \{\mathbf{P}_{j}\}_{j = 0}^{n})\}_{i = 0}^n$
- 曲线 $\mathbf{C}_{2}$ 的控制点为 $\{\mathbf{R}_{i} = \mathbf{f}(\underbrace{u, \dots, u}_{n - i}, \underbrace{1, \dots, 1}_{i}; \{\mathbf{P}_{j}\}_{j = 0}^{n})\}_{i = 0}^n$
