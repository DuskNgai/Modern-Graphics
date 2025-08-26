# Bézier Curve

### Notations and Symbols

这些是与 Bernstein 多项式和 Bézier 曲线相关的符号。

| Symbol | Description |
| :----: | :---------: |
| $n \in \mathbb{N}$ | Degree of Bernstein polynomial |
| $B_{i}^{n}: [0, 1] \mapsto \mathbb{R}$ | $i$-th Bernstein polynomial of degree $n$ |
| $\mathbf{P}_{i} \in \mathbb{R}^{d}$ | $i$-th Control point |

## Resources

1. [Wikipedia](https://en.wikipedia.org/wiki/Bernstein_polynomial)
2. [The NURBS Book](https://link.springer.com/book/10.1007/978-3-642-59223-2)

## Bernstein Polynomial

定义 Bernstein 多项式 $B_{i}^{n}: [0, 1] \mapsto \mathbb{R}$ 为：
$$
B_{i}^{n}(t) = \binom{n}{i} t^{i} (1 - t)^{n - i}
$$
其中 $i \in \{0, \dots, n\}$。

### Properties

#### Non-negativity

$$
B_{i}^{n}(t) \geq 0
$$

#### Symmetry Property

$$
B_{i}^{n}(t) = B_{n - i}^{n}(1 - t)
$$

#### Partition of Unity

$$
\sum_{i = 0}^{n} B_{i}^{n}(t) = 1
$$

#### Recurrence Relation

$$
B_{i}^{n}(t) = (1 - t)B_{i}^{n - 1}(t) + tB_{i - 1}^{n - 1}(t)
$$

> $$
> \begin{aligned}
> B_{i}^{n}(t) &= \binom{n}{i} t^{i} (1 - t)^{n - i} \\
> &= \left[\binom{n - 1}{i} + \binom{n - 1}{i - 1}\right] t^{i} (1 - t)^{n - i} \\
> &= \binom{n - 1}{i} t^{i} (1 - t)^{n - i} + \binom{n - 1}{i - 1} t^{i - 1} (1 - t)^{n - i} \\
> &= (1 - t)B_{i}^{n - 1}(t) + tB_{i - 1}^{n - 1}(t)
> \end{aligned}
> $$

#### First Order Derivatives

$$
\frac{\mathrm{d}}{\mathrm{d}t} B_{i}^{n}(t) = n \left[B_{i - 1}^{n - 1}(t) - B_{i}^{n - 1}(t)\right]
$$

> $$
> \begin{aligned}
> \frac{\mathrm{d}}{\mathrm{d}t} B_{i}^{n}(t) &= \frac{\mathrm{d}}{\mathrm{d}t} \left[\binom{n}{i} t^{i} (1 - t)^{n - i}\right] \\
> &= i \binom{n}{i} t^{i - 1} (1 - t)^{n - i} - (n - i) \binom{n}{i} t^{i} (1 - t)^{n - i - 1} \\
> &= n \left[B_{i - 1}^{n - 1}(t) - B_{i}^{n - 1}(t)\right]
> \end{aligned}
> $$

#### Second Order Derivatives

$$
\frac{\mathrm{d}^{2}}{\mathrm{d}t^{2}} B_{i}^{n}(t) = n(n - 1) \left[B_{i - 2}^{n - 2}(t) - 2 B_{i - 1}^{n - 2}(t) + B_{i}^{n - 2}(t)\right]
$$

> $$
> \begin{aligned}
> \frac{\mathrm{d}^{2}}{\mathrm{d}t^{2}} B_{i}^{n}(t) &= \frac{\mathrm{d}}{\mathrm{d}t} n \left[B_{i - 1}^{n - 1}(t) - B_{i}^{n - 1}(t)\right] \\
> &= n \left[\frac{\mathrm{d}}{\mathrm{d}t} B_{i - 1}^{n - 1}(t) - \frac{\mathrm{d}}{\mathrm{d}t} B_{i}^{n - 1}(t)\right] \\
> &= n(n - 1) \left[B_{i - 2}^{n - 2}(t) - 2 B_{i - 1}^{n - 2}(t) + B_{i}^{n - 2}(t)\right]
> \end{aligned}
> $$

#### Maximum Value Points

先求得一阶导数为零的点：
$$
\frac{\mathrm{d}}{\mathrm{d}t} B_{i}^{n}(t) = 0 \implies i (1 - t) = (n - i)t \implies t = \frac{i}{n}
$$
容易验证在该点处二阶导数小于零，因此该点为极大值点。

#### Basis

$\{B_{i}^{n}: i \in \{0, \dots, n\}\}$ 是 $n$ 次多项式空间的基函数。

> 我们首先证明 $B_{0}^{N}, \dots, B_{n}^{n}$ 是一个线性无关集合。假设
> $$
> \sum_{i = 0}^{n} a_{i}B_{i}^{n}(x) = 0
> $$
> 其中 $a_{0}, a_{1}, \dots, a_{m} \in \mathbb{F}$。由于 $B_{0}^{n}(x) \propto (1 - x)^{m}$ 是集合中唯一有常数项的多项式，所以 $a_{0}$ 一定为零。对后面的多项式重复这一论证，我们会发现 $a_{1} = a_{2} = \dots = a_{m} = 0$。因此，它是一个线性无关集合。此外，这个集合的长度为 $n + 1$，因此满足作为 $n$ 次多项式空间的基的条件。

## Bézier Curve

Bézier 曲线是一个多项式曲线，定义为：
$$
\mathbf{C}\left(t; \{\mathbf{P}_{i}\}_{i = 0}^{n}\right) = \sum_{i = 0}^{n} B_{i}^{n}(t) \mathbf{P}_{i}, \quad t \in [0, 1],
$$

### Properties

很多 Bézier 曲线的性质都可以从 Bernstein 多项式的性质中推导出来。这里额外介绍一些 Bézier 曲线的性质。

#### De Casteljau

Bézier 曲线可以通过 De Casteljau 算法计算出来。
$$
\mathbf{C}\left(t; \{\mathbf{P}_{i}\}_{i = 0}^{n}\right) = (1 - t)\mathbf{C}\left(t; \{\mathbf{P}_{i}\}_{i = 0}^{n - 1}\right) + t\mathbf{C}\left(t; \{\mathbf{P}_{i}\}_{i = 1}^{n}\right)
$$

> $$
> \begin{aligned}
> \mathbf{C}\left(t; \{\mathbf{P}_{i}\}_{i = 0}^{n}\right) &= \sum_{i = 0}^{n} B_{i}^{n}(t) \mathbf{P}_{i} \\
> &= \sum_{i = 0}^{n} \left[(1 - t)B_{i}^{n - 1}(t) + tB_{i - 1}^{n - 1}(t)\right] \mathbf{P}_{i} \\
> &= (1 - t) \sum_{i = 0}^{\color{red} n - 1} B_{i}^{n - 1}(t) \mathbf{P}_{i} + t \sum_{\color{red} i = 1}^{n} B_{i - 1}^{n - 1}(t) \mathbf{P}_{i} \\
> &= (1 - t)\mathbf{C}\left(t; \{\mathbf{P}_{i}\}_{i = 0}^{n - 1}\right) + t\mathbf{C}\left(t; \{\mathbf{P}_{i}\}_{i = 1}^{n}\right)
> \end{aligned}
> $$

$$
\begin{matrix}
\mathbf{C}\left(t; \{\mathbf{P}_{0}\}\right) \\
 & \searrow \\
 & & \mathbf{C}\left(t; \{\mathbf{P}_{0}, \mathbf{P}_{1}\}\right) \\
 & \nearrow & & \searrow \\
\mathbf{C}\left(t; \{\mathbf{P}_{1}\}\right) & & & & \mathbf{C}\left(t; \{\mathbf{P}_{0}, \mathbf{P}_{1}, \mathbf{P}_{2}\}\right) \\
 & \searrow & & \nearrow & & \searrow \\
 & & \mathbf{C}\left(t; \{\mathbf{P}_{1}, \mathbf{P}_{2}\}\right) & & \vdots & & \mathbf{C}\left(t; \{\mathbf{P}_{i}\}_{i = 0}^{n}\right) \\
 & \nearrow & & \searrow & & \nearrow \\
\vdots & & \vdots & & \mathbf{C}\left(t; \{\mathbf{P}_{n - 2}, \mathbf{P}_{n - 1}, \mathbf{P}_{n}\}\right) \\
 & \searrow & & \nearrow \\
 & & \mathbf{C}\left(t; \{\mathbf{P}_{n - 1}, \mathbf{P}_{n}\}\right) \\
 & \nearrow \\
\mathbf{C}\left(t; \{\mathbf{P}_{n}\}\right)
\end{matrix}
$$
其中 $\searrow$ 表示贡献为 $(1 - t)$，$\nearrow$ 表示贡献为 $t$。

De Casteljau 算法是从上图的右侧到左侧进行计算的。形象的来说，它会分别递归计算两个节点，然后再线性组合起来。如果不存储中间结果，那这种递归的做法是低效的。显然也可以从左侧到右侧进行计算，首先计算相邻两个节点的线性组合，然后一路往上，直到只剩下一个节点。这种做法是高效的。

#### Variation Diminishing Property

在空间中，任意平面穿过控制点折线的次数比穿过 Bézier 曲线的次数多。

#### Convex Hull Property

Bézier 曲线的所有点都在控制多边形的凸包内。

#### Shift of Control Point

给 Bézier 曲线的某个控制点 $\mathbf{P}_{j}$ 加上一个向量 $\mathbf{v}$ 会导致曲线的形状发生改变：

> $$
> \begin{aligned}
> \mathbf{C}'(t) &= \sum_{i = 0}^{n} B_{i}^{n}(t) \mathbf{P}_{i} + B_{j}^{n}(t) \mathbf{v} \\
> &= \mathbf{C}(t) + B_{j}^{n}(t) \mathbf{v}
> \end{aligned}
> $$

#### Splitting Property

将一条 Bézier 曲线 $\mathbf{C}$ 拆分为两条短的 Bézier 曲线 $\mathbf{C}_{1}$ 和 $\mathbf{C}_{2}$ 之后，曲线 $\mathbf{C}_{1}$ 和 $\mathbf{C}_{2}$ 的控制点由 $\mathbf{C}$ 的控制点和拆分点决定。具体来说，设在 $\mathbf{C}(u), u \in [0, 1]$ 处拆分，则
- 曲线 $\mathbf{C}_{1}$ 的控制点为 $\{\mathbf{Q}_{i} = \sum_{j = 0}^{i} B_{j}^{i}(u) \mathbf{P}_{j}\}_{i = 0}^{n}$
- 曲线 $\mathbf{C}_{2}$ 的控制点为 $\{\mathbf{R}_{i} = \sum_{j = i}^{n} B_{j - i}^{n - i}(u) \mathbf{P}_{j}\}_{i = 0}^{n}$

> 这里证明曲线 $\mathbf{C}_{1}$ 的表达式，对于曲线 $\mathbf{C}_{2}$ 的表达式类似。
> $$
> \begin{aligned}
> \mathbf{C}_{1}(t) &= \sum_{i = 0}^{n} B_{i}^{n}(t) \left[\sum_{j = 0}^{i} B_{j}^{i}(u) \mathbf{P}_{j} \right] \\
> &= \sum_{j = 0}^{n} \left[\sum_{i = j}^{n} B_{i}^{n}(t) B_{j}^{i}(u) \right] \mathbf{P}_{j} \\
> &= \sum_{j = 0}^{n} \left[\sum_{i = j}^{n} \binom{n}{i} t^{i} (1 - t)^{n - i} \binom{i}{j} u^{j} (1 - u)^{i - j} \right] \mathbf{P}_{j} \\
> &= \sum_{j = 0}^{n} \binom{n}{j} (ut)^{j} \left[\sum_{i = j}^{n} \binom{n - j}{i - j} (1 - t)^{n - i} (t - ut)^{i - j} \right] \mathbf{P}_{j} && \binom{n}{i} \binom{i}{j} = \binom{n}{j} \binom{n - j}{i - j} \\
> &= \sum_{j = 0}^{n} \binom{n}{j} (ut)^{j} \left[\sum_{k = 0}^{n - j} \binom{n - j}{k} (1 - t)^{n - j - k} (t - ut)^{k} \right] \mathbf{P}_{j} \\
> &= \sum_{j = 0}^{n} \binom{n}{j} (ut)^{j} (1 - ut)^{n - j} \mathbf{P}_{j} \\
> &= \sum_{j = 0}^{n} B_{j}^{n}(ut) \mathbf{P}_{j} \\
> &= \mathbf{C}(ut)
> \end{aligned}
> $$
> 这说明 $\mathbf{C}_{1}(t)$ 与 $\mathbf{C}(ut)$ 完全贴合。

#### Elevation Property

将一条 $n$ 度的 Bézier 曲线 $\mathbf{C}$ 升阶到 $(n + 1)$ 度的 Bézier 曲线 $\mathbf{C}'$，它的控制点为：
$$
\begin{cases}
\mathbf{Q}_{0} &= \mathbf{P}_{0} \\
\mathbf{Q}_{i} &= \dfrac{i}{n + 1} \mathbf{P}_{i - 1} + \left(1 - \dfrac{i}{n + 1}\right) \mathbf{P}_{i} \\
\mathbf{Q}_{n + 1} &= \mathbf{P}_{n}
\end{cases}
$$

> $$
> \begin{aligned}
> \mathbf{C}'(t) &= \sum_{i = 0}^{n + 1} B_{i}^{n + 1}(t) \mathbf{Q}_{i} \\
> &= B_{0}^{n + 1}(t) \mathbf{P}_{0} + \sum_{i = 1}^{n} B_{i}^{n + 1}(t) \left[\frac{i}{n + 1} \mathbf{P}_{i - 1} + \left(1 - \frac{i}{n + 1}\right) \mathbf{P}_{i}\right] + B_{n + 1}^{n + 1}(t) \mathbf{P}_{n} \\
> &= \sum_{i = 0}^{n} \left[\left(1 - \frac{i}{n + 1}\right) B_{i}^{n + 1}(t) + \frac{i + 1}{n + 1} B_{i + 1}^{n + 1}(t) \right] \mathbf{P}_{i} \\
> &= \sum_{i = 0}^{n} B_{i}^{n}(t) \mathbf{P}_{i} \\
> &= \mathbf{C}(t)
> \end{aligned}
> $$
