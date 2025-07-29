# Bézier Curve

### Notations and Symbols

这些是与 Bernstein 多项式和 Bézier 曲线相关的符号。

| Symbol | Description |
| :----: | :---------: |
| $n \in \mathbb{N}$ | Order of Bernstein Polynomial |
| $B_{i}^{n}: \mathbb{R} \mapsto \mathbb{R}$ | Bernstein Polynomial |
| $\mathbf{C}(t): [0, 1] \mapsto \mathbb{R}^{d}$ | Bézier Curve |
| $\mathbf{P}_{i} \in \mathbb{R}^{d}$ | Control Point |

## Resources

1. [Wikipedia](https://en.wikipedia.org/wiki/Bernstein_polynomial)
2. [The NURBS Book](https://link.springer.com/book/10.1007/978-3-642-59223-2)

## Bernstein Polynomial

定义 Bernstein 多项式 $B_{i}^{n}: \mathbb{R} \mapsto \mathbb{R}$ 为：
$$
B_{i}^{n}(t) = \binom{n}{i} t^{i} (1 - t)^{n - i}
$$
其中 $i \in \{0, \dots, n\}$，$t \in [0, 1]$。

### Properties

#### Non-negativity

$$
B_{i}^{n}(t) \geq 0
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
\mathbf{C}(t) = \sum_{i = 0}^{n} B_{i}^{n}(t) \mathbf{P}_{i}, t \in [0, 1],
$$

### Properties

很多 Bézier 曲线的性质都可以从 Bernstein 多项式的性质中推导出来。这里额外介绍一些 Bézier 曲线的性质。

#### Variation Diminishing Property

Bézier 曲线的变化不超过控制多边形的变化。也就是说，Bézier 曲线的极值点不超过控制点的个数。

#### Convex Hull Property

Bézier 曲线的所有点都在控制多边形的凸包内。
