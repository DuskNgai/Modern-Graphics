# B-spline Curve

### Notations and Symbols

这里是与 B-spline 曲线相关的符号。

| Symbols | Description |
| :-----: | :---------: |
| $m \in \mathbb{N}$ | Number of knots |
| $n \in \mathbb{N}$ | Number of control points |
| $\{t_{i}\}_{i = 0}^{m}$ | Knots |
| $B_{i}^{p}: [0, 1] \mapsto \mathbb{R}$ | $i$-th B-spline basis function of degree $p$ |
| $\mathbf{P}_{i} \in \mathbb{R}^{d}$ | $i$-th Control Point |

## B-spline Basis Function

这里采用递归的方式定义 B-spline 的基函数。定义在 $\{t_{i}\}_{i = 0}^{m}$ ($t_{i} \le t_{i + 1}$) 上的第 $i$ 个 $p \in \{0, \dots, m - 1\}$ 度的 B-spline 基函数为：
$$
B_{i}^{p}(t) = \frac{t - t_{i}}{t_{i + p} - t_{i}} B_{i}^{p - 1}(t) + \frac{t_{i + p + 1} - t}{t_{i + p + 1} - t_{i + 1}} B_{i + 1}^{p - 1}(t), \quad t \in [t_{i}, t_{i + p + 1})
$$
可以看出它是两个 $p - 1$ 度的 B-spline 基函数的线性组合。而递归终点为：
$$
B_{i}^{0}(t) = \begin{cases}
1, \quad t \in [t_{i}, t_{i + 1}) \\
0, \quad \text{otherwise}
\end{cases}
$$
注意，插值时 $t_{i} = t_{i + p}$ 的话，则规定该分式计算结果为 0。

### Property

#### Local Support

$$
\begin{matrix}
t_{0} \\
 & \searrow \\
 & & B_{0}^{0} \\
 & \nearrow \\
t_{1} & & & \searrow \\
 & \searrow & & & B_{0}^{1} \\
 & & & \nearrow & & \searrow \\
 & & B_{1}^{0} & & & & B_{0}^{2} \\
 & \nearrow & & \searrow & & \nearrow & & \searrow \\
t_{2} & & & & B_{1}^{1} & & \vdots & & B_{0}^{n} \\
 & \searrow & & \nearrow & & \searrow & & \nearrow \\
 & & \vdots & & \vdots & & B_{m - 3}^{2} \\
 & \nearrow & & \searrow & & \nearrow \\
\vdots & & & & B_{m - 2}^{1} \\
 & \searrow & & \nearrow \\
 & & B_{m - 1}^{0} \\
 & \nearrow \\
t_{m}
\end{matrix}
$$

$B_{i}^{p}(t)$ 只在 $t \in [t_{i}, t_{i + p + 1})$ 时非零的。反过来说，$[t_{i}, t_{i + 1})$ 只对 $\{B_{j}^{p}\}_{j = i - p}^{i}$ 有影响。举例来说，$[t_{i}, t_{i + 1})$ 上非零的 0 度基函数为 $B_{i}^{0}$，非零的 1 度基函数为 $B_{i - 1}^{1}$ 和 $B_{i}^{1}$，等等。

#### Partition of Unity

对于任意的节点区间 $[t_{i}, t_{i + 1})$，其上面 $p$ 度非零的基函数之和为 1：
$$
\sum_{j = i - p}^{i} B_{j}^{p}(t) = 1, \quad t \in [t_{i}, t_{i + 1})
$$

> 首先是做一步展开：
> $$
> \sum_{j = i - p}^{i} B_{j}^{p}(t) = \sum_{j = i - p}^{i} \left[\frac{t - t_{j}}{t_{j + p} - t_{j}} B_{j}^{p - 1}(t) + \frac{t_{j + p + 1} - t}{t_{j + p + 1} - t_{j + 1}} B_{j + 1}^{p - 1}(t)\right]
> $$
> 然后是注意处在边界上的基函数，$B_{i - p}^{p - 1}$ 的非零范围为 $[t_{i - p}, t_{i})$，$B_{i + 1}^{p - 1}$ 的非零范围为 $[t_{i + 1}, t_{i + p + 1})$。这两个基函数的非零范围超出了节点区间 $[t_{i}, t_{i + 1})$，因此可以直接忽略。
> $$
> \begin{aligned}
> \sum_{j = i - p}^{i} B_{j}^{p}(t) &= \sum_{\color{red}j = i - p + 1}^{i} \frac{t - t_{j}}{t_{j + p} - t_{j}} B_{j}^{p - 1}(t) + \sum_{j = i - p}^{\color{red}i - 1} \frac{t_{j + p + 1} - t}{t_{j + p + 1} - t_{j + 1}} B_{j + 1}^{p - 1}(t) \\
> &= \sum_{\color{red}j = i - p + 1}^{i} \frac{t - t_{j}}{t_{j + p} - t_{j}} B_{j}^{p - 1}(t) + \sum_{\color{red}j = i - p + 1}^{i} \frac{t_{\color{red}j + p} - t}{t_{\color{red}j + p} - t_{\color{red}j}} B_{\color{red}j}^{p - 1}(t) \\
> &= \sum_{j = i - p + 1}^{i} \left[\frac{t - t_{j}}{t_{j + p} - t_{j}} + \frac{t_{j + p} - t}{t_{j + p} - t_{j}}\right] B_{j}^{p - 1}(t) \\
> &= \sum_{j = i - p + 1}^{i} B_{j}^{p - 1}(t) = \sum_{j = i - p + 2}^{i} B_{j}^{p - 2}(t) = \cdots = \sum_{j = i}^{i} B_{j}^{0}(t) = 1
> \end{aligned}
> $$

#### First Order Derivative

$$
\frac{\mathrm{d}}{\mathrm{d}t} B_{i}^{p}(t) = \frac{p}{t_{i + p} - t_{i}} B_{i}^{p - 1}(t) - \frac{p}{t_{i + p + 1} - t_{i + 1}} B_{i + 1}^{p - 1}(t), \quad t \in [t_{i}, t_{i + p + 1})
$$

> 这里用归纳法证明，后面我们省略 t。$p = 1$ 的时候：
> $$
> \frac{\mathrm{d}}{\mathrm{d}t} B_{i}^{1}(t) = \begin{cases}
> \displaystyle{\frac{\mathrm{d}}{\mathrm{d}t} \left[\frac{t - t_{i}}{t_{i + 1} - t_{i}} \cdot 1 + \frac{t_{i + 1} - t}{t_{i + 1} - t_{i}} \cdot 0\right]} & t \in [t_{i}, t_{i + 1}) \\
> \displaystyle{\frac{\mathrm{d}}{\mathrm{d}t} \left[\frac{t - t_{i}}{t_{i + 1} - t_{i}} \cdot 0 + \frac{t_{i + 1} - t}{t_{i + 1} - t_{i}} \cdot 1\right]} & t \in [t_{i + 1}, t_{i + 2}) \\
> \end{cases}
> = 1, \quad t \in [t_{i}, t_{i + 2})
> $$
> 然后是任意的 $p$：
> $$
> \begin{aligned}
> \frac{\mathrm{d}}{\mathrm{d}t} B_{i}^{p} &= \frac{\mathrm{d}}{\mathrm{d}t} \left[\frac{t - t_{i}}{t_{i + p} - t_{i}} B_{i}^{p - 1} + \frac{t_{i + p + 1} - t}{t_{i + p + 1} - t_{i + 1}} B_{i + 1}^{p - 1}\right] \\
> &= \frac{1}{t_{i + p} - t_{i}} B_{i}^{p - 1} + \frac{t - t_{i}}{t_{i + p} - t_{i}} \frac{\mathrm{d}}{\mathrm{d}t} B_{i}^{p - 1} \\
> &- \frac{1}{t_{i + p + 1} - t_{i + 1}} B_{i + 1}^{p - 1} + \frac{t_{i + p + 1} - t}{t_{i + p + 1} - t_{i + 1}} \frac{\mathrm{d}}{\mathrm{d}t} B_{i + 1}^{p - 1} \\
> \end{aligned}
> $$
> 带入导数的表达式得到：
> $$
> \begin{aligned}
> \frac{\mathrm{d}}{\mathrm{d}t} B_{i}^{p} &= \frac{1}{t_{i + p} - t_{i}} B_{i}^{p - 1} + \frac{t - t_{i}}{t_{i + p} - t_{i}} \left[\frac{p - 1}{t_{i + p - 1} - t_{i}} B_{i}^{p - 2} - \frac{p - 1}{t_{i + p} - t_{i + 1}} B_{i + 1}^{p - 2}\right] \\
> &- \frac{1}{t_{i + p + 1} - t_{i + 1}} B_{i + 1}^{p - 1} + \frac{t_{i + p + 1} - t}{t_{i + p + 1} - t_{i + 1}} \left[\frac{p - 1}{t_{i + p} - t_{i + 1}} B_{i + 1}^{p - 2} - \frac{p - 1}{t_{i + p + 1} - t_{i + 2}} B_{i + 2}^{p - 2}\right] \\
> &= \frac{1}{t_{i + p} - t_{i}} B_{i}^{p - 1} - \frac{1}{t_{i + p + 1} - t_{i + 1}} B_{i + 1}^{p - 1} \\
> &+ \frac{p - 1}{t_{i + p - 1} - t_{i}} \frac{t - t_{i}}{t_{i + p} - t_{i}} B_{i}^{p - 2} \\
> &+ \frac{p - 1}{t_{i + p} - t_{i + 1}} \left[\frac{t_{i + p + 1} - t}{t_{i + p + 1} - t_{i + 1}} - \frac{t - t_{i}}{t_{i + p} - t_{i}}\right] B_{i + 1}^{p - 2} \\
> &- \frac{p - 1}{t_{i + p + 1} - t_{i + 2}} \frac{t_{i + p + 1} - t}{t_{i + p + 1} - t_{i + 1}} B_{i + 2}^{p - 2} \\
> \end{aligned}
> $$
> 由于：
> $$
> \begin{aligned}
> \frac{t_{i + p + 1} - t}{t_{i + p + 1} - t_{i + 1}} - \frac{t - t_{i}}{t_{i + p} - t_{i}} &= - 1 + \frac{t_{i + p + 1} - t}{t_{i + p + 1} - t_{i + 1}} + 1 - \frac{t - t_{i}}{t_{i + p} - t_{i}} \\
> &= \frac{t_{i + p} - t}{t_{i + p} - t_{i}} - \frac{t - t_{i + 1}}{t_{i + p + 1} - t_{i + 1}} \\
> \end{aligned}
> $$
> 因此可以得到：
> $$
> \begin{aligned}
> \frac{\mathrm{d}}{\mathrm{d}t} B_{i}^{p} &= \frac{1}{t_{i + p} - t_{i}} B_{i}^{p - 1} - \frac{1}{t_{i + p + 1} - t_{i + 1}} B_{i + 1}^{p - 1} \\
> &+ \frac{p - 1}{t_{i + p} - t_{i}} \underbrace{\left[\frac{t - t_{i}}{t_{i + p - 1} - t_{i}} B_{i}^{p - 2} + \frac{t_{i + p} - t}{t_{i + p} - t_{i + 1}} B_{i + 1}^{p - 2}\right]}_{B_{i}^{p - 1}} \\
> &- \frac{p - 1}{t_{i + p + 1} - t_{i + 1}} \underbrace{\left[\frac{t - t_{i + 1}}{t_{i + p} - t_{i + 1}} B_{i + 1}^{p - 2} + \frac{t_{i + p + 1} - t}{t_{i + p + 1} - t_{i + 2}} B_{i + 2}^{p - 2}\right]}_{B_{i + 1}^{p - 1}} \\
> &= \frac{p}{t_{i + p} - t_{i}} B_{i}^{p - 1} - \frac{p}{t_{i + p + 1} - t_{i + 1}} B_{i + 1}^{p - 1} \\
> \end{aligned}
> $$


## B-spline Curve

$p$ 度的 B-spline 定义为：
$$
\mathbf{C}(t) = \sum_{i = 0}^{n} B_{i}^{p}(t) \mathbf{P}_{i}
$$
其中 $B_{i}^{p}$ 是定义在 $\{\underbrace{0, \dots, 0}_{p + 1}, t_{p + 1}, \dots, t_{m - p - 1}, \underbrace{1, \dots, 1}_{p + 1}\}$ 上的 $p$ 度的 B-spline 基函数。控制点数量 $n + 1$、节点数量 $m + 1$、多项式次数 $p + 1$ 之间满足：
$$
m = n + p + 1
$$

## Example

### Bézier Curve is a Special Case of B-spline Curve

容易验证，当节点为 $\{\underbrace{0, \dots, 0}_{n + 1}, \underbrace{1, \dots, 1}_{n + 1}\}$ 时，全体的 $n$ 次 B-spline 基函数都是 Bernstein 多项式，因此 Bézier 曲线就是 Bernstein 多项式曲线的特例。
