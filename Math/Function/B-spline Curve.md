# B-spline Curve

### Notations and Symbols

这里是与 B-spline 曲线相关的符号。

| Symbols | Description |
| :-----: | :---------: |
| $\{t_{i}\}_{i = 0}^{m}$ | Knots |
| $N_{i}^{p}$ | $i$-th B-spline basis function of degree $p$ |

## B-spline Basis Function

这里采用递归的方式定义 B-spline 的基函数。定义在 $\{t_{i}\}_{i = 0}^{m}$ 上的第 $i$ 个 $p$ 度的 B-spline 基函数为：
$$
N_{i}^{p}(t) = \frac{t - t_{i}}{t_{i + p} - t_{i}} N_{i}^{p - 1}(t) + \frac{t_{i + p + 1} - t}{t_{i + p + 1} - t_{i + 1}} N_{i + 1}^{p - 1}(t), \quad t \in [t_{i}, t_{i + p + 1})
$$
可以看出它是两个 $p - 1$ 度的 B-spline 基函数的线性组合。而递归终点为：
$$
N_{i}^{0}(t) = \begin{cases}
1, \quad t \in [t_{i}, t_{i + 1}) \\
0, \quad \text{otherwise}
\end{cases}
$$
注意，插值时分母为 0 的话，则规定该分式计算结果为 0。

### Property

#### Local Support

$N_{i}^{p}(t)$ 只在 $t \in [t_{i}, t_{i + p + 1})$ 时非 0 的。

#### Partition of Unity

$$
\sum_{j = i - p}^{i} N_{j}^{p}(t) = 1
$$

#### First Order Derivative

$$
\frac{\mathrm{d}}{\mathrm{d}t} N_{i}^{p}(t) = \frac{p}{t_{i + p} - t_{i}} N_{i}^{p - 1}(t) - \frac{p}{t_{i + p + 1} - t_{i + 1}} N_{i + 1}^{p - 1}(t)
$$

> 这里用归纳法证明。$p = 1$ 的时候的是显然的。后面我们省略 t：
> $$
> \begin{aligned}
> \frac{\mathrm{d}}{\mathrm{d}t} N_{i}^{p} &= \frac{\mathrm{d}}{\mathrm{d}t} \left[\frac{t - t_{i}}{t_{i + p} - t_{i}} N_{i}^{p - 1} + \frac{t_{i + p + 1} - t}{t_{i + p + 1} - t_{i + 1}} N_{i + 1}^{p - 1}\right] \\
> &= \frac{1}{t_{i + p} - t_{i}} N_{i}^{p - 1} + \frac{t - t_{i}}{t_{i + p} - t_{i}} \frac{\mathrm{d}}{\mathrm{d}t} N_{i}^{p - 1} \\
> &- \frac{1}{t_{i + p + 1} - t_{i + 1}} N_{i + 1}^{p - 1} + \frac{t_{i + p + 1} - t}{t_{i + p + 1} - t_{i + 1}} \frac{\mathrm{d}}{\mathrm{d}t} N_{i + 1}^{p - 1} \\
> \end{aligned}
> $$
> 带入得到：
> $$
> \begin{aligned}
> \frac{\mathrm{d}}{\mathrm{d}t} N_{i}^{p} &= \frac{1}{t_{i + p} - t_{i}} N_{i}^{p - 1} + \frac{t - t_{i}}{t_{i + p} - t_{i}} \left[\frac{p - 1}{t_{i + p - 1} - t_{i}} N_{i}^{p - 2} - \frac{p - 1}{t_{i + p} - t_{i + 1}} N_{i + 1}^{p - 2}\right] \\
> &- \frac{1}{t_{i + p + 1} - t_{i + 1}} N_{i + 1}^{p - 1} + \frac{t_{i + p + 1} - t}{t_{i + p + 1} - t_{i + 1}} \left[\frac{p - 1}{t_{i + p} - t_{i + 1}} N_{i + 1}^{p - 2} - \frac{p - 1}{t_{i + p + 1} - t_{i + 2}} N_{i + 2}^{p - 2}\right] \\
> &= \frac{1}{t_{i + p} - t_{i}} N_{i}^{p - 1} - \frac{1}{t_{i + p + 1} - t_{i + 1}} N_{i + 1}^{p - 1} \\
> &+ \frac{p - 1}{t_{i + p - 1} - t_{i}} \frac{t - t_{i}}{t_{i + p} - t_{i}} N_{i}^{p - 2} \\
> &+ \frac{p - 1}{t_{i + p} - t_{i + 1}} \left[\frac{t_{i + p + 1} - t}{t_{i + p + 1} - t_{i + 1}} - \frac{t - t_{i}}{t_{i + p} - t_{i}}\right] N_{i + 1}^{p - 2} \\
> &- \frac{p - 1}{t_{i + p + 1} - t_{i + 2}} \frac{t_{i + p + 1} - t}{t_{i + p + 1} - t_{i + 1}} N_{i + 2}^{p - 2} \\
> \end{aligned}
> $$
> 由于：
> $$
> \begin{aligned}
> \frac{t_{i + p + 1} - t}{t_{i + p + 1} - t_{i + 1}} - \frac{t - t_{i}}{t_{i + p} - t_{i}} &=
> \end{aligned}
> $$

## Example

### Bézier Curve is a Special Case of B-spline Curve

容易验证，当节点为 $\{\underbrace{0, \dots, 0}_{n + 1}, \underbrace{1, \dots, 1}_{n + 1}\}$ 时，所有 $n$ 次 B-spline 基函数都是 Bernstein 多项式，因此 Bézier 曲线就是 Bernstein 多项式曲线的特例。
