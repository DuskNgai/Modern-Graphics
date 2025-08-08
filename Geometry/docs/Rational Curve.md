# Rational Curve

圆锥曲线不能表示为多项式曲线，但是可以表示为有理式曲线。比如半径为 $1$ 的圆（的第一象限部分）可以表示为：
$$
x(t) = \frac{1 - t^{2}}{1 + t^{2}}, \quad y(t) = \frac{2t}{1 + t^{2}}, \quad t \in [0, 1]
$$

## Rational Bézier/B-spline Curve

因此拓展 Bézier/B-spline 曲线为有理式 Bézier/B-spline 曲线：
$$
\mathbf{C}(t) = \frac{\sum_{i = 0}^{n} B_{i}^{n}(t) w_{i} \mathbf{P}_{i}}{\sum_{i = 0}^{n} B_{i}^{n}(t) w_{i}}, \quad t \in [0, 1],
$$
其中 $w_{i} > 0$ 是权重。定义：
$$
R_{i}^{n}(t) = \frac{B_{i}^{n}(t) w_{i}}{\sum_{j = 0}^{n} B_{j}^{n}(t) w_{j}}, \quad t \in [0, 1]
$$
则有理式 Bézier 曲线可以表示为：
$$
\mathbf{C}(t) = \sum_{i = 0}^{n} R_{i}^{n}(t) \mathbf{P}_{i}, \quad t \in [0, 1]
$$

特别是有一种表达能力特别强的曲线，称为 **Non-uniform Rational B-spline (NURBS)** 曲线。

### Projection

对于任意点 $\mathbf{P} = (x, y, z)$, 定义其齐次坐标 $\mathbf{P}^{w} = (wx, wy, wz, w)$, ($w \ne 0$)。定义投影变换 $H: \mathbb{R}^{4} \to \mathbb{R}^{3}$ 为 $H(\mathbf{P}^{w}) = \mathbf{P}$。利用齐次坐标定义的多项式 Bézier 曲线
$$
\mathbf{C}^{w}(t) = \sum_{i = 0}^{n} B_{i}^{n}(t) \mathbf{P}_{i}^{w}, \quad t \in [0, 1]
$$
然后应用投影变换 $H$ 得到有理式 Bézier 曲线：
$$
\mathbf{C}(t) = H(\mathbf{C}^{w}(t))
$$

### Properties

Convex Hull Property, Variation Diminishing Property 仍然是满足的。

#### First Order Derivatives

一阶导数用齐次坐标下的对应曲线表示会简单一点。

> 令：
> $$
> w(t) = \sum_{i = 0}^{n} B_{i}^{n}(t) w_{i}
> $$
> 是有理式曲线的权重函数，
> $$
> \mathbf{A}(t) := w(t) \mathbf{C}(t)
> $$
> 是齐次坐标下的有理式曲线 $\mathbf{C}^{w}$ 的前三个分量，是一个多项式曲线。
> $$
> \begin{aligned}
> \frac{\mathrm{d}}{\mathrm{d}t} \mathbf{C}(t) &= \frac{\mathrm{d}}{\mathrm{d}t} \frac{w(t) \mathbf{C}(t)}{w(t)} \\
> &= \frac{w(t) \dfrac{\mathrm{d}}{\mathrm{d}t} \mathbf{A}(t) - \dfrac{\mathrm{d}}{\mathrm{d}t} w(t) \mathbf{A}(t)}{w(t)^{2}} \\
> &= \frac{w(t) \dfrac{\mathrm{d}}{\mathrm{d}t} \mathbf{A}(t) - \dfrac{\mathrm{d}}{\mathrm{d}t} w(t) \cdot w(t) \mathbf{C}(t)}{w(t)^{2}} \\
> &= \frac{\dfrac{\mathrm{d}}{\mathrm{d}t} \mathbf{A}(t) - \dfrac{\mathrm{d}}{\mathrm{d}t} w(t) \cdot \mathbf{C}(t)}{w(t)}
> \end{aligned}
> $$

## Examples

### Rational Bézier Curve For Circle

求解半径为 $1$ 的圆的第一象限部分的有理式 Bézier 曲线。

> 显然，圆是分子分母的二次曲线。我们先关注分母部分：
> $$
> 1 + t^{2} = \sum_{i = 0}^{2} B_{i}^{2}(t) w_{i} = (1 - t)^{2}w_{0} + 2t(1 - t)w_{1} + t^{2}w_{2}
> $$
> 求解线性方程组得到：
> $$
> w_{0} = 1, \quad w_{1} = 1, \quad w_{2} = 2
> $$
> 再看分子部分，需要满足：
> $$
> \begin{cases}
> 1 - t^{2} = (1 - t)^{2}w_{0}P_{0x} + 2t(1 - t)w_{1}P_{1x} + t^{2}w_{2}P_{2x} \\
> 2t = (1 - t)^{2}w_{0}P_{0y} + 2t(1 - t)w_{1}P_{1y} + t^{2}w_{2}P_{2y}
> \end{cases}
> $$
> 而且有 $P_{0} = (1, 0)$，$P_{2} = (0, 1)$。因此得到 $P_{1} = (1, 1)$。
