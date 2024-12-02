# Spline

### Notations and Symbols

这些是与曲线相关的符号。

| Symbol | Description |
| :----: | :---------: |
| $n \in \mathbb{N}$ | Number of Order |
| $t \in [0, 1]$ | Curve Parameter |
| $\mathbf{p}_{i} \in \mathbb{R}^{d}$ | Control Points |
| $\mathbf{p}_{0, n}: \mathbb{R} \mapsto \mathbb{R}^{d}$ | Bezier Curve |

## Spline Curve

### Bezier Curve

#### Definition

给定一组控制点 $\mathbf{p}_{0}, \dots, \mathbf{p}_{n}$，定义 Bezier 曲线 $\mathbf{p}_{0, n}(t)$ 为：
$$
\mathbf{p}_{0, n}(t) = \sum_{i = 0}^{n} \mathbf{p}_{i} B_{i}^{n}(t)
$$
其中 $B_{i}^{n}: \mathbb{R} \mapsto \mathbb{R}$ 是 Bernstein 多项式，定义为：
$$
B_{i}^{n}(t) = \binom{n}{i} t^{i} (1 - t)^{n - i}
$$
矩阵形式的 Bezier 曲线为：
$$
\mathbf{p}_{0, n}(t) = 
\underbrace{\begin{bmatrix} \mathbf{p}_{0} & \cdots & \mathbf{p}_{n} \end{bmatrix}}_{\mathbb{R}^{d \times n}}
\underbrace{\left[
(-1)^{j - i}\binom{n}{i}\binom{n - i}{n - j}
\right]_{ij}}_{\mathbb{R}^{n \times n}}
\underbrace{\begin{bmatrix} 1 \\ \vdots \\ t^{n} \end{bmatrix}}_{\mathbb{R}^{n \times 1}}
$$

#### De Casteljau Algorithm

De Casteljau 算法可以用于计算 Bezier 曲线。具体来说，它递归地计算了曲线上点的坐标：
$$
\mathbf{p}_{0, n}(t) = (1 - t) \mathbf{p}_{0, n - 1}(t) + t \mathbf{p}_{1, n}(t)
$$

---

$$
\begin{aligned}
\mathbf{p}_{0, n}(t) &= \sum_{i = 0}^{n} \mathbf{p}_{i} B_{i}^{n}(t) \\
&= \sum_{i = 0}^{n} \mathbf{p}_{i} \left[(1 - t)B_{i}^{n - 1}(t) + tB_{i - 1}^{n - 1}(t)\right] \\
&= (1 - t) \sum_{i = 0}^{\color{red}n - 1} \mathbf{p}_{i} B_{i}^{n - 1}(t) + t \sum_{\color{red}i = 1}^{n} \mathbf{p}_{i} B_{i - 1}^{n - 1}(t) \\
&= (1 - t) \mathbf{p}_{0, n - 1}(t) + t \mathbf{p}_{1, n}(t)
\end{aligned}
$$

#### Tangent

Bezier 曲线的一阶导数为：
$$
\frac{\mathrm{d}}{\mathrm{d}t} \mathbf{p}_{0, n}(t) = n \left[\mathbf{p}_{1, n}(t) - \mathbf{p}_{0, n - 1}(t)\right]
$$

---

$$
\begin{aligned}
\frac{\mathrm{d}}{\mathrm{d}t} \mathbf{p}_{0, n}(t) &= \frac{\mathrm{d}}{\mathrm{d}t} \sum_{i = 0}^{n} \mathbf{p}_{i} B_{i}^{n}(t) \\
&= \sum_{i = 0}^{n} \mathbf{p}_{i} \frac{\mathrm{d}}{\mathrm{d}t} B_{i}^{n}(t) \\
&= \sum_{i = 0}^{n} \mathbf{p}_{i} n \left[B_{i - 1}^{n - 1}(t) - B_{i}^{n - 1}(t)\right] \\
&= n \left[\sum_{\color{red}i = 1}^{n} \mathbf{p}_{i} B_{i - 1}^{n - 1}(t) - \sum_{i = 0}^{\color{red}n - 1} \mathbf{p}_{i} B_{i}^{n - 1}(t)\right] \\
&= n \left[\mathbf{p}_{1, n}(t) - \mathbf{p}_{0, n - 1}(t)\right]
\end{aligned}
$$

#### Curvature

Bezier 曲线的二阶导数为：
$$
\frac{\mathrm{d}^{2}}{\mathrm{d}t^{2}} \mathbf{p}_{0, n}(t) = n (n - 1) \left[\mathbf{p}_{2, n}(t) - 2 \mathbf{p}_{1, n - 1}(t) + \mathbf{p}_{0, n - 2}(t)\right]
$$
因此曲率为：
$$
\kappa(t) = \frac{\dfrac{\mathrm{d}^{2}}{\mathrm{d}t^{2}} \mathbf{p}_{0, n}(t)}{\left[1 + \left(\dfrac{\mathrm{d}}{\mathrm{d}t} \mathbf{p}_{0, n}(t)\right)^{2}\right]^{\frac{3}{2}}}
$$

#### Splitting

对于一个给定的 Bezier 曲线 $\mathbf{p}_{0, n}(t)$，可以将其在任意的 $\tau \in [0, 1]$ 的位置分割为两个阶数相同的 Bezier 曲线。设处于 $u \in [0, \tau]$ 段的 Bezier 曲线为 $\mathbf{q}_{0, n}(u)$，则其控制点满足：
$$
\mathbf{q}_{i} = \mathbf{p}_{0, i}(\tau)
$$
处于 $u \in [\tau, 1]$ 段的 Bezier 曲线为 $\mathbf{r}_{0, n}(u)$，则其控制点满足：
$$
\mathbf{r}_{i} = \mathbf{p}_{i, n}(\tau)
$$

---

$$
\begin{aligned}
\mathbf{q}_{0, n}(u) &= \sum_{i = 0}^{n} \mathbf{q}_{i} B_{i}^{n}(u) \\
&= \sum_{i = 0}^{n} \mathbf{p}_{0, i}(\tau) B_{i}^{n}(u) \\
&= \sum_{i = 0}^{n} \sum_{j = 0}^{i} \mathbf{p}_{j} B_{j}^{i}(\tau) B_{i}^{n}(u) \\
&= \sum_{\color{red}j = 0}^{n} \mathbf{p}_{j} \sum_{\color{red}i = j}^{\color{red}n} B_{j}^{i}(\tau) B_{i}^{n}(u) \\
&= \sum_{j = 0}^{n} \mathbf{p}_{j} \sum_{i = j}^{n} \binom{i}{j} \tau^{j} (1 - \tau)^{i - j} \binom{n}{i} u^{i} (1 - u)^{n - i} \\
&= \sum_{j = 0}^{n} \mathbf{p}_{j} (\tau u)^{j} \sum_{i = j}^{n} \binom{n}{i} \binom{i}{j} [(1 - \tau) u]^{i - j} (1 - u)^{n - i} \\
&= \sum_{j = 0}^{n} \mathbf{p}_{j} \binom{n}{j} (\tau u)^{j} \sum_{i = j}^{n} \binom{n - i}{i - j} (u- \tau u)^{i - j} (1 - u)^{n - i} \\
&= \sum_{j = 0}^{n} \mathbf{p}_{j} \binom{n}{j} (\tau u)^{j} \sum_{i = 0}^{n - j} \binom{n - j - i}{i} (u- \tau u)^{i} (1 - u)^{n - j - i} \\
&= \sum_{j = 0}^{n} \mathbf{p}_{j} \binom{n}{j} (\tau u)^{j} (1 - u + u - \tau u)^{n - j} \\
&= \mathbf{p}_{0, n}(\tau u)
\end{aligned}
$$

#### Linear Transform

对于一个给定的 Bezier 曲线 $\mathbf{p}_{0, n}(t)$，可以对其进行线性变换。设变换矩阵为 $A \in \mathbb{R}^{d \times d}$，则进行线性变换后的曲线的控制点为：
$$
\mathbf{q}_{i} = A \mathbf{p}_{i}
$$

---

$$
\begin{aligned}
A\mathbf{p}_{0, n}(t) &= A\left(\sum_{i = 0}^{n} \mathbf{p}_{i} B_{i}^{n}(t)\right) \\
&= \sum_{i = 0}^{n} (A\mathbf{p}_{i}) B_{i}^{n}(t)
\end{aligned}
$$

### Continuity

| Order | Meaning |
| :---: | :-----: |
| $G_{0}$ | Curves are continuous |
| $C_{0}$ | Same as $G_{0}$ |
| $G_{1}$ | Tangents are continuous |
| $C_{1}$ | Tangents are same |
| $G_{2}$ | Curvatures are continuous |
| $C_{2}$ | Curvatures are same |


