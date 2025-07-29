# Path Tracing Theory

路径追踪的理论基础主要源自于辐射传输理论。

## Preliminaries

以立体角为微元的渲染方程：
$$
L_{o}(\mathbf{x}, \boldsymbol{\omega}) = L_{e}(\mathbf{x}, \boldsymbol{\omega}) + \int_{\mathbb{S}^{2}} L_{i}(\mathbf{x}, \boldsymbol{\omega}') f_{r}(\mathbf{x}, \boldsymbol{\omega}, \boldsymbol{\omega}') |\cos \langle \boldsymbol{\omega}', \mathbf{n}(\mathbf{x}) \rangle| \mathrm{d}\boldsymbol{\omega}',
$$
以面积为微元的渲染方程：
$$
L_{o}(\mathbf{x}, \boldsymbol{\omega}) = L_{e}(\mathbf{x}, \boldsymbol{\omega}) + \int_{A} L_{i}(\mathbf{y}, -\boldsymbol{\omega}') f_{r}(\mathbf{x}, \boldsymbol{\omega}, \boldsymbol{\omega}') V(\mathbf{x}, \mathbf{y}) \frac{|\cos \langle \boldsymbol{\omega}', \mathbf{n}(\mathbf{x}) \rangle| |\cos \langle -\boldsymbol{\omega}', \mathbf{n}(\mathbf{y}) \rangle|}{\|\mathbf{x} - \mathbf{y}\|^{2}} \mathrm{d}\mathbf{y},
$$

其中的 $L_{i}$ 可以由另外一个表面反射而来。因此，该方程可以不断的递归下去。

---

> 在一个圆球的内部，其表面材质是 Lambertian 的，即 $f_{r}(\mathbf{x}, \boldsymbol{\omega}, \boldsymbol{\omega}') = c$，且每一点向外辐射均为 $L_{e}$，求该圆球内表面任意一点任意方向的辐射亮度 $L$。

$$
L = L_{e} + \int_{\mathbb{S}^{2}} c L |\cos \langle \boldsymbol{\omega}', \mathbf{n}(\mathbf{x}) \rangle| \mathrm{d}\boldsymbol{\omega}' = L_{e} + c \pi L \implies L = \frac{L_{e}}{1 - c \pi}.
$$

---



## The Path Space

光源发出的光线可以经过任意多次反射和折射，形成一条光路。我们把所有可能的路径收集起来，形成一个路径空间 $\mathcal{P}$。每条路径 $\mathbf{p} \in \mathcal{P}$ 都可以表示为一系列的顶点 $\mathbf{x}_{0}, \mathbf{x}_{1}, \ldots, \mathbf{x}_{n}$。因此，辐射传输方程可以表示为：
$$
L(\mathbf{x}_{1} \to \mathbf{x}_{0}) = L_{e}(\mathbf{x}_{1} \to \mathbf{x}_{0}) + \int_{A_{2}} L(\mathbf{x}_{2} \to \mathbf{x}_{1}) f_{r}(\mathbf{x}_{2} \to \mathbf{x}_{1} \to \mathbf{x}_{0}) V(\mathbf{x}_{1}, \mathbf{x}_{2}) G(\mathbf{x}_{1}, \mathbf{x}_{2}) \mathrm{d}\mathbf{x}_{2},
$$
再递归一层得到：
$$
\begin{aligned}
L(\mathbf{x}_{1} \to \mathbf{x}_{0}) &= L_{e}(\mathbf{x}_{1} \to \mathbf{x}_{0}) \\
&+ \int_{A_{2}} L_{e}(\mathbf{x}_{2} \to \mathbf{x}_{1}) f_{r}(\mathbf{x}_{2} \to \mathbf{x}_{1} \to \mathbf{x}_{0}) V(\mathbf{x}_{1}, \mathbf{x}_{2}) G(\mathbf{x}_{1}, \mathbf{x}_{2}) \mathrm{d}\mathbf{x}_{2} \\
&+ \int_{A_{2}} \int_{A_{3}} L(\mathbf{x}_{3} \to \mathbf{x}_{2}) f_{r}(\mathbf{x}_{3} \to \mathbf{x}_{2} \to \mathbf{x}_{1}) V(\mathbf{x}_{2}, \mathbf{x}_{3}) G(\mathbf{x}_{2}, \mathbf{x}_{3}) f_{r}(\mathbf{x}_{2} \to \mathbf{x}_{1} \to \mathbf{x}_{0}) V(\mathbf{x}_{1}, \mathbf{x}_{2}) G(\mathbf{x}_{1}, \mathbf{x}_{2}) \mathrm{d}\mathbf{x}_{3} \mathrm{d}\mathbf{x}_{2}
\end{aligned},
$$
显然，无限递归下去得到：
$$
L(\mathbf{x}_{1} \to \mathbf{x}_{0}) = L_{e}(\mathbf{x}_{1} \to \mathbf{x}_{0}) + \sum_{i = 2}^{\infty}
\int_{A_{2}} \cdots \int_{A_{i}} L_{e}(\mathbf{x}_{i} \to \mathbf{x}_{i - 1}) \left[ \prod_{j = 2}^{i} f_{r}(\mathbf{x}_{j} \to \mathbf{x}_{j - 1} \to \mathbf{x}_{j - 2}) V(\mathbf{x}_{j-1}, \mathbf{x}_{j}) G(\mathbf{x}_{j - 1}, \mathbf{x}_{j}) \right] \mathrm{d}\mathbf{x}_{i} \cdots \mathrm{d}\mathbf{x}_{2}.
$$

