# Volume Rendering Theory

体像可以看作由无数微小颗粒组成的集合。当光线进入体像时，会与这些微小颗粒发生相互作用，如吸收、散射、激发等现象。然而，在实际渲染过程中，直接以微小颗粒进行精确的光纤传输建模是不现实的，因为这将带来极其庞大的计算量。因此，我们采取折衷的方法，通过建模光线在体像中平均行为的方式，来近似描述这些复杂的交互过程。

## Radiance Transfer Theory

#### Meaning of $(\boldsymbol{\omega} \cdot \nabla)$

$(\boldsymbol{\omega} \cdot \nabla)L$ 表示 $L$ 在 $\boldsymbol{\omega}$ 上的方向导数，也等价于 $\nabla_{\boldsymbol{\omega}}L$ 或 $\nabla L \cdot \boldsymbol{\omega}$。根据光线参数化 $\mathbf{x} = \mathbf{x}_{0} + t\boldsymbol{\omega}$，有：
$$
(\boldsymbol{\omega} \cdot \nabla)L = \underbrace{\nabla L}_{\mathbb{R}^{3 \times 1}} \cdot \boldsymbol{\omega} = \underbrace{\frac{\partial L}{\partial \mathbf{x}}}_{\mathbb{R}^{1 \times 3}} \frac{\partial \mathbf{x}}{\partial t} = \frac{\mathrm{d}L}{\mathrm{d}t}
$$

### Absorption

当光线与微粒发生碰撞时，光线的辐射能被微粒吸收，转化为微粒的内能。
$$
(\boldsymbol{\omega} \cdot \nabla) L(\mathbf{x}, \boldsymbol{\omega}) = - \sigma_{a}(\mathbf{x}) L(\mathbf{x}, \boldsymbol{\omega})
$$

### Emission

在光线传播的路径上，微粒可能会自行或者受激发射辐射能。
$$
(\boldsymbol{\omega} \cdot \nabla) L(\mathbf{x}, \boldsymbol{\omega}) = \sigma_{a}(\mathbf{x}) L_{e}(\mathbf{x}, \boldsymbol{\omega})
$$
其中 $\sigma_{a}(\mathbf{x})L_{e}(\mathbf{x}, \boldsymbol{\omega})$ 表示微粒的发射能量。实际上这里只需要 $L_{e}(\mathbf{x}, \boldsymbol{\omega})$ 即可表示微粒的发射能量，与 $\sigma_{a}(\mathbf{x})$ 相乘是为了达成形式上的一致性。

### Out-Scattering

当光线与微粒发生碰撞时，光线的辐射能被微粒散射，能量辐射到其他方向上。
$$
(\boldsymbol{\omega} \cdot \nabla) L(\mathbf{x}, \boldsymbol{\omega}) = - \sigma_{s}(\mathbf{x}) L(\mathbf{x}, \boldsymbol{\omega})
$$

### In-Scattering

在光线传播的路径上，其他方向的辐射能量会被微粒散射，转移到当前方向上。
$$
(\boldsymbol{\omega} \cdot \nabla) L(\mathbf{x}, \boldsymbol{\omega}) = \sigma_{s}(\mathbf{x}) \int_{\mathbb{S}^{2}} f_{p}(\mathbf{x}, \boldsymbol{\omega}, \boldsymbol{\omega}') L(\mathbf{x}, \boldsymbol{\omega}') \mathrm{d}\boldsymbol{\omega}' = \sigma_{s}(\mathbf{x}) L_{s}(\mathbf{x}, \boldsymbol{\omega})
$$

### Radiance Transfer Equation

$$
(\boldsymbol{\omega} \cdot \nabla) L(\mathbf{x}, \boldsymbol{\omega}) = \underbrace{-\sigma_{t}(\mathbf{x})L(\mathbf{x}, \boldsymbol{\omega})}_{\text{Losses}} + \underbrace{\sigma_{a}(\mathbf{x})L_{e}(\mathbf{x}, \boldsymbol{\omega}) + \sigma_{s}(\mathbf{x})L_{s}(\mathbf{x}, \boldsymbol{\omega})}_{\text{Gains}}
$$

## Volume Rendering Equation

Volume Rendering Equation 是 Radiance Transfer Equation 的空间积分形式。为了便于分析，我们首先将其改写为关于 $t$ 的微分形式：
$$
\frac{\mathrm{d}}{\mathrm{d}t}L(t) = -\sigma_{t}(\mathbf{x}_{t})L(t) + \sigma_{a}(\mathbf{x}_{t})L_{e}(t) + \sigma_{s}(\mathbf{x}_{t})L_{s}(t),
$$
光线的起点可以是光源、表面或者其他体像位置。设光线的起点为 $\mathbf{x}{d}$，终点为 $\mathbf{x}{0}$。在实际光线追踪实现中，通常定义光线的参数化方向与上述公式中的光线方向相反，因此光线沿负方向传播。将光线重新参数化为 $\mathbf{x} = \mathbf{x}_{0} - t\boldsymbol{\omega}$，代入上述公式后，得到：
$$
L(0) = L(d)\exp\left(-\int_{0}^{d}\sigma_{t}(\mathbf{x}_{t}) \mathrm{d}t\right) + \int_{0}^{d}\left[\sigma_{a}(\mathbf{x}_{t})L_{e}(t) + \sigma_{s}(\mathbf{x}_{t})L_{s}(t)\right] \exp\left(-\int_{0}^{t}\sigma_{t}(\mathbf{x}_{s}) \mathrm{d}s\right) \mathrm{d}t.
$$
进一步改写为坐标形式为：
$$
L(\mathbf{x}_{0}, \boldsymbol{\omega}) = L(\mathbf{x}_{d}, \boldsymbol{\omega})\exp\left(-\int_{0}^{d}\sigma_{t}(\mathbf{x}_{t})\mathrm{d}t\right) + \int_{0}^{d}\left[\sigma_{a}(\mathbf{x}_{t})L_{e}(\mathbf{x}_{t}, \boldsymbol{\omega}) + \sigma_{s}(\mathbf{x}_{t})L_{s}(\mathbf{x}_{t}, \boldsymbol{\omega})\right] \exp\left(-\int_{0}^{t}\sigma_{t}(\mathbf{x}_{s}) \mathrm{d}s\right) \mathrm{d}t
$$

### Transmittance

在上述解析解中，光的衰减由一个指数衰减项控制。我们将其定义为透射率：
$$
T(t) = \exp\left(-\int_{0}^{t}\sigma_{t}(\mathbf{x}_{s}) \mathrm{d}s\right) \in [0, 1]
$$
进一步改写为坐标形式为：
$$
T(\mathbf{x}_{0}, \mathbf{x}_{t}) = \exp\left(-\int_{0}^{t}\sigma_{t}(\mathbf{x} - s\boldsymbol{\omega}) \mathrm{d}s\right)
$$
透射率描述了光线的能量在传输过程中的衰减比例。从能量角度看，透射率还可以通过光线在起点和当前点的辐亮度比定义为：
$$
T(\mathbf{x}_{0}, \mathbf{x}_{t}) = \frac{L(\mathbf{x}_{0}, \boldsymbol{\omega})}{L(\mathbf{x}_{t}, \boldsymbol{\omega})}
$$

将透射率代入 Volume Rendering Equation 的解析解，得到：
$$
L(\mathbf{x}_{0}, \boldsymbol{\omega}) = T(\mathbf{x}_{0}, \mathbf{x}_{d})L(\mathbf{x}_{d}, \boldsymbol{\omega}) + \int_{0}^{d} T(\mathbf{x}_{0}, \mathbf{x}_{t})\left[\sigma_{a}(\mathbf{x}_{t})L_{e}(\mathbf{x}_{t}, \boldsymbol{\omega}) + \sigma_{s}(\mathbf{x}_{t})L_{s}(\mathbf{x}_{t}, \boldsymbol{\omega})\right] \mathrm{d}t
$$

### Monte Carlo Integration

由于上述解析解的计算量较大，通常采用 Monte Carlo 方法进行近似计算。对应的 Monte Carlo 估计器为：
$$
\langle L(\mathbf{x}_{0}, \boldsymbol{\omega}) \rangle = \underbrace{\frac{T(\mathbf{x}_{0}, \mathbf{x}_{d})}{p(d)}}_{\text{todo}}L(\mathbf{x}_{d}, \boldsymbol{\omega}) + \frac{T(\mathbf{x}_{0}, \mathbf{x}_{t})}{p(t)} \left[\sigma_{a}(\mathbf{x}_{t})L_{e}(\mathbf{x}_{t}, \boldsymbol{\omega}) + \sigma_{s}(\mathbf{x}_{t})L_{s}(\mathbf{x}_{t}, \boldsymbol{\omega})\right]
$$

## Distance Sampling

在 Monte Carlo 积分中，PDF 的选择对计算效率至关重要。

### Free-path Sampling

一种选取方式是选择归一化的透射率作为 PDF。透射率从统计的角度表示光线能量的衰减变化，从微观概率角度则描述了光子在特定距离内未发生碰撞的概率。设随机变量 $X$ 表示光线的自由程，其概率分布满足：
$$
P(X > t) = 1 - P(X \le t) = T(t)
$$
$X$ 的 CDF 为：
$$
F(X = t) = 1 - T(t)
$$
$X$ 的 PDF 为：
$$
P(X = t) = \frac{\mathrm{d}F(X = t)}{\mathrm{d}t} = - \frac{\mathrm{d}T(t)}{\mathrm{d}t} = \sigma_{t}(t)\exp\left(-\int_{0}^{t}\sigma_{t}(\mathbf{x}_{s}) \mathrm{d}s\right) = \sigma_{t}(t)T(t)
$$
带入去除了边界项对 Monte Carlo 积分器可得：
$$
\langle L(\mathbf{x}_{0}, \boldsymbol{\omega}) \rangle = \frac{1}{\sigma_{t}(t)} [\sigma_{a}(\mathbf{x}_{t})L_{e}(\mathbf{x}_{t}, \boldsymbol{\omega}) + \sigma_{s}(\mathbf{x}_{t})L_{s}(\mathbf{x}_{t}, \boldsymbol{\omega})]
$$

### Closed-from Tracking (Infinite)

对于只存在吸收和外散射，且消光系数 $\sigma_{t}(\mathbf{x}) \equiv c$ 为常数的场景，透射率可以简化为解析形式：
$$
T(t) = \exp\left(-ct\right)
$$
CDF 为：
$$
F(X = t) = 1 - \exp\left(-ct\right)
$$
PDF 为：
$$
p(X = t) = c\exp\left(-ct\right)
$$
利用逆变换采样法，可以直接得到自由程的采样距离：
$$
t = -\frac{\ln (1 - \xi)}{c} = -\frac{\ln \xi}{c}
$$
对于这种情况，我们还可以得到自由程的均值：
$$
\mathbb{E}[X] = \int_{0}^{\infty} t c \exp\left(-ct\right) \mathrm{d}t = \frac{1}{c}
$$

---

消光系数为 $c\exp(-at)$ 的场景，透射率为：
$$
T(t) = \exp\left(-\int_{0}^{d} c\exp(-as)\mathrm{d}s\right) = \exp\left(-\frac{c}{a} (1 - \exp(-at))\right)
$$
CDF 为：
$$
F(X = t) = 1 - \exp\left(-\frac{c}{a} (1 - \exp(-at))\right)
$$
PDF 为：
$$
p(X = t) = c\exp(-at)\exp\left(-\frac{c}{a} (1 - \exp(-at))\right)
$$
逆采样为：
$$
t = -\frac{1}{a} \ln\left(1 + \frac{a}{c} \ln \xi\right)
$$
均值为：
$$
\begin{aligned}
\mathbb{E}[X] &= \int_{0}^{\infty} t c\exp(-at)\exp\left(-\frac{c}{a} (1 - \exp(-at))\right) \mathrm{d}t \\
&= -\frac{c}{a^{2}}\exp\left(-\frac{c}{a}\right) \int_{0}^{1} \ln u \exp\left(\frac{c}{a} u\right) \mathrm{d}u & u = \exp(-at)
\end{aligned}
$$
积分积不出来。

### Closed-from Tracking (Finite)

上面分析时，没有限制自由程的最大距离。如果做了限制，那么 PDF 为：
$$
p(X = t) = c\exp\left(-ct\right)\mathbb{I}_{[0, d]}(t)
$$

### Regular Tracking


