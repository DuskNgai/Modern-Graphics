# Volume Rendering Theory

体像可以看作由无数微小颗粒组成的集合。当光线进入体像时，会与这些微小颗粒发生相互作用，如吸收、散射、激发等现象。然而，在实际渲染过程中，直接以微小颗粒进行精确的光纤传输建模是不现实的，因为这将带来极其庞大的计算量。因此，我们采取折衷的方法，通过建模光线在体像中平均行为的方式，来近似描述这些复杂的交互过程。

## Notation

记号在 [Monte Carlo Volume Rendering](Monte%20Carlo%20Volume%20Rendering.md)。

## Radiance Transfer Theory

#### Meaning of $(\boldsymbol{\omega} \cdot \nabla)$

$(\boldsymbol{\omega} \cdot \nabla)L$ 表示 $L$ 在 $\boldsymbol{\omega}$ 上的方向导数，也等价于 $\nabla_{\boldsymbol{\omega}}L$ 或 $\nabla L \cdot \boldsymbol{\omega}$。根据光线参数化 $\mathbf{x}_{t} = \mathbf{x}_{0} + t\boldsymbol{\omega}$，有：
$$
(\boldsymbol{\omega} \cdot \nabla)L = \underbrace{\nabla L}_{\mathbb{R}^{3 \times 1}} \cdot \boldsymbol{\omega} = \underbrace{\frac{\partial L}{\partial \mathbf{x}_{t}}}_{\mathbb{R}^{1 \times 3}} \frac{\partial \mathbf{x}_{t}}{\partial t} = \frac{\mathrm{d}L}{\mathrm{d}t}.
$$

### Absorption

当光线与微粒发生碰撞时，光线的辐射能被微粒吸收，转化为微粒的内能。
$$
(\boldsymbol{\omega} \cdot \nabla) L(\mathbf{x}, \boldsymbol{\omega}) = - \sigma_{a}(\mathbf{x}) L(\mathbf{x}, \boldsymbol{\omega}),
$$
这个公式就是 Beer-Lambert 定律的微分形式。

### Emission

在光线传播的路径上，微粒可能会自行或者受激发射辐射能。
$$
(\boldsymbol{\omega} \cdot \nabla) L(\mathbf{x}, \boldsymbol{\omega}) = \sigma_{a}(\mathbf{x}) L_{e}(\mathbf{x}, \boldsymbol{\omega}),
$$
其中 $\sigma_{a}(\mathbf{x})L_{e}(\mathbf{x}, \boldsymbol{\omega})$ 表示微粒的发射能量。实际上这里只需要 $L_{e}(\mathbf{x}, \boldsymbol{\omega})$ 即可表示微粒的发射能量，与 $\sigma_{a}(\mathbf{x})$ 相乘是为了后面 RTE 的形式上的一致性。

### Out-Scattering

当光线与微粒发生碰撞时，光线的辐射能被微粒散射，能量辐射到其他方向上。
$$
(\boldsymbol{\omega} \cdot \nabla) L(\mathbf{x}, \boldsymbol{\omega}) = - \sigma_{s}(\mathbf{x}) L(\mathbf{x}, \boldsymbol{\omega}).
$$

### In-Scattering

在光线传播的路径上，其他方向的辐射能量会被微粒散射，转移到当前方向上。
$$
(\boldsymbol{\omega} \cdot \nabla) L(\mathbf{x}, \boldsymbol{\omega}) = \sigma_{s}(\mathbf{x}) \int_{\mathbb{S}^{2}} f_{p}(\mathbf{x}, \boldsymbol{\omega}, \boldsymbol{\omega}') L(\mathbf{x}, \boldsymbol{\omega}') \mathrm{d}\boldsymbol{\omega}' = \sigma_{s}(\mathbf{x}) L_{s}(\mathbf{x}, \boldsymbol{\omega}).
$$

### Radiance Transfer Equation

把上述所有的事件结合起来，得到 Radiance Transfer Equation (RTE)：
$$
(\boldsymbol{\omega} \cdot \nabla) L(\mathbf{x}, \boldsymbol{\omega}) = \underbrace{-\sigma_{t}(\mathbf{x})L(\mathbf{x}, \boldsymbol{\omega})}_{\text{Losses}} + \underbrace{\sigma_{a}(\mathbf{x})L_{e}(\mathbf{x}, \boldsymbol{\omega}) + \sigma_{s}(\mathbf{x})L_{s}(\mathbf{x}, \boldsymbol{\omega})}_{\text{Gains}}.
$$

## Volume Rendering Equation

Volume Rendering Equation 是 Radiance Transfer Equation 的积分形式。为了便于分析，我们首先将其改写为关于 $t$ 的微分方程：
$$
\frac{\mathrm{d}}{\mathrm{d}t}L(t) = -\sigma_{t}(\mathbf{x}_{t})L(t) + \sigma_{a}(\mathbf{x}_{t})L_{e}(t) + \sigma_{s}(\mathbf{x}_{t})L_{s}(t),
$$
光线的起点可以是光源、表面或者其他体像。设光线的起点为 $\mathbf{x}_{d}$，终点为 $\mathbf{x}_{0}$。在实际光线追踪实现中，通常定义光线的参数化方向与上述公式中的光线方向相反，因此光线沿负方向传播。将光线重新参数化为 $\mathbf{x}_{t} = \mathbf{x}_{0} - t\boldsymbol{\omega}$，代入上述公式后，求解微分方程得到：
$$
L(0) = L(d)\exp\left(-\int_{0}^{d}\sigma_{t}(\mathbf{x}_{t}) \mathrm{d}t\right) + \int_{0}^{d}\left[\sigma_{a}(\mathbf{x}_{t})L_{e}(t) + \sigma_{s}(\mathbf{x}_{t})L_{s}(t)\right] \exp\left(-\int_{0}^{t}\sigma_{t}(\mathbf{x}_{s}) \mathrm{d}s\right) \mathrm{d}t.
$$
进一步改写为坐标形式为：
$$
L(\mathbf{x}_{0}, \boldsymbol{\omega}) = L(\mathbf{x}_{d}, \boldsymbol{\omega})\exp\left(-\int_{0}^{d}\sigma_{t}(\mathbf{x}_{t})\mathrm{d}t\right) + \int_{0}^{d}\left[\sigma_{a}(\mathbf{x}_{t})L_{e}(\mathbf{x}_{t}, \boldsymbol{\omega}) + \sigma_{s}(\mathbf{x}_{t})L_{s}(\mathbf{x}_{t}, \boldsymbol{\omega})\right] \exp\left(-\int_{0}^{t}\sigma_{t}(\mathbf{x}_{s}) \mathrm{d}s\right) \mathrm{d}t.
$$

### Transmittance

在上述解析解中，光的衰减由一个指数衰减项控制。我们将其定义为透射率：
$$
T(t) = \exp\left(-\int_{0}^{t}\sigma_{t}(\mathbf{x}_{s}) \mathrm{d}s\right) \in [0, 1]
$$
透射率描述了光线的能量在传输过程中的衰减比例。从能量角度看，透射率还可以通过光线在起点和当前点的辐亮度比定义为：
$$
T(t) = \frac{L(\mathbf{x}_{0}, \boldsymbol{\omega})}{L(\mathbf{x}_{t}, \boldsymbol{\omega})}
$$

将透射率代入 Volume Rendering Equation 的解析解，得到：
$$
L(\mathbf{x}_{0}, \boldsymbol{\omega}) = T(d)L(\mathbf{x}_{d}, \boldsymbol{\omega}) + \int_{0}^{d} T(t)\left[\sigma_{a}(\mathbf{x}_{t})L_{e}(\mathbf{x}_{t}, \boldsymbol{\omega}) + \sigma_{s}(\mathbf{x}_{t})L_{s}(\mathbf{x}_{t}, \boldsymbol{\omega})\right] \mathrm{d}t.
$$
由于积分是从 $0$ 增长到 $d$，我们可以把 VRE 看作是沿着光线射出的方向不断收集辐射。

### Monte Carlo Integration

由于上述解析解的计算量较大，通常采用 Monte Carlo 方法进行近似计算。对应的 Monte Carlo 估计器为：
$$
\langle L(\mathbf{x}_{0}, \boldsymbol{\omega}) \rangle = \frac{T(d)}{p(d)} L(\mathbf{x}_{d}, \boldsymbol{\omega}) + \frac{T(t)}{p(t)} \left[\sigma_{a}(\mathbf{x}_{t})L_{e}(\mathbf{x}_{t}, \boldsymbol{\omega}) + \sigma_{s}(\mathbf{x}_{t})L_{s}(\mathbf{x}_{t}, \boldsymbol{\omega})\right]
$$

## Tracking Approach

在 Monte Carlo 积分中，PDF 的选择对计算效率至关重要。

### Free-path Sampling

一种选取方式是选择归一化的透射率作为 PDF。透射率从统计的角度表示光线能量的衰减变化，从微观概率角度则描述了光子在特定距离内未发生碰撞的概率。设随机变量 $X$ 表示光线的自由程，即与微粒发生碰撞前自由前进的距离，其概率分布满足：
$$
P(X > t) = 1 - P(X \le t) = T(t).
$$
$X$ 的 CDF 为：
$$
F(X = t) = 1 - T(t),
$$
$X$ 的 PDF 为：
$$
P(X = t) = \frac{\mathrm{d}F(X = t)}{\mathrm{d}t} = - \frac{\mathrm{d}T(t)}{\mathrm{d}t} = \sigma_{t}(t)\exp\left(-\int_{0}^{t}\sigma_{t}(\mathbf{x}_{s}) \mathrm{d}s\right) = \sigma_{t}(t)T(t).
$$
带入去除了边界项的 Monte Carlo 积分器可得：
$$
\begin{aligned}
\langle L(\mathbf{x}_{0}, \boldsymbol{\omega}) \rangle &= \frac{1}{\sigma_{t}(t)} [\sigma_{a}(\mathbf{x}_{t})L_{e}(\mathbf{x}_{t}, \boldsymbol{\omega}) + \sigma_{s}(\mathbf{x}_{t})L_{s}(\mathbf{x}_{t}, \boldsymbol{\omega})] \\
&= P_{a}(\mathbf{x}_{t})L_{e}(\mathbf{x}_{t}, \boldsymbol{\omega}) + P_{s}(\mathbf{x}_{t})L_{s}(\mathbf{x}_{t}, \boldsymbol{\omega})
\end{aligned}
$$
其中：
$$
P_{a} = \frac{\sigma_{a}}{\sigma_{t}}, \quad P_{s} = \frac{\sigma_{s}}{\sigma_{t}},
$$
显然有 $P_{a} + P_{s} = 1$，本质上定义了发射事件和内散射事件发生的比例，以及由之带来的辐射强度的混合比例。

#### Closed-from Tracking (Infinite)

对于只存在吸收和外散射，且消光系数 $\sigma_{t}(\mathbf{x}) \equiv c$ 为常数的场景，透射率可以简化为解析形式：
$$
T(t) = \exp\left(-ct\right).
$$
CDF 为：
$$
F(X = t) = 1 - \exp\left(-ct\right),
$$
PDF 为：
$$
p(X = t) = c\exp\left(-ct\right).
$$
利用逆变换采样法，可以直接得到自由程的采样距离：
$$
t = -\frac{\ln (1 - \xi)}{c} = -\frac{\ln \xi}{c}.
$$
对于这种情况，我们还可以得到自由程的均值：
$$
\mathbb{E}[X] = \int_{0}^{\infty} t c \exp\left(-ct\right) \mathrm{d}t = \frac{1}{c}.
$$

这种情况下的伪代码如下：
```python
def closed_form_tracking(x: vec3, w: vec3, d: float) -> float:
    while True:
        # Distance sampling
        t = - math.log(random.random()) / sigma_t

        # Retrieve the radiance from outside of the volume
        if t > d:
            return L(x + w * d, w)

        # Move away from camera
        x = x - w * t

        # Emission event
        if random.random() < sigma_a / sigma_t:
            return Le(x, w)
        else:
            w = phase_fn.sample()
            d =
```

<!-- TODO -->

---

消光系数为 $c\exp(-at)$ 的场景，透射率为：
$$
T(t) = \exp\left(-\int_{0}^{t} c\exp(-as)\mathrm{d}s\right) = \exp\left(-\frac{c}{a} (1 - \exp(-at))\right).
$$
CDF 为：
$$
F(X = t) = 1 - \exp\left(-\frac{c}{a} (1 - \exp(-at))\right),
$$
PDF 为：
$$
p(X = t) = c\exp(-at)\exp\left(-\frac{c}{a} (1 - \exp(-at))\right).
$$
逆采样为：
$$
t = -\frac{1}{a} \ln\left(1 + \frac{a}{c} \ln \xi\right).
$$
均值为：
$$
\begin{aligned}
\mathbb{E}[X] &= \int_{0}^{\infty} t c\exp(-at)\exp\left(-\frac{c}{a} (1 - \exp(-at))\right) \mathrm{d}t \\
&= -\frac{c}{a^{2}}\exp\left(-\frac{c}{a}\right) \int_{0}^{1} \ln u \exp\left(\frac{c}{a} u\right) \mathrm{d}u & u = \exp(-at).
\end{aligned}
$$
积分积不出来。

---

<!-- TODO -->

消光系数为 $c\exp(-(t - \mu)^{2} / \sigma^{2})$ 的场景，透射率为：
$$
T(t) = \exp\left(-\int_{0}^{t} c\exp\left(-\frac{(s - \mu)^{2}}{\sigma^{2}}\right)\mathrm{d}s\right) = \exp\left(-c \mathop{\mathrm{erf}}\left(\frac{t - \mu}{\sigma}\right)\right)
$$


### Regular Tracking

分段常数的场景可以分段采用解析解得到积分的结果。
