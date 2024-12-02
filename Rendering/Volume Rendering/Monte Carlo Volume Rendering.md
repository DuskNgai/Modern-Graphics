# Monte Carlo Volume Rendering

现代的体渲染技术很大一部份基于核物理学中的辐射传输理论，甚至连 Monte Carlo 算法都是在核武器研发期间开发的。

### Notations and Symbols

这些是与光线相关的符号。

| Symbol | Description | Unit |
| :----: | :---------: | :--: |
| $\mathbf{x}_{0} \in \mathbb{R}^{3}$ | Ray Origin | $[m]$ |
| $\boldsymbol{\omega} \in \mathbb{S}^{2}$ | Ray Direction | $[sr]$ |
| $t \in \mathbb{R}$ | Ray Parameter | $[sr/m]$ |
| $\mathbf{x}_{t} = \mathbf{x}_{0} + t \boldsymbol{\omega}$ | Ray Position | $[m]$ |

这些是与辐射相关的符号。

| Symbol | Description | Unit |
| :----: | :---------: | :--: |
| $L: \mathbb{R}^{3} \times \mathbb{S}^{2} \mapsto \mathbb{R}$ | Radiance at $\mathbf{x}$ in Direction $\boldsymbol{\omega}$ | $[W / (m^{2} \cdot sr)]$ |
| $L_{e}: \mathbb{R}^{3} \times \mathbb{S}^{2} \mapsto \mathbb{R}$ | (Pseudo) Emission at $\mathbf{x}$ in Direction $\boldsymbol{\omega}$ | $[W / (m^{3} \cdot sr)]$ |

这些是与体像相关的符号。

| Symbol | Description | Unit |
| :----: | :---------: | :--: |
| $\sigma_{a}: \mathbb{R}^{3} \mapsto \mathbb{R}$ | Absorption Coefficient | $[m^{-1}]$ |
| $\sigma_{s}: \mathbb{R}^{3} \mapsto \mathbb{R}$ | Scattering Coefficient | $[m^{-1}]$ |
| $\sigma_{t}: \mathbb{R}^{3} \mapsto \mathbb{R}$ | Extinction Coefficient, $\sigma_{t} = \sigma_{a} + \sigma_{s}$ | $[m^{-1}]$ |
| $\alpha: \mathbb{R}^{3} \mapsto \mathbb{R}$ | Single Scattering Albedo, $\alpha = \sigma_{s} / \sigma_{t}$ | |
| $f_{p}: \mathbb{R}^{3} \times \mathbb{S}^{2} \times \mathbb{S}^{2} \mapsto \mathbb{R}$ | Phase Function | |

这些是与数值计算相关的符号。

| Symbol | Description |
| :----: | :---------: |
| $\langle \cdot \rangle$ | Estimator of Monte Carlo Integral |
| $\xi, \eta \sim U(0, 1)$ | Uniform Random Variable |
| $p$ | Probability Density Function |
| $F$ | Cumulative Distribution Function |

## Resources

1. [Siggraph 2018 Course](https://cs.dartmouth.edu/~wjarosz/publications/novak18monte-sig.html)

## [Volume Rendering Theory](./Volume%20Rendering%20Theory.md)

## Stochastic Geometry
