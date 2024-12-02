# Bernstein Polynomial

### Notations and Symbols

这些是与 Bernstein 多项式相关的符号。

| Symbol | Description |
| :----: | :---------: |
| $n \in \mathbb{N}$ | Order |
| $B_{i}^{n}: \mathbb{R} \mapsto \mathbb{R}$ | Bernstein Polynomial |

## Resources

1. [Wikipedia](https://en.wikipedia.org/wiki/Bernstein_polynomial)

## Bernstein Polynomial

定义 Bernstein 多项式 $B_{i}^{n}: \mathbb{R} \mapsto \mathbb{R}$ 为：
$$
B_{i}^{n}(t) = \binom{n}{i} t^{i} (1 - t)^{n - i}
$$
其中 $i \in \{0, \dots, n\}$，$t \in [0, 1]$。

### Properties

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

#### Basis

$\{B_{i}^{n}: i \in \{0, \dots, n\}\}$ 是 $n$ 次多项式空间的基函数。

> 我们首先证明 $B_{0}^{N}, \dots, B_{n}^{n}$ 是一个线性无关集合。假设
> $$
> \sum_{i = 0}^{n} a_{i}B_{i}^{n}(x) = 0
> $$
> 其中 $a_{0}, a_{1}, \dots, a_{m} \in \mathbb{F}$。由于 $B_{0}^{n}(x) \propto (1 - x)^{m}$ 是集合中唯一有常数项的多项式，所以 $a_{0}$ 一定为零。对后面的多项式重复这一论证，我们会发现 $a_{1} = a_{2} = \dots = a_{m} = 0$。因此，它是一个线性无关集合。此外，这个集合的长度为 $n + 1$，因此满足作为 $n$ 次多项式空间的基的条件。

