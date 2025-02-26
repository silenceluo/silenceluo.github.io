# [Back Propagation of MMA](#sec:mma_bp)

The forward propagation of MMA can be denoted as:

$$
\begin{equation}
\begin{aligned}
Y(m, n) = \sum^{K-1}_{k=0} X(m, k) * W(k, n)
\end{aligned}
\end{equation}
$$

## [BPA of MMA](#sec:mma_bpa)

The gradient of Input Activation can be defined as:

$$
\begin{aligned}
d X(m, k) &= \frac{\delta L}{\delta X(m, k)} \\
          &= \sum_{n=0}^{N-1} \frac{dL}{dY(m,n)} * \frac{dY(m,n)}{d X(m, k)}  \\
          &= \sum_{n=0}^{N-1} Y'(m,n) * \frac{dY(m,n)}{d X(m, k)} \\
          &= \sum_{n=0}^{N-1} Y'(m,n) * W(k, n)
\end{aligned}
$$


Thus Back Propagation of Activation (BPA) can be denoted as:

$$
X' = Y' * W^T
$$

## [BPW of MMA](#sec:mma_bpw)

The gradient of Weight can be defined as:

$$
\begin{aligned}
d W(k, n) 
  &= \frac{\delta L}{\delta W(k, n)} \\
  &= \sum_{m=0}^{M-1} \frac{dL}{dY(m,n)} * \frac{dY(m,n)}{d W(k, n)}  \\
  &= \sum_{m=0}^{M-1} Y'(m,n) * \frac{dY(m,n)}{d W(k, n)}  \\
  &= \sum_{m=0}^{M-1} Y'(m,n) * X(m, k)
\end{aligned}
$$

Thus Back Propagation of Weight (BPW) can be denoted as:

$$
W' = X^T * Y'  
$$