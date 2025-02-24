# Back Propagation of MMA

The forward propagation of MMA can be denoted as:

$$
\begin{equation}
\begin{aligned}
Y(m, n) = \sum^{K-1}_{k=0} X(m, k) * W(k, n)
\end{aligned}
\end{equation}
$$

## BPA of MMA #{sec_sec_MMA_BPA}

The gradient of Input Activation can be defined as:

$$
\begin{equation}
\begin{aligned}
d X(m, k) 
  &= \frac{\delta L}{\delta X(m, k)} \\
  &= \sum^{N-1}_{n=0} \frac{dL}{dY(m,n)} * \frac{dY(m,n)}{d X(m, k)}  \\
  &= \sum^{N-1}_{n=0} Y'(m,n) * \frac{dY(m,n)}{d X(m, k)}  \\
  &= \sum^{N-1}_{n=0} Y'(m,n) * W(k, n)
\end{aligned}
\end{equation}
$$

Thus Back Propagation of Activation (BPA) can be denoted as:
$$
\begin{equation}
\begin{aligned}
X' = Y' * W^T  
\end{aligned}
\end{equation}
$$

## BPW of MMA #{sec_sec_MMA_BPW}

The gradient of Weight can be defined as:

$$
\begin{equation}
\begin{aligned}
d W(k, n) 
  &= \frac{\delta L}{\delta W(k, n)} \\
  &= \sum^{M-1}_{m=0} \frac{dL}{dY(m,n)} * \frac{dY(m,n)}{d W(k, n)}  \\
  &= \sum^{M-1}_{m=0} Y'(m,n) * \frac{dY(m,n)}{d W(k, n)}  \\
  &= \sum^{M-1}_{m=0} Y'(m,n) * X(m, k)
\end{aligned}
\end{equation}
$$

Thus Back Propagation of Weight (BPW) can be denoted as:

$$
\begin{equation}
\begin{aligned}
W' = X^T * Y'  
\end{aligned}
\end{equation}
$$