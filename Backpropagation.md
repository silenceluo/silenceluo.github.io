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



# [Back Propagation of MMA](#sec:conv2d)

The 2D convolution for a single batch (N=1) can be defined as following:

$$
O(n, p, q, k) 
  = \sum^{R-1}_{r=0} \sum^{S-1}_{s=0} \sum^{C-1}_{c=0} A(n, h, w, c) * W(k, r, s, c)
$$
The size of OA is decided by the input size and convolution parameters:

$$
\left\{
\begin{aligned}
P &= \floor{ \frac{H + 2*pad_H - d_H *(R-1) - 1}{stride_H} } +1  \\
Q &= \floor{ \frac{W + 2*pad_W - d_W *(S-1) - 1}{stride_W} } +1
\end{aligned}
\right.
$$

Given the OA coordinate $(p, q)$ and the Kernel coordinate $(r, s)$, we can get the corresponding IA coordinate:

$$
\left\{
\begin{aligned}
h &= p * stride_H - pad_H + r*d_H  \\
w &= q * stride_W - pad_W + s*d_W  
\end{aligned}
\right.
$$

$$
\left\{
\begin{aligned}
p &= \frac{h + pad_H - r * d_H}{stride_H}  \\
q &= \frac{w + pad_W - s * d_W}{stride_W}
\end{aligned}
\right.
$$


Firstly, let us ignore N and C to simplify the work.

## [Stride=1, Dilation=1](#sec:conv_S1D1)

To simplify the work, we assume that stride and dilation parameters are all 1 in all direction, And let us omit $K$ and $C$ direction first. Then the sie of OA can he simplified as:

$$
\left\{
\begin{aligned}
P &= \floor{ \frac{H + 2*pad H - d_H * (R-1) - l}{stride H}} + 1 = H + 2*pad_H - R + 1 \\
Q &= \floor{ \frac{W + 2*pad W - d_W * (S-l) - l}{stride w}} + 1 = W + 2*pad_W - S + 1
\end{aligned} { }
\right.
$$

Given $(h, w)$ and $(r, s)$, $(p, g)$ can be derived as:

$$
\label{eq:pq_s1d1}
\left\{
\begin{aligned}
p &= h + pad_H - r \\
q &= w + pad_W - s
\end{aligned} {}
\right .
$$


### [BPW](#sec:conv_S1D1_bpw_)

Let us ignore the $K$ and $c$ direction first, to simplify the derivation of BPW and BPA

$$
\begin{aligned}
d W(r,s) &= \frac{\delta L}{\delta W(r,s)}   \\
         &= \sum^{P-1}_{p=0} \sum^{Q-1}_{q=0} \frac{\delta L(p, q)} {\delta O(p, q)} * \frac{\delta O(p, q)} {\delta W(r, s)} \\ 
         &= \sum^{P-1}_{p=0} \sum^{Q-1}_{q=0} O'(p, q) * \frac{\delta O(p, q)}{\delta W(r, s)}
\end{aligned}{ }
$$

#### BPW as Convolution of $A$ and $O'$
As

$$
O(p,q) = \sum^{R-l}_{r=0} \sum^{S-1}_{s=0} A(h,w)* W(r,s)
$$

Then

$$
\frac{\delta O(p, q)}{\delta W(r, s)} = A(h, w) = A(p-pad_H+r, q-pad_W+s)
$$


Then $dW(r,s)$ can be denoted as:

$$\label{eg:dw_s1d1}
\begin{aligned}
d W(r, s) &= \sum^{P-1}_{p=0} \sum^{Q-1}_{q=0} O'(p, q) * \frac{\delta O(p, q)}{\delta W(r, s)} \\
          &= \sum^{P-1}_{p=0} \sum^{Q-1}_{q=0} O'(p, q) * A(p-pad_H+r, q-pad_W+s)  \\
          &= \sum^{P-l}_{p=0} \sum^{Q-1}_{q=0} A(r-pad_H+p, s-pad_W+q) * O'(p, q)
\end{aligned}{}
$$

Put (\ref{eq:pq_s1d1}) into the original convolution (\ref{eg:conv2d_pkg}), the original convolution can be denoted as:

$$
O(p,q) = \sum^{R-l}_{r=0}  \sum^{S-1}_{s=0} A(p - pad_H + r, q - pad_W + s) * W(r, s)
$$

Compare (\ref{eg:dw_s1d1}) and (\ref{eq:conv2d_sldl}), we can see that $d W$ is a convolution of $A$ and $O'$, and the padding size is still $(pad_H, pad_W)$.

\paragraph{BPW as Convolution of $O'$ and $A$}

$$
\begin{aligned}
d W(r, s) &= \sum^{P-1}_{p=0} \sum^{Q-1}_{q=0} O'(p, q) * \frac{\delta O(p, q)}{\delta W(r, s)} \\
          &= \sum^{P-1}_{p=0} \sum^{Q-1}_{q=0} O'(h+pad_H-r, w+pad_W-s) * A(h, w)  \\
          &= \sum^{P-1}_{p=0} \sum^{Q-1}_{q=0} O'(h-(R-1-pad_H)+(R-1-r), w-(S-1-pad_W)+(S-1-s)) * A(h, w)
\end{aligned}{}
$$

And it will generate 
$$\begin{aligned}
d W(R-1-r', S-1-s') &= \sum^{P-1}_{p=0} \sum^{Q-1}_{q=0} O'(h-(R-1-pad_H)+r', w-(S-1-pad_W)+s') * A(h, w)
\end{aligned}{}
$$
in which

$$\left\{
\begin{aligned}
r' = R-1-r  \\
s' = S-1-s
\end{aligned} {}
\right.
$$

In this view of BPW, it is a convolution of $O'$ and $A$, with padding size $(R-1-pad_H, S-1-pad_W)$. The generated $W(R-1-r', S-1-s')$ should be rotated \dag{180} to generate $W$.

## [BPA]

The BPA of Conv2D is as following:

$$
\begin{aligned}
d A(h,w) &= \frac{\delta L}{\delta A(h, w)}  \\
         &= \sum^{P-1}_{p=0} \sum^{Q-1}_{q=0} \frac{\delta L(p, q)}{\delta O(p, q)} * \frac{\delta O(p, q)}{\delta A(h, w)}   \\
         &= \sum^{P-1}_{p=0} \sum^{Q-1}_{q=0} O'(p, q) * \frac{\delta O(p, q)}{\delta A(h, w)}     \\
         &= \sum^{ }_{p}     \sum^{}_{q}      O'(p, q) * W(r, s)
\end{aligned}{}
$$


We can denote BPA as following:
$$
\begin{aligned}
d A(h,w) &= \sum^{ }_{p} \sum^{}_{q} O'(p, q) * W(r, s)    \\
         &= \sum^{ }_{r} \sum^{}_{s} O'(h + pad_H - r, w + pad_W - s) * W(r, s) \\
         &= \sum^{ }_{r} \sum^{}_{s} O'(h - (R-1-pad_H) + (R-1-r), w - (S-1-pad_W) + (S-1-s)) * W(r, s) \\
         &= \sum^{ }_{r'} \sum^{}_{s'} O'(h - (R-1-pad_H) + r', w - (S-1-pad_W) + s') * W(R-1-r', S-1-s')
\end{aligned}{}
$$

In which $r'$ and $s'$ are defined as below:

$$
\left\{
\begin{aligned}
r' = R-1-r  \\
s' = S-1-s
\end{aligned} {}
\right.
$$

From the above equation, we can see that BPA is a convolution of $O'$ and $W'=W(R-1-r, S-1-s)$, in which $W'$ is a \ang{180} rotated $W$. The padding size is $(R-1-pad_H, S-1-pad_W)$.

## [Dilation=1, Stride$>$1]

\subsubsection{BPW}

$$
\begin{aligned}
d W(r,s) &= \frac{\delta L}{\delta W(r, s)}   \\
         &= \sum^{P-1}_{p=0} \sum^{Q-1}_{q=0} O'(p, q) * A(h, w)  \\
         &= \sum^{ }_{p}     \sum^{}_{q}      O'(p, q) * A(p*stride_H-pad_H+r, q*stride_W-pad_W+s)  \\
         &= \sum^{ }_{p}     \sum^{}_{q}      A(p*stride_H-pad_H+r, q*stride_W-pad_W+s) * O'(p, q)
\end{aligned} {}
$$

Thus it is a stride convolution with dilation value $(stride_H, stride_W)$.

\subsubsection{BPA}

As
\begin{equation}
\left\{
\begin{aligned}
h &= p * stride_H - pad_H + r*d_H  \\
w &= q * stride_W - pad_W + s*d_W  
\end{aligned}
\right.
\end{equation}

we have:
\begin{equation}
\begin{aligned}
d A(h, w) &= \sum_{p} \sum_{q} O'(p, q) * W(r, s)  \\
          &= \sum_{r} \sum_{s} O'(\frac{h + pad_H - r}{stride_H}, \frac{w + pad_W - s}{stride_W}) * W(r, s) \\
          &= \sum_{r} \sum_{s} O'(\frac{h - (R-1-pad_H) + (R-1-r)}{stride_H}, \frac{w - (S-1-pad_w) + (S-1-s)}{stride_W}) * W(r, s) \\
          &= \sum_{r'} \sum_{s'} O'(\frac{h - (R-1-pad_H) + r'}{stride_H}, \frac{w - (S-1-pad_w) + s')}{stride_W}) * W(R-1-r', S-1-s')
\end{aligned} {}
\end{equation}

with 

\begin{equation}
\left\{
\begin{aligned}
r' = R-1-r  \\
s' = S-1-s
\end{aligned} {}
\right.
\end{equation}

Thus for BPA with Dilation=1 and Stride$>$1:
\begin{enumerate}
\item Same as BPA with Stride$=$1, Weight data should be \ang{180} rotated.
\item As the coordinate in $O'$ is divided by the stride value, it behaves as
  \begin{itemize}
  \item Firstly pad the $O'$ with padding value $(R-1-pad_H, S-1-pad_W)$.
  \item Then dilate the padded $O'$ by inserting $stride_H$ zeros between every element.
  \end{itemize}
\end{enumerate}


\subsection{Dilation $>$ 1, Stride $>$ 1}

\subsubsection{BPW}
\begin{equation}
\begin{aligned}
d W(r,s) &= \frac{\delta L}{\delta W(r, s)}   \\
         &= \sum^{P-1}_{p=0} \sum^{Q-1}_{q=0} O'(p, q) * A(h, w)  \\
         &= \sum^{ }_{p}     \sum^{}_{q}      O'(p, q) * A(p*stride_H-pad_H+r*d_H, q*stride_W-pad_W+s*d_W)  \\
         &= \sum^{ }_{p}     \sum^{}_{q}      A(p*stride_H-pad_H+r*d_H, q*stride_W-pad_W+s*d_W) * O'(p, q)
\end{aligned} {}
\end{equation}

Thus it is a dilated convolution with $(dilate_H=stride_H, dilate_W=stride_W)$ and $(stride_H=d_H, stride_W=d_W)$.
 
\subsubsection{BPA}

\begin{equation}
d A(h, w) = \sum_{p} \sum_{q} O'(p, q) * W(r, s) 
\end{equation}{ }

\begin{equation}
\left\{
\begin{aligned}
h &= p * stride_H - pad_H + r*d_H  \\
w &= q * stride_W - pad_W + s*d_W  
\end{aligned}
\right.
\end{equation}

\begin{equation}
\begin{aligned}
d A(h, w) &= \sum_{p} \sum_{q} O'( \frac{h+pad_H-r*d_H}{stride_H}, \frac{w+pad_W-s*d_W}{stride_W} ) * W(r, s)  \\
          &= \sum_{r} \sum_{s} O'( \frac{h-((R-1)*d_H-pad_H)+(R-1-r)*d_H}{stride_H}, \frac{w-((S-1)*d_W-pad_W)+(S-1-s)*d_W}{stride_W} ) * W(r, s)  \\
          &= \sum_{r'} \sum_{s'} O'( \frac{h-((R-1)*d_H-pad_H)+r'*d_H}{stride_H}, \frac{w-((S-1)*d_W-pad_W)+s'*d_W}{stride_W} ) * W(R-1-r', S-1-s')
\end{aligned} {}
\end{equation}

with 

\begin{equation}
\left\{
\begin{aligned}
r' = R-1-r  \\
s' = S-1-s
\end{aligned} {}
\right.
\end{equation}

Thus for BPA with Dilation$>$1 and Stride$>$1:
\begin{enumerate}
\item Same as BPA with Stride$=$1, Weight data should be \ang{180} rotated.
\item With Dilation$>$1, Weight data should be dilated.
\item As the coordinate in $O'$ is divided by the stride value, it behaves as
  \begin{itemize}
  \item Firstly pad the $O'$ with padding value $((R-1)*d_H-pad_H, (S-1)*d_W-pad_W)$.
  \item Then dilate the padded $O'$ by inserting $stride_H$ zeros between every element.
  \end{itemize}
\end{enumerate}


\iffalse
\begin{equation}
d A(h, w) = \sum_{p} \sum_{q} O'(p, q) * W(\frac{h-p*stride_H+pad_H}{d_H}, \frac{w-q*stride_W+pad_W}{d_W})  
\end{equation}{ }

For the $(stride_H, stride_W)$ in FPROP, it will dilate the Weight data in BPA. For the $(d_H, d_W)$ in FPROP, it will dilate the $O'$ data in BPA.
\fi

\subsection{Take $N$, $K$ and $C$ into Consideration}

\begin{equation}
\begin{aligned}
O(n, p, q, k) 
  = \sum^{R-1}_{r=0} \sum^{S-1}_{s=0} \sum^{C-1}_{c=0} A(n, h, w, c) * W(k, r, s, c)
\end{aligned}
\end{equation}

\subsubsection{BPW}

\begin{equation}
\begin{aligned}
d W(k,r,s,c) &= \frac{\delta L}{\delta W(k, r, s, c)}  \\
             &= \sum^{N-1}_{n=0} \sum^{P-1}_{p=0} \sum^{Q-1}_{q=0} \frac{\delta L(n,p,q,k)}{\delta O(n, p, q, k)} * \frac{\delta O(n,p,q,k)}{\delta W(k, r, s, c)} \\
             &= \sum^{N-1}_{n=0} \sum^{P-1}_{p=0} \sum^{Q-1}_{q=0} O'(n, p, q, k) * \frac{\delta O(n,p,q,k)}{\delta W(k, r, s, c)} \\
\end{aligned} {}
\end{equation}

\begin{equation}
\begin{aligned}
d W(k,r,s,c) = \sum^{N-1}_{n=0} \sum^{P-1}_{p=0} \sum^{Q-1}_{q=0} O'(n, p, q, k) * A(n, h, w, c)
\end{aligned} {}
\end{equation}

\begin{equation}
\left\{
\begin{aligned}
h &= p * stride_H - pad_H + r*d_H  \\
w &= q * stride_W - pad_W + s*d_W  
\end{aligned}
\right.
\end{equation}

There are two ways to map BPW to MMA operation.

\paragraph{The first view of BPW as Original Convolution}

The first way to implement BPW is shown in (\ref{eq:bpw_reorg_AO}).
\begin{equation}
\label{eq:bpw_reorg_AO}
\begin{aligned}
d W(c,r,s,k) = \sum^{N-1}_{n=0} \sum^{P-1}_{p=0} \sum^{Q-1}_{q=0} A(c, h, w, n) *O'(k, p, q, n) 
\end{aligned} {}
\end{equation}

We can view (\ref{eq:bpw_reorg_AO}) as the original convolution by mapping the coordinates.

\paragraph{The Second view of BPW as Original Convolution}

The second way to implement BPW is shown in (\ref{eq:bpw_reorg}).
\begin{equation}
\label{eq:bpw_reorg}
\begin{aligned}
d W(k,r,s,c) = \sum^{N-1}_{n=0} \sum^{P-1}_{p=0} \sum^{Q-1}_{q=0} O'(k, p, q, n) * A(c, h, w, n)
\end{aligned} {}
\end{equation}

We can view (\ref{eq:bpw_reorg}) as the original convolution by mapping the coordinates.


\subsubsection{BPA}

\begin{equation}
\begin{aligned}
d A(n, h, w, c)  &= \frac{\delta L}{\delta A(n, h, w, c)}  \\
                 &= \sum_{k} \sum_{p} \sum_{q} \frac{\delta L(n,p,q,k)}{\delta O(n, p, q, k)} * \frac{\delta O(n,p,q,k)}{\delta A(n, h, w, c)} \\
                 &= \sum_{k} \sum_{p} \sum_{q} O'(n, p, q, k) * \frac{\delta O(n,p,q,k)}{\delta A(n, h, w, c)} \\
                 &= \sum_{k} \sum_{r} \sum_{s} O'(n, p, q, k) * W(k, r, s, c) \\
                 &= \sum_{k} \sum_{r} \sum_{s} O'(n, \frac{h + pad_H - r * d_H}{stride_H}, \frac{w + pad_W - s * d_W}{stride_W}, k) * W(k, r, s, c) 
\end{aligned} {}
\end{equation}
