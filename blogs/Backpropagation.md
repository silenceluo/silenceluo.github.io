- [1. Back Propagation of MMA](#1-back-propagation-of-mma)
  - [1.1. BPA of MMA](#11-bpa-of-mma)
  - [1.2. BPW of MMA](#12-bpw-of-mma)
- [2. Back Propagation of Convolution](#2-back-propagation-of-convolution)
  - [2.1. Stride=1, Dilation=1](#21-stride1-dilation1)
    - [2.1.1. BPW](#211-bpw)
      - [2.1.1.1. BPW as Convolution of $A$ and $O'$](#2111-bpw-as-convolution-of-a-and-o)
      - [2.1.1.2. BPW as Convolution of $O'$ and $A$](#2112-bpw-as-convolution-of-o-and-a)
    - [2.1.2. BPA](#212-bpa)
  - [2.2. Dilation=1, Stride\>1](#22-dilation1-stride1)
    - [2.2.1. BPW](#221-bpw)
    - [2.2.2. BPA](#222-bpa)
  - [2.3. Dilation \> 1, Stride \> 1](#23-dilation--1-stride--1)
    - [2.3.1. BPW](#231-bpw)
    - [2.3.2. BPA](#232-bpa)
  - [2.4. Take $N$, $K$ and $C$ into Consideration](#24-take-n-k-and-c-into-consideration)
    - [2.4.1. BPW](#241-bpw)
      - [2.4.1.1. BPW as Convolution of $A$ and $O'$](#2411-bpw-as-convolution-of-a-and-o)
      - [2.4.1.2. BPW as Convolution of $O'$ and $A$](#2412-bpw-as-convolution-of-o-and-a)
    - [2.4.2. BPA](#242-bpa)


# 1. [Back Propagation of MMA](#sec:mma_bp)

The forward propagation of MMA can be denoted as:

$$
\begin{aligned}
Y(m, n) = \sum_{k=0}^{K-1} X(m, k) * W(k, n) \\
\end{aligned}
\tag{1}
$$

## 1.1. [BPA of MMA](#sec:mma_bpa)

The gradient of Input Activation can be defined as:

$$
\begin{aligned}
d X(m, k) &= \frac{\delta L}{\delta X(m, k)} \\
          &= \sum_{n=0}^{N-1} \frac{dL}{dY(m,n)} * \frac{dY(m,n)}{d X(m, k)}  \\
          &= \sum_{n=0}^{N-1} Y'(m,n) * \frac{dY(m,n)}{d X(m, k)} \\
          &= \sum_{n=0}^{N-1} Y'(m,n) * W(k, n)
\end{aligned}
\label{eq:dX}
$$

Thus Back Propagation of Activation (BPA) can be denoted as:

$$
X' = Y' * W^T
$$

## 1.2. [BPW of MMA](#sec:mma_bpw)

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



# 2. [Back Propagation of Convolution](#sec:conv2d)

The 2D convolution for a single batch (N=1) can be defined as following:

$$
O(n, p, q, k) = \sum_{r=0}^{R-1} \sum_{s=0}^{S-1} \sum_{c=0}^{C-1} A(n, h, w, c) * W(k, r, s, c)
$$

The size of OA is decided by the input size and convolution parameters:

$$
\begin{aligned}
P &= \lfloor \frac{H + 2 * pad_H - d_H * (R-1) - 1}{stride_H} \rfloor + 1 \\
Q &= \lfloor \frac{W + 2 * pad_W - d_W * (S-1) - 1}{stride_W} \rfloor + 1
\end{aligned}
$$

Given the OA coordinate $(p, q)$ and the Kernel coordinate $(r, s)$, we can get the corresponding IA coordinate:


$$
\begin{aligned}
h &= p * stride_H - pad_H + r * d_H \\
w &= q * stride_W - pad_W + s * d_W  
\end{aligned}
$$

$$
\begin{aligned}
p &= \frac{h + pad_H - r * d_H}{stride_H}  \\
q &= \frac{w + pad_W - s * d_W}{stride_W}
\end{aligned}
$$


Firstly, let us ignore N and C to simplify the work.

## 2.1. [Stride=1, Dilation=1](#sec:conv_S1D1)

To simplify the work, we assume that stride and dilation parameters are all 1 in all direction, And let us omit $K$ and $C$ direction first. Then the sie of OA can he simplified as:

$$
\begin{aligned}
P &= \lfloor \frac{H + 2 * pad H - d_H * (R-1) - l}{stride H} \rfloor + 1 = H + 2 * pad_H - R + 1 \\
Q &= \lfloor \frac{W + 2 * pad W - d_W * (S-l) - l}{stride w} \rfloor + 1 = W + 2 * pad_W - S + 1
\end{aligned}
$$


Given $(h, w)$ and $(r, s)$, $(p, g)$ can be derived as:

$$
\begin{aligned}
p &= h + pad_H - r \\
q &= w + pad_W - s
\end{aligned} {}
$$


### 2.1.1. [BPW](#sec:conv_S1D1_bpw_)

Let us ignore the $K$ and $c$ direction first, to simplify the derivation of BPW and BPA

$$
\begin{aligned}
d W(r,s) &= \frac{\delta L}{\delta W(r,s)}   \\
         &= \sum_{p=0}^{P-1} \sum_{q=0}^{Q-1} \frac{\delta L(p, q)} {\delta O(p, q)} * \frac{\delta O(p, q)} {\delta W(r, s)} \\ 
         &= \sum_{p=0}^{P-1} \sum_{q=0}^{Q-1} O'(p, q) * \frac{\delta O(p, q)}{\delta W(r, s)}
\end{aligned}{ }
$$

#### 2.1.1.1. BPW as Convolution of $A$ and $O'$
As

$$
O(p,q) = \sum_{r=0}^{R-l} \sum_{s=0}^{S-1} A(h,w)* W(r,s)
$$

Then

$$
\frac{\delta O(p, q)}{\delta W(r, s)} = A(h, w) = A(p-pad_H+r, q-pad_W+s)
$$


Then $dW(r,s)$ can be denoted as:

$$
\begin{aligned}
d W(r, s) &= \sum_{p=0}^{P-1} \sum_{q=0}^{Q-1} O'(p, q) * \frac{\delta O(p, q)}{\delta W(r, s)} \\
          &= \sum_{p=0}^{P-1} \sum_{q=0}^{Q-1} O'(p, q) * A(p-pad_H+r, q-pad_W+s)  \\
          &= \sum_{p=0}^{P-l} \sum_{q=0}^{Q-1} A(r-pad_H+p, s-pad_W+q) * O'(p, q)
\end{aligned}{}
$$

Put (\ref{eq:pq_s1d1}) into the original convolution (\ref{eg:conv2d_pkg}), the original convolution can be denoted as:

$$
O(p,q) = \sum_{r=0}^{R-l}  \sum_{s=0}^{S-1} A(p - pad_H + r, q - pad_W + s) * W(r, s)
$$

Compare (\ref{eg:dw_s1d1}) and (\ref{eq:conv2d_sldl}), we can see that $d W$ is a convolution of $A$ and $O'$, and the padding size is still $(pad_H, pad_W)$.

#### 2.1.1.2. BPW as Convolution of $O'$ and $A$

$$
\begin{aligned}
d W(r, s) &= \sum_{p=0}^{P-1} \sum_{q=0}^{Q-1} O'(p, q) * \frac{\delta O(p, q)}{\delta W(r, s)} \\
          &= \sum_{p=0}^{P-1} \sum_{q=0}^{Q-1} O'(h+pad_H-r, w+pad_W-s) * A(h, w)  \\
          &= \sum_{p=0}^{P-1} \sum_{q=0}^{Q-1} O'(h-(R-1-pad_H)+(R-1-r), w-(S-1-pad_W)+(S-1-s)) * A(h, w)
\end{aligned}{}
$$

And it will generate 

$$
d W(R-1-r', S-1-s') = \sum_{p=0}^{P-1} \sum_{q=0}^{Q-1} O'(h - (R-1-pad_H) + r', w - (S - 1 - pad_W) + s') * A(h, w)
$$

in which

$$
\begin{aligned}
r' &= R-1-r  \\
s' &= S-1-s
\end{aligned} {}
$$

In this view of BPW, it is a convolution of $O'$ and $A$, with padding size $(R-1-pad_H, S-1-pad_W)$. The generated $W(R-1-r', S-1-s')$ should be rotated \dag{180} to generate $W$.

### 2.1.2. BPA

The BPA of Conv2D is as following:

$$
\begin{aligned}
d A(h,w) &= \frac{\delta L}{\delta A(h, w)}  \\
         &= \sum_{p=0}^{P-1} \sum_{q=0}^{Q-1} \frac{\delta L(p, q)}{\delta O(p, q)} * \frac{\delta O(p, q)}{\delta A(h, w)}   \\
         &= \sum_{p=0}^{P-1} \sum_{q=0}^{Q-1} O'(p, q) * \frac{\delta O(p, q)}{\delta A(h, w)}     \\
         &= \sum_{p} \sum_{q} O'(p, q) * W(r, s)
\end{aligned}{}
$$

We can denote BPA as following:

$$
\begin{aligned}
d A(h,w) &= \sum_{p}^{ } \sum_{q} O'(p, q) * W(r, s)    \\
         &= \sum_{r}^{ } \sum_{s} O'(h + pad_H - r, w + pad_W - s) * W(r, s) \\
         &= \sum_{r}^{ } \sum_{s} O'(h - (R-1-pad_H) + (R-1-r), w - (S-1-pad_W) + (S-1-s)) * W(r, s) \\
         &= \sum_{r'}^{ } \sum_{s'} O'(h - (R-1-pad_H) + r', w - (S-1-pad_W) + s') * W(R-1-r', S-1-s')
\end{aligned}{}
$$

In which $r'$ and $s'$ are defined as below:

$$
\begin{aligned}
r' &= R-1-r  \\
s' &= S-1-s
\end{aligned} {}
$$

From the above equation, we can see that BPA is a convolution of $O'$ and $W'=W(R-1-r, S-1-s)$, in which $W'$ is a 180 degree rotated $W$. The padding size is $(R-1-pad_H, S-1-pad_W)$.

## 2.2. Dilation=1, Stride>1

### 2.2.1. BPW

$$
\begin{aligned}
d W(r,s) &= \frac{\delta L}{\delta W(r, s)}   \\
         &= \sum_{p=0}^{P-1} \sum_{q=0}^{Q-1} O'(p, q) * A(h, w)  \\
         &= \sum_{p} \sum_{q} O'(p, q) * A(p * stride_H - pad_H + r, q * stride_W - pad_W + s)  \\
         &= \sum_{p} \sum_{q} A(p * stride_H - pad_H + r, q * stride_W - pad_W + s) * O'(p, q)
\end{aligned}{}
$$

Thus it is a stride convolution with dilation value $(stride_H, stride_W)$.

### 2.2.2. BPA

As

$$
\begin{aligned}
h &= p * stride_H - pad_H + r * d_H  \\
w &= q * stride_W - pad_W + s * d_W  
\end{aligned}
$$

we have:

$$
\begin{aligned}
d A(h, w) &= \sum_{p} \sum_{q} O'(p, q) * W(r, s)  \\
          &= \sum_{r} \sum_{s} O'(\frac{h + pad_H - r}{stride_H}, \frac{w + pad_W - s}{stride_W}) * W(r, s) \\
          &= \sum_{r} \sum_{s} O'(\frac{h - (R-1-pad_H) + (R-1-r)}{stride_H}, \frac{w - (S-1-pad_w) + (S-1-s)}{stride_W}) * W(r, s) \\
          &= \sum_{r'} \sum_{s'} O'(\frac{h - (R-1-pad_H) + r'}{stride_H}, \frac{w - (S-1-pad_w) + s')}{stride_W}) * W(R-1-r', S-1-s')
\end{aligned} {}
$$

with 

$$
\begin{aligned}
r' = R-1-r  \\
s' = S-1-s
\end{aligned} {}
$$

Thus for BPA with Dilation=1 and Stride$>$1:

- Same as BPA with Stride$=$1, Weight data should be 180 degree rotated.
- As the coordinate in $O'$ is divided by the stride value, it behaves as
  1. Firstly pad the $O'$ with padding value $(R-1-pad_H, S-1-pad_W)$.
  2. Then dilate the padded $O'$ by inserting $stride_H$ zeros between every element.


## 2.3. Dilation > 1, Stride > 1

### 2.3.1. BPW

$$
\begin{aligned}
d W(r,s) &= \frac{\delta L}{\delta W(r, s)}   \\
         &= \sum_{p=0}^{P-1} \sum_{q=0}^{Q-1} O'(p, q) * A(h, w)  \\
         &= \sum_{p} \sum_{q} O'(p, q) * A(p * stride_H - pad_H + r * d_H, q * stride_W - pad_W + s * d_W)  \\
         &= \sum_{p} \sum_{q} A(p * stride_H - pad_H + r * d_H, q * stride_W - pad_W + s * d_W) * O'(p, q) 
\end{aligned} {}
$$

Thus it is a dilated convolution with $(dilate_H=stride_H, dilate_W=stride_W)$ and $(stride_H=d_H, stride_W=d_W)$.
 
### 2.3.2. BPA

$$
d A(h, w) = \sum_{p} \sum_{q} O'(p, q) * W(r, s) 
$$

and 

$$
\begin{aligned}
h &= p * stride_H - pad_H + r * d_H  \\
w &= q * stride_W - pad_W + s * d_W  
\end{aligned}
$$

Then,

$$
\begin{aligned}
d A(h, w) &= \sum_{p} \sum_{q} O'( \frac{h+pad_H-r*d_H}{stride_H}, \frac{w+pad_W-s * d_W}{stride_W} ) * W(r, s)  \\
          &= \sum_{r} \sum_{s} O'( \frac{h-((R-1)*d_H-pad_H)+(R-1-r) * d_H}{stride_H}, \frac{w-((S-1) * d_W-pad_W)+(S-1-s) * d_W}{stride_W} ) * W(r, s)  \\
          &= \sum_{r'} \sum_{s'} O'( \frac{h-((R-1)*d_H-pad_H)+r' * d_H}{stride_H}, \frac{w-((S-1) * d_W-pad_W)+s' * d_W}{stride_W} ) * W(R-1-r', S-1-s')
\end{aligned} {}
$$

with 

$$
\begin{aligned}
r' = R-1-r  \\
s' = S-1-s
\end{aligned} {}
$$

Thus for BPA with Dilation$>$1 and Stride$>$1:
- Same as BPA with Stride$=$1, Weight data should be 180 degree rotated.
- With Dilation$>$1, Weight data should be dilated.
- As the coordinate in $O'$ is divided by the stride value, it behaves as
  1. Firstly pad the $O'$ with padding value $((R-1)*d_H-pad_H, (S-1)*d_W-pad_W)$.
  2. Then dilate the padded $O'$ by inserting $stride_H$ zeros between every element.



## 2.4. Take $N$, $K$ and $C$ into Consideration

$$
\begin{aligned}
O(n, p, q, k) 
  = \sum_{r=0}^{R-1} \sum_{s=0}^{S-1} \sum_{c=0}^{C-1} A(n, h, w, c) * W(k, r, s, c)
\end{aligned}
$$

### 2.4.1. BPW

$$
\begin{aligned}
d W(k,r,s,c) &= \frac{\delta L}{\delta W(k, r, s, c)}  \\
             &= \sum_{n=0}^{N-1} \sum_{p=0}^{P-1} \sum_{q=0}^{Q-1} \frac{\delta L(n,p,q,k)}{\delta O(n, p, q, k)} * \frac{\delta O(n,p,q,k)}{\delta W(k, r, s, c)} \\
             &= \sum_{n=0}^{N-1} \sum_{p=0}^{P-1} \sum_{q=0}^{Q-1} O'(n, p, q, k) * \frac{\delta O(n,p,q,k)}{\delta W(k, r, s, c)} \\
\end{aligned} {}
$$

$$
\begin{aligned}
d W(k,r,s,c) = \sum_{n=0}^{N-1} \sum_{p=0}^{P-1} \sum_{q=0}^{Q-1} O'(n, p, q, k) * A(n, h, w, c)
\end{aligned} {}
$$

$$
\begin{aligned}
h &= p * stride_H - pad_H + r * d_H  \\
w &= q * stride_W - pad_W + s * d_W  
\end{aligned}
$$

There are two ways to map BPW to MMA operation.

#### 2.4.1.1. BPW as Convolution of $A$ and $O'$

The first way to implement BPW is shown in (\ref{eq:bpw_reorg_AO}).

$$
\begin{aligned}
d W(c,r,s,k) = \sum_{n=0}^{N-1} \sum_{p=0}^{P-1} \sum_{q=0}^{Q-1} A(c, h, w, n) * O'(k, p, q, n) 
\end{aligned}
$$

We can view (\ref{eq:bpw_reorg_AO}) as the original convolution by mapping the coordinates.

#### 2.4.1.2. BPW as Convolution of $O'$ and $A$

The second way to implement BPW is shown in (\ref{eq:bpw_reorg}).

$$
\begin{aligned}
d W(k,r,s,c) = \sum_{n=0}^{N-1} \sum_{p=0}^{P-1} \sum_{q=0}^{Q-1} O'(k, p, q, n) * A(c, h, w, n)
\end{aligned}
$$

We can view (\ref{eq:bpw_reorg}) as the original convolution by mapping the coordinates.


### 2.4.2. BPA

$$
\begin{aligned}
d A(n, h, w, c)  &= \frac{\delta L}{\delta A(n, h, w, c)}  \\
                 &= \sum_{k} \sum_{p} \sum_{q} \frac{\delta L(n,p,q,k)}{\delta O(n, p, q, k)} * \frac{\delta O(n,p,q,k)}{\delta A(n, h, w, c)} \\
                 &= \sum_{k} \sum_{p} \sum_{q} O'(n, p, q, k) * \frac{\delta O(n,p,q,k)}{\delta A(n, h, w, c)} \\
                 &= \sum_{k} \sum_{r} \sum_{s} O'(n, p, q, k) * W(k, r, s, c) \\
                 &= \sum_{k} \sum_{r} \sum_{s} O'(n, \frac{h + pad_H - r * d_H}{stride_H}, \frac{w + pad_W - s * d_W}{stride_W}, k) * W(k, r, s, c) 
\end{aligned} {}
$$
