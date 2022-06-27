# low-rank ASDL

Extension for the [Autometic Second-order Differentiation Library](https://github.com/kazukiosawa/asdfghjkl/tree/0.1) (ASDL) that enables a low-rank [Kronecker-factored Approximate Curvature](https://arxiv.org/abs/1503.05671)(K-FAC) for training deep neural networks. 

Note that the code requires CUDA to run.

# Overview
Natural Gradient Descent (NGD)[add ref.] is a second order optimization algorithm given by
$$
\theta^{t+1} \leftarrow \theta^t - \eta (\mathbf{F} + \lambda \mathbf{I})^{-1} \nabla \mathcal{L}(\theta^t)
$$
where $\theta^t$ are the paramters of step $t$,  $\eta$ the learning rate,  $F$ the Fisher information matrix, $\lambda$ the damping value for the inverse calculation and $\nabla \mathcal{L}(\theta^t)$ the loss gradient.
The advantage to first-order optimization methods such as SGD or Adam is the faster convergence in the number of steps.
However the dimension of $F$ is usually of tremendous size which makes NGD not practically usable due to the cubic time complexity of the inverse.

A popular approach to reduce the size of the Fisher is to assume that there only is correlation between adjacent layers, which reduces $F$ to a block diagonal matrix. To further reduce the dimension of the layer-wise blocks $F_l$, we can apply K-FAC, which introduces the approximation
$$
\mathbf{F}_l \approx \mathbf{A}_l \otimes \mathbf{B}_l
$$
where $\otimes$ denotes the Kronecker product. 
Despite the huge dimensinality reduction is K-FAC still much slower than first-order methods.

To further reduce the time and memory consumption can low-rank K-FAC[add reference] be applied, which approximates the Kronecker-factors as follows:
$$
    \begin{split}
        \mathbf{A}_l &\approx \text{rank}_k(\mathbf{A}_l) \\
        \mathbf{B}_l &\approx \text{rank}_k(\mathbf{B}_l - \text{diag}(\mathbf{B}_l)) + \text{diag}(\mathbf{B}_l)
    \end{split}
$$
The diagonal approach for $\mathbf{B}_l$ is a strong diagonal in those factors could be observed.

The low-rank approximation is calculated efficiently using the power-iteration and the inverse is received by recursively applying the Shermann-Morrison inverse.
