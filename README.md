# Low-rank ASDL

Extended [Automatic Second-order Differentiation Library](https://github.com/kazukiosawa/asdfghjkl/tree/0.1) (ASDL) that enables low-rank [Kronecker-factored Approximate Curvature](https://arxiv.org/abs/1503.05671)(K-FAC) for training deep neural networks. 

Note that the code requires CUDA to run.

# Overview
Natural Gradient Descent (NGD) is a second order optimization algorithm which uses the Fisher information matrix as curvature information.
The advantage to first-order optimization methods such as SGD or Adam is the faster convergence in the number of steps.
However the dimension of Fisher usually is of tremendous size which makes NGD not practically usable due to the cubic time complexity of the inverse.

A popular approach to reduce the size of the Fisher is to assume that there only is correlation between adjacent layers, which reduces the Fisher to a block diagonal matrix. To further reduce the dimension of the layer-wise blocks can K-FAC be applied, which approximates the layer-wise Fisher with a Kronecker-product of factors A and B.
Despite the huge dimensinality reduction is a K-FAC iteration still much slower than first-order methods.

To further reduce the time and memory consumption can low-rank K-FAC be applied, which approximates the Kronecker-factors A and B by a low-rank and low-rank + diagonal approach respectively. The B factor uses the additional diagonal approach as a diminant diagonal could be observed.

The low-rank approximation is calculated efficiently using the power-iteration and the inverse is received by recursively applying the Shermann-Morrison inverse.
