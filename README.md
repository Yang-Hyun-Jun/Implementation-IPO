# Implementation: CDDT

My implementation code of IPO algorithm
[***IPO: Interior-point Policy Optimization under Constraints***](https://arxiv.org/pdf/1910.09615.pdf)

# Overview

- IPO stands for Interior Point Method, which is a mathematical optimization methodology used to solve Constrained Policy Optimization.
-   When a constraint is violated, a strong penalty is applied to the loss through the log-barrier function. If not, the policy gradient is calculated and updated in a direction that lowers the constraint value.
$$L^{IPO}(\theta)=L^{CLIP}(\theta)+\frac{\log(\alpha-\hat{J}^{\pi_{\theta}}_{C_{i}})}{t}$$
-  This code applies the IPO to a Constrained Portfolio Optimization problem.


