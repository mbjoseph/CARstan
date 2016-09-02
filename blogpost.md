Two weird tricks for fast conditional autoregressive models in Stan
===================================================================

Conditional autoregressive (CAR) models are popular as prior distributions for spatial random effects with areal spatial data. 
Historically, MCMC algorithms for CAR models have benefitted from efficient Gibbs sampling via full conditional distributions for the spatial random effects. 
But, these conditional specifications do not work in Stan, where the joint density needs to be specified (up to a multiplicative constant).

CAR models can still be implemented in Stan by specifying a multivariate normal prior on the spatial random effects, parameterized by a mean vector and a precision matrix. 
This works, but is slow and hard to scale to large datasets. 

Order(s) of magnitude speedups can be achieved by combining 1) sparse matrix multiplications from Kyle Foreman (outlined [on the stan-users mailing list](https://groups.google.com/d/topic/stan-users/M7T7EIlyhoo/discussion)), and 2) a fancy determinant trick from Jin, Carlin, and Banerjee (2005). 
With the oft-used Scotland lip cancer dataset, the sparse CAR implementation with the NUTS (No-U-Turn Sampler) algorithm in Stan gives 120 effective samples/sec compared to 7 effective samples/sec for the precision matrix implementation.
Details for these sparse exact methods can be found at https://github.com/mbjoseph/CARstan.

(Max Joseph is part of the Earth Lab Analytics Hub, University of Colorado - Boulder.)

### References

Jin, Xiaoping, Bradley P. Carlin, and Sudipto Banerjee. "Generalized hierarchical multivariate CAR models for areal data." Biometrics 61.4 (2005): 950-961.
