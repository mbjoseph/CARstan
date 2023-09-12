---
title: "Exact sparse CAR models in Stan"
author: "Max Joseph"
date: "August 20, 2016"
output: 
  html_document:
    keep_md: true
---


This document details sparse exact conditional autoregressive (CAR) models in Stan as an extension of previous work on approximate sparse CAR models in Stan. 
Sparse representations seem to give order of magnitude efficiency gains, scaling better for large spatial data sets. 

## CAR priors for spatial random effects

Conditional autoregressive (CAR) models are popular as prior distributions for spatial random effects with areal spatial data. 
If we have a random quantity $\phi = (\phi_1, \phi_2, ..., \phi_n)'$ at $n$ areal locations, the CAR model is often expressed via full conditional distributions:

$$\phi_i \mid \phi_j, j \neq i \sim N(\alpha \sum_{j = 1}^n b_{ij} \phi_j, \tau_i^{-1})$$

where $\tau_i$ is a spatially varying precision parameter, and $b_{ii} = 0$. 

By Brook's Lemma, the joint distribution of $\phi$ is then:

$$\phi \sim N(0, [D_\tau (I - \alpha B)]^{-1}).$$

If we assume the following:

- $D_\tau = \tau D$
- $D = diag(m_i)$: an $n \times n$ diagonal matrix with $m_i$ = the number of neighbors for location $i$
- $I$: an $n \times n$ identity matrix
- $\alpha$: a parameter that controls spatial dependence ($\alpha = 0$ implies spatial independence, and $\alpha = 1$ collapses to an *intrisnic conditional autoregressive* (IAR) specification)
- $B = D^{-1} W$: the scaled adjacency matrix
- $W$: the adjacency matrix ($w_{ii} = 0, w_{ij} = 1$ if $i$ is a neighbor of $j$, and $w_{ij}=0$ otherwise)

then the CAR prior specification simplifies to: 

$$\phi \sim N(0, [\tau (D - \alpha W)]^{-1}).$$

The $\alpha$ parameter ensures propriety of the joint distrbution of $\phi$ as long as $| \alpha | < 1$ (Gelfand & Vounatsou 2003).
However, $\alpha$ is often taken as 1, leading to the IAR specification which creates a singular precision matrix and an improper prior distribution.

## A Poisson specification

Suppose we have aggregated count data $y_1, y_2, ..., y_n$ at $n$ locations, and we expect that neighboring locations will have similar counts. 
With a Poisson likelihood: 

$$y_i \sim \text{Poisson}(\text{exp}(X_{i} \beta + \phi_i + \log(\text{offset}_i)))$$

where $X_i$ is a design vector (the $i^{th}$ row from a design matrix), $\beta$ is a vector of coefficients, $\phi_i$ is a spatial adjustment, and $\log(\text{offset}_i)$ accounts for differences in expected values or exposures at the spatial units (popular choices include area for physical processes, or population size for disease applications). 

If we specify a proper CAR prior for $\phi$, then we have that $\phi \sim \text{N}(0, [\tau (D - \alpha W)]^{-1})$ where $\tau (D - \alpha W)$ is the precision matrix $\Sigma^{-1}$.
A complete Bayesian specification would include priors for the remaining parameters $\alpha$, $\tau$, and $\beta$, such that our posterior distribution is: 

$$p(\phi, \beta, \alpha, \tau \mid y) \propto p(y \mid \beta, \phi) p(\phi \mid \alpha, \tau) p(\alpha) p(\tau) p(\beta)$$

## Example: Scottish lip cancer data

To demonstrate this approach we'll use the Scottish lip cancer data example (some documentation [here](https://cran.r-project.org/web/packages/CARBayesdata/CARBayesdata.pdf)). 
This data set includes observed lip cancer case counts at 56 spatial units in Scotland, with an expected number of cases to be used as an offset, and an area-specific continuous covariate that represents the proportion of the population employed in agriculture, fishing, or forestry.
The model structure is identical to the Poisson model outlined above. 


```
## Reading layer `scotland' from data source 
##   `/vsicurl/https://github.com/mbjoseph/CARstan/raw/master/data/scotland.shp' 
##   using driver `ESRI Shapefile'
## Simple feature collection with 56 features and 9 fields
## Geometry type: MULTIPOLYGON
## Dimension:     XY
## Bounding box:  xmin: 7150.759 ymin: 529557.2 xmax: 468393.4 ymax: 1218479
## Projected CRS: OSGB36 / British National Grid
```

![](README_files/figure-html/make-scotland-map-1.png)<!-- -->

Let's start by loading packages and data, specifying the number of MCMC iterations and chains.


```r
library(ggmcmc)
library(bayesplot)
library(cmdstanr)
library(posterior)
source('https://raw.githubusercontent.com/mbjoseph/CARstan/master/data/scotland_lip_cancer.RData')
```

To fit the full model, we'll pull objects loaded with our Scotland lip cancer data. 
I'll use `model.matrix` to generate a design matrix, centering and scaling the continuous covariate `x` to reduce correlation between the intercept and slope estimates. 


```r
W <- A # adjacency matrix
scaled_x <- c(scale(x))
X <- model.matrix(~scaled_x)
  
full_d <- list(n = nrow(X),         # number of observations
               p = ncol(X),         # number of coefficients
               X = X,               # design matrix
               y = O,               # observed number of cases
               log_offset = log(E), # log(expected) num. cases
               W = W)               # adjacency matrix
```

#### Stan implementation: CAR with `multi_normal_prec`

Our model statement mirrors the structure outlined above, with explicit normal and gamma priors on $\beta$ and $\tau$ respectively, and a $\text{Uniform}(0, 1)$ prior for $\alpha$. 
The prior on $\phi$ is specified via the `multi_normal_prec` function, passing in $\tau (D - \alpha W)$ as the precision matrix.


```
data {
  int<lower=1> n;
  int<lower=1> p;
  matrix[n, p] X;
  array[n] int<lower=0> y;
  vector[n] log_offset;
  matrix<lower=0, upper=1>[n, n] W;
}
transformed data {
  vector[n] zeros;
  matrix<lower=0>[n, n] D;
  {
    vector[n] W_rowsums;
    for (i in 1 : n) {
      W_rowsums[i] = sum(W[i,  : ]);
    }
    D = diag_matrix(W_rowsums);
  }
  zeros = rep_vector(0, n);
}
parameters {
  vector[p] beta;
  vector[n] phi;
  real<lower=0> tau;
  real<lower=0, upper=1> alpha;
}
model {
  phi ~ multi_normal_prec(zeros, tau * (D - alpha * W));
  beta ~ normal(0, 1);
  tau ~ gamma(2, 2);
  y ~ poisson_log(X * beta + phi + log_offset);
}
```

Fitting the model with `rstan`:


```r
# Define MCMC parameters 
niter <- 1E4   # definitely overkill, but good for comparison
nchains <- 4

mod <- cmdstan_model('stan/car_prec.stan')

full_fit <- mod$sample(
  data = full_d,
  parallel_chains = nchains,
  iter_sampling = niter, 
  show_messages = FALSE, 
  show_exceptions = FALSE
)

print(full_fit$summary(c('beta', 'tau', 'alpha', 'lp__')))
```

```
## # A tibble: 5 × 10
##   variable     mean   median     sd    mad      q5     q95  rhat ess_bulk
##   <chr>       <num>    <num>  <num>  <num>   <num>   <num> <num>    <num>
## 1 beta[1]   -0.0156  -0.0103 0.285  0.232   -0.467   0.422  1.01    1026.
## 2 beta[2]    0.270    0.271  0.0942 0.0931   0.114   0.423  1.00    7961.
## 3 tau        1.66     1.59   0.508  0.476    0.950   2.58   1.00    9844.
## 4 alpha      0.934    0.952  0.0617 0.0409   0.818   0.993  1.00    3813.
## 5 lp__     821.     821.     6.70   6.68   809.    831.     1.00    8610.
## # ℹ 1 more variable: ess_tail <num>
```

```r
# visualize results 
to_plot <- c('beta', 'tau', 'alpha', 'phi[1]', 'phi[2]', 'phi[3]', 'lp__')
mcmc_trace(full_fit$draws(to_plot))
```

![](README_files/figure-html/fit-prec-model-1.png)<!-- -->

### A more efficient sparse representation

Although we could specify our multivariate normal prior for $\phi$ directly in Stan via `multi_normal_prec`, as we did above, in this case we will accrue computational efficiency gains by manually specifying $p(\phi \mid \tau, \alpha)$ directly via the log probability accumulator. 
The log probability of $\phi$ is: 

$$\log(p(\phi \mid \tau, \alpha)) = - \frac{n}{2} \log(2 \pi) + \frac{1}{2} \log(\text{det}( \Sigma^{-1})) - \frac{1}{2} \phi^T \Sigma^{-1} \phi$$

In Stan, we only need the log posterior up to an additive constant so we can drop the first term. 
Then, substituting  $\tau (D - \alpha W)$ for $\Sigma^{-1}$:

$$\frac{1}{2} \log(\text{det}(\tau (D - \alpha W))) - \frac{1}{2} \phi^T \Sigma^{-1} \phi$$

$$ = \frac{1}{2} \log(\tau ^ n \text{det}(D - \alpha W)) - \frac{1}{2} \phi^T \Sigma^{-1} \phi$$

$$ = \frac{n}{2} \log(\tau) + \frac{1}{2} \log(\text{det}(D - \alpha W)) - \frac{1}{2} \phi^T \Sigma^{-1} \phi$$

There are two ways that we can accrue computational efficiency gains: 

1. Sparse representations of $\Sigma^{-1}$ to expedite computation of $\phi^T \Sigma^{-1} \phi$ (this work was done by Kyle foreman previously, e.g., https://groups.google.com/d/topic/stan-users/M7T7EIlyhoo/discussion). 

2. Efficient computation of the determinant. Jin, Carlin, and Banerjee (2005) show that:

$$\text{det}(D - \alpha W) \propto \prod_{i = 1}^n (1 - \alpha \lambda_i)$$

where $\lambda_1, ..., \lambda_n$ are the eigenvalues of $D^{-\frac{1}{2}} W D^{-\frac{1}{2}}$, which can be computed ahead of time and passed in as data. 
Because we only need the log posterior up to an additive constant, we can use this result which is proportional up to some multiplicative constant $c$: 

$$\frac{n}{2} \log(\tau) + \frac{1}{2} \log(c \prod_{i = 1}^n (1 - \alpha \lambda_i)) - \frac{1}{2} \phi^T \Sigma^{-1} \phi$$

$$= \frac{n}{2} \log(\tau) + \frac{1}{2} \log(c) +  \frac{1}{2} \log(\prod_{i = 1}^n (1 - \alpha \lambda_i)) - \frac{1}{2} \phi^T \Sigma^{-1} \phi$$

Again dropping additive constants: 

$$\frac{n}{2} \log(\tau) + \frac{1}{2} \log(\prod_{i = 1}^n (1 - \alpha \lambda_i)) - \frac{1}{2} \phi^T \Sigma^{-1} \phi$$

$$= \frac{n}{2} \log(\tau) + \frac{1}{2} \sum_{i = 1}^n \log(1 - \alpha \lambda_i) - \frac{1}{2} \phi^T \Sigma^{-1} \phi$$

### Stan implementation: sparse CAR

In the Stan model statement's `transformed data` block, we compute $\lambda_1, ..., \lambda_n$ (the eigenvalues of $D^{-\frac{1}{2}} W D^{-\frac{1}{2}}$), and generate a sparse representation for W (`Wsparse`), which is assumed to be symmetric, such that the adjacency relationships can be represented in a two column matrix where each row is an adjacency relationship between two sites. 

The Stan model statement for the sparse implementation never constructs the precision matrix, and does not call any of the `multi_normal*` functions. 
Instead, we use define a `sparse_car_lpdf()` function and use it in the model block. 


```
functions {
  /**
  * Return the log probability of a proper conditional autoregressive (CAR) prior 
  * with a sparse representation for the adjacency matrix
  *
  * @param phi Vector containing the parameters with a CAR prior
  * @param tau Precision parameter for the CAR prior (real)
  * @param alpha Dependence (usually spatial) parameter for the CAR prior (real)
  * @param W_sparse Sparse representation of adjacency matrix (int array)
  * @param n Length of phi (int)
  * @param W_n Number of adjacent pairs (int)
  * @param D_sparse Number of neighbors for each location (vector)
  * @param lambda Eigenvalues of D^{-1/2}*W*D^{-1/2} (vector)
  *
  * @return Log probability density of CAR prior up to additive constant
  */
  real sparse_car_lpdf(vector phi, real tau, real alpha,
                       array[,] int W_sparse, vector D_sparse, vector lambda,
                       int n, int W_n) {
    row_vector[n] phit_D; // phi' * D
    row_vector[n] phit_W; // phi' * W
    vector[n] ldet_terms;
    
    phit_D = (phi .* D_sparse)';
    phit_W = rep_row_vector(0, n);
    for (i in 1 : W_n) {
      phit_W[W_sparse[i, 1]] = phit_W[W_sparse[i, 1]] + phi[W_sparse[i, 2]];
      phit_W[W_sparse[i, 2]] = phit_W[W_sparse[i, 2]] + phi[W_sparse[i, 1]];
    }
    
    for (i in 1 : n) {
      ldet_terms[i] = log1m(alpha * lambda[i]);
    }
    return 0.5
           * (n * log(tau) + sum(ldet_terms)
              - tau * (phit_D * phi - alpha * (phit_W * phi)));
  }
}
data {
  int<lower=1> n;
  int<lower=1> p;
  matrix[n, p] X;
  array[n] int<lower=0> y;
  vector[n] log_offset;
  matrix<lower=0, upper=1>[n, n] W; // adjacency matrix
  int W_n; // number of adjacent region pairs
}
transformed data {
  array[W_n, 2] int W_sparse; // adjacency pairs
  vector[n] D_sparse; // diagonal of D (number of neigbors for each site)
  vector[n] lambda; // eigenvalues of invsqrtD * W * invsqrtD
  
  {
    // generate sparse representation for W
    int counter;
    counter = 1;
    // loop over upper triangular part of W to identify neighbor pairs
    for (i in 1 : (n - 1)) {
      for (j in (i + 1) : n) {
        if (W[i, j] == 1) {
          W_sparse[counter, 1] = i;
          W_sparse[counter, 2] = j;
          counter = counter + 1;
        }
      }
    }
  }
  for (i in 1 : n) {
    D_sparse[i] = sum(W[i]);
  }
  {
    vector[n] invsqrtD;
    for (i in 1 : n) {
      invsqrtD[i] = 1 / sqrt(D_sparse[i]);
    }
    lambda = eigenvalues_sym(quad_form(W, diag_matrix(invsqrtD)));
  }
}
parameters {
  vector[p] beta;
  vector[n] phi;
  real<lower=0> tau;
  real<lower=0, upper=1> alpha;
}
model {
  phi ~ sparse_car(tau, alpha, W_sparse, D_sparse, lambda, n, W_n);
  beta ~ normal(0, 1);
  tau ~ gamma(2, 2);
  y ~ poisson_log(X * beta + phi + log_offset);
}
```

Fitting the model:


```r
sp_d <- list(n = nrow(X),         # number of observations
             p = ncol(X),         # number of coefficients
             X = X,               # design matrix
             y = O,               # observed number of cases
             log_offset = log(E), # log(expected) num. cases
             W_n = sum(W) / 2,    # number of neighbor pairs
             W = W)               # adjacency matrix

sp_fit <- cmdstan_model('stan/car_sparse.stan')$sample(
  data = sp_d,
  parallel_chains = nchains,
  iter_sampling = niter, 
  show_messages = FALSE, 
  show_exceptions = FALSE
)


print(sp_fit$summary(c('beta', 'tau', 'alpha', 'lp__')))
```

```
## # A tibble: 5 × 10
##   variable     mean   median     sd    mad      q5     q95  rhat ess_bulk
##   <chr>       <num>    <num>  <num>  <num>   <num>   <num> <num>    <num>
## 1 beta[1]   -0.0117  -0.0134 0.263  0.228   -0.436   0.422  1.00     964.
## 2 beta[2]    0.272    0.273  0.0944 0.0936   0.117   0.426  1.00    9052.
## 3 tau        1.64     1.58   0.498  0.471    0.952   2.55   1.00   11371.
## 4 alpha      0.933    0.950  0.0625 0.0418   0.814   0.992  1.00    5830.
## 5 lp__     783.     783.     6.64   6.55   772.    793.     1.00    9875.
## # ℹ 1 more variable: ess_tail <num>
```

```r
mcmc_trace(sp_fit$draws(to_plot))
```

![](README_files/figure-html/fit-sparse-model-1.png)<!-- -->

### MCMC Efficiency comparison
 
The main quantity of interest is the effective number of samples per unit time. 
Sparsity gives us an order of magnitude or so gains, mostly via reductions in run time. 


|Model  | Number of effective samples| Elapsed time (sec)| Effective samples / sec)|
|:------|---------------------------:|------------------:|------------------------:|
|full   |                    8533.779|           29.82268|                 286.1506|
|sparse |                    9869.593|            2.10133|                4696.8318|

### Posterior distribution comparison

Let's compare the estimates to make sure that we get the same answer with both approaches. 
In this case, I've used more MCMC iterations than we would typically need in to get a better estimate of the tails of each marginal posterior distribution so that we can compare the 95% credible intervals among the two approaches. 

![](README_files/figure-html/compare-parameter-estimates-1.png)<!-- -->


The two approaches give the same answers (more or less, with small differences arising due to MCMC sampling error). 

## Postscript: sparse IAR specification

2023 update: for a much more comprehensive treatment of IAR models, see Morris, Mitzi, Katherine Wheeler-Martin, Dan Simpson, Stephen J. Mooney, Andrew Gelman, and Charles DiMaggio. "Bayesian hierarchical spatial models: Implementing the Besag York Mollié model in stan." Spatial and spatio-temporal epidemiology 31 (2019): 100301. https://doi.org/10.1016/j.sste.2019.100301

Although the IAR prior for $\phi$ that results from $\alpha = 1$ is improper, it remains popular (Besag, York, and Mollie, 1991). 
In practice, these models are typically fit with a sum to zero constraints: $\sum_{i\text{ in connected coponent}} \phi_i = 0$ for each connected component of the graph. This allows us to interpret both the overall mean and the component-wise means.

With $\alpha$ fixed to one, we have: 

$$\log(p(\phi \mid \tau)) = - \frac{n}{2} \log(2 \pi) + \frac{1}{2} \log(\text{det}^*(\tau (D - W))) - \frac{1}{2} \phi^T \tau (D - W) \phi$$

$$ = - \frac{n}{2} \log(2 \pi) + \frac{1}{2} \log(\tau^{n-k} \text{det}^*(D - W)) - \frac{1}{2} \phi^T \tau (D - W) \phi$$

$$ = - \frac{n}{2} \log(2 \pi) + \frac{1}{2} \log(\tau^{n-k}) + \frac{1}{2} \log(\text{det}^*(D - W)) - \frac{1}{2} \phi^T \tau (D - W) \phi$$

Here $\text{det}^*(A)$ is the generalized determinant of the square matrix $A$ defined as the product of its non-zero eigenvalues, and $k$ is the number of connected components in the graph. For the Scottish Lip Cancer data, there is only one connected component and $k=1$.  The reason that we need to use the generalized determinant is that the precision matrix is, by definition, singular in intrinsic models as the support of the Gaussian distribution is on a subspace with fewer than $n$ dimensions.  For the classical ICAR(1) model, we know that the directions correpsonding to the zero eigenvalues are exactly the vectors that are constant on each connected component of the graph and hence $k$ is the number of connected components.


Dropping additive constants, the quantity to increment becomes: 

$$ \frac{1}{2} \log(\tau^{n-k}) - \frac{1}{2} \phi^T \tau (D - W) \phi$$

And the corresponding Stan syntax would be:


```
functions {
  /**
  * Return the log probability of a proper intrinsic autoregressive (IAR) prior 
  * with a sparse representation for the adjacency matrix
  *
  * @param phi Vector containing the parameters with a IAR prior
  * @param tau Precision parameter for the IAR prior (real)
  * @param W_sparse Sparse representation of adjacency matrix (int array)
  * @param n Length of phi (int)
  * @param W_n Number of adjacent pairs (int)
  * @param D_sparse Number of neighbors for each location (vector)
  * @param lambda Eigenvalues of D^{-1/2}*W*D^{-1/2} (vector)
  *
  * @return Log probability density of IAR prior up to additive constant
  */
  real sparse_iar_lpdf(vector phi, real tau, array[,] int W_sparse,
                       vector D_sparse, vector lambda, int n, int W_n) {
    row_vector[n] phit_D; // phi' * D
    row_vector[n] phit_W; // phi' * W
    vector[n] ldet_terms;
    
    phit_D = (phi .* D_sparse)';
    phit_W = rep_row_vector(0, n);
    for (i in 1 : W_n) {
      phit_W[W_sparse[i, 1]] = phit_W[W_sparse[i, 1]] + phi[W_sparse[i, 2]];
      phit_W[W_sparse[i, 2]] = phit_W[W_sparse[i, 2]] + phi[W_sparse[i, 1]];
    }
    
    return 0.5 * ((n - 1) * log(tau) - tau * (phit_D * phi - (phit_W * phi)));
  }
}
data {
  int<lower=1> n;
  int<lower=1> p;
  matrix[n, p] X;
  array[n] int<lower=0> y;
  vector[n] log_offset;
  matrix<lower=0, upper=1>[n, n] W; // adjacency matrix
  int W_n; // number of adjacent region pairs
}
transformed data {
  array[W_n, 2] int W_sparse; // adjacency pairs
  vector[n] D_sparse; // diagonal of D (number of neigbors for each site)
  vector[n] lambda; // eigenvalues of invsqrtD * W * invsqrtD
  
  {
    // generate sparse representation for W
    int counter;
    counter = 1;
    // loop over upper triangular part of W to identify neighbor pairs
    for (i in 1 : (n - 1)) {
      for (j in (i + 1) : n) {
        if (W[i, j] == 1) {
          W_sparse[counter, 1] = i;
          W_sparse[counter, 2] = j;
          counter = counter + 1;
        }
      }
    }
  }
  for (i in 1 : n) {
    D_sparse[i] = sum(W[i]);
  }
  {
    vector[n] invsqrtD;
    for (i in 1 : n) {
      invsqrtD[i] = 1 / sqrt(D_sparse[i]);
    }
    lambda = eigenvalues_sym(quad_form(W, diag_matrix(invsqrtD)));
  }
}
parameters {
  vector[p] beta;
  vector[n] phi_unscaled;
  real<lower=0> tau;
}
transformed parameters {
  vector[n] phi; // brute force centering
  phi = phi_unscaled - mean(phi_unscaled);
}
model {
  phi_unscaled ~ sparse_iar(tau, W_sparse, D_sparse, lambda, n, W_n);
  beta ~ normal(0, 1);
  tau ~ gamma(2, 2);
  y ~ poisson_log(X * beta + phi + log_offset);
}
```

## References

Besag, Julian, Jeremy York, and Annie Mollié. "Bayesian image restoration, with two applications in spatial statistics." Annals of the institute of statistical mathematics 43.1 (1991): 1-20.

Gelfand, Alan E., and Penelope Vounatsou. "Proper multivariate conditional autoregressive models for spatial data analysis." Biostatistics 4.1 (2003): 11-15.

Jin, Xiaoping, Bradley P. Carlin, and Sudipto Banerjee. "Generalized hierarchical multivariate CAR models for areal data." Biometrics 61.4 (2005): 950-961.
