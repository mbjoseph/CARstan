data {
  int<lower = 1> n;
  int<lower = 1> p;
  matrix[n, p] X;
  int<lower = 0> y[n];
  vector[n] log_offset;
  int W_n;                // number of adjacent region pairs
  int W1[W_n];            // first half of adjacency pairs
  int W2[W_n];            // second half of adjacency pairs
  vector[n] D_sparse;     // diagonal of D (number of neigbors for each site)
  vector[n] lambda;       // eigenvalues of invsqrtD * W * invsqrtD
}
parameters {
  vector[p] beta;
  vector[n] phi;
  real<lower = 0> tau;
  real<lower = 0, upper = 1> rho;
}
model {
  row_vector[n] phit_D; // phi' * D
  row_vector[n] phit_W; // phi' * W
  vector[n] ldet_terms;

  // From Kyle Foreman:
  // (phi' * Tau * phi) = tau * ((phi' * D * phi) - rho * (phi' * W * phi))
  phit_D = (phi .* D_sparse)';
  phit_W = rep_row_vector(0, n);
  for (i in 1:W_n) {
    phit_W[W1[i]] = phit_W[W1[i]] + phi[W2[i]];
    phit_W[W2[i]] = phit_W[W2[i]] + phi[W1[i]];
  }

  // prior for phi
  for (i in 1:n) ldet_terms[i] = log1m(rho * lambda[i]);
  target += 0.5 * n * log(tau)
          + 0.5 * sum(ldet_terms)
          - 0.5 * tau * (phit_D * phi - rho * (phit_W * phi)) ;
  
  beta ~ normal(0, 1);
  tau ~ gamma(0.5, .0005);
  y ~ poisson_log(X * beta + phi + log_offset);
}
