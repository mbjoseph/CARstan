data {
  int<lower = 1> n;
  int<lower = 1> p;
  matrix[n, p] X;
  int<lower = 0> y[n];
  vector[n] log_offset;
  matrix<lower = 0, upper = 1>[n, n] W;
  matrix<lower = 0>[n, n] D;
}
transformed data{
  vector[n] zeros;
  zeros = rep_vector(0, n);
}
parameters {
  vector[p] beta;
  vector[n] phi;
  real<lower = 0> tau;
  real<lower = 0, upper = 1> rho;
}
model {
  phi ~ multi_normal_prec(zeros, tau * (D - rho * W));
  beta ~ normal(0, 1);
  tau ~ gamma(0.5, .0005);
  y ~ poisson_log(X * beta + phi + log_offset);
}
