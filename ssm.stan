//functions{
//    real new_normal_lpdf(vector y, vector loc, real scale);
//}
// make STANCFLAGS=--allow-undefined USER_HEADER=./stan/lib/stan_math/stan/math/prim/prob/new_normal_lpdf.hpp models/ssm
// make STANCFLAGS=--allow-undefined USER_HEADER=~/git/cmdstan/stan/lib/stan_math/stan/math/prim/prob/new_normal_lpdf.hpp models/ssm
// make STANCFLAGS=--allow-undefined models/ssm
// make models/ssm

// to build using new compiled stanc3
// ./_build/default/src/stanc/stanc.exe ssm.stan

data {
    int<lower=0> N;
    vector[N] u;
    vector[N] y;
}

parameters {
    real<lower=0.,upper=1.> a;
    real b;
    vector[N] x;
    real<lower=1e-8> q;
    real<lower=1e-8> r;
}

transformed parameters {
    vector[N] mu = a * x + b * u;
}

model {
    // priors
    q ~ cauchy(0., 1.);
    r ~ cauchy(0., 1.);

    // likelihoods
//    x[2:N] ~ normal(mu[1:N-1], q);
    x[2:N] ~ new_normal(mu[1:N-1], q);
//    target += new_normal_lpdf(x[2:N] | mu[1:N-1], q);
    y ~ normal(x, r);

}

generated quantities {
    vector[N] yhat = x + normal_rng(0, r);
}