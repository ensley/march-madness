#include functions.stan

data {
    int N;
    int T;
    array[N] int tid;
}

transformed data {
    int R = to_int(log2(N));
}

parameters {
    real<lower=0> sigma;
    vector[T] mu_off;
    vector[T] mu_def;
    vector[T] mu_home;
}

generated quantities {
    array[R,N] int bracket = simulate_bracket_rng(tid, mu_off, mu_def, mu_home, sigma);
}