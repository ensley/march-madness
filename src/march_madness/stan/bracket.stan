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
    real mu_scoring;
    sum_to_zero_vector[T] mu_off_centered;
    sum_to_zero_vector[T] mu_def_centered;
    vector[T] mu_home;
}

transformed parameters {
    vector[T] mu_off = mu_off_centered + mu_scoring / 2;
    vector[T] mu_def = mu_def_centered - mu_scoring / 2;
}

generated quantities {
    array[R,N] int bracket = simulate_bracket_rng(tid, mu_off, mu_def, mu_home, sigma);
}
