#include functions.stan

data {
    int N;
    int T;
    array[2,N] int tid;
    array[2,N] int H;
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
    array[2] vector[N] mu_games = map_mu(mu_off, mu_def, mu_home, tid, H, 0.0);
    array[N] int<lower=0, upper=1> win;
    for (n in 1:N) {
        win[n] = normal_rng(mu_games[1,n], sigma) > normal_rng(mu_games[2,n], sigma);
    }
}
