#include functions.stan

data {
    int N;
    int T;
    array[2,N] int tid;
    array[2,N] int H;
}

parameters {
    real<lower=0> sigma;
    vector[T] mu_off;
    vector[T] mu_def;
    vector[T] mu_home;
}

generated quantities {
    array[2] vector[N] mu_games = map_mu(mu_off, mu_def, mu_home, tid, H, 0.0);
    array[N] int<lower=0, upper=1> win;
    for (n in 1:N) {
        win[n] = normal_rng(mu_games[1,n], sigma) > normal_rng(mu_games[2,n], sigma);
    }
}
