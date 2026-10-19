#include functions.stan

data {
    int N;
    int T;
    array[2,N] int Y;
    array[2,N] int tid;
    array[N] int minutes;
    array[2,N] int H;
}

transformed data {
    real ppm = overall_ppm(Y, minutes);
    array[2] vector[N] X = convert_points_to_ppm(Y, minutes);
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

model {
    // Each original team-effect mean has variance 1/T; their difference has variance 2/T.
    mu_scoring ~ normal(0, sqrt(2.0 / T));
    mu_off_centered ~ std_normal();
    mu_def_centered ~ std_normal();
    mu_home ~ std_normal();

    array[2] vector[N] mu_games = map_mu(mu_off, mu_def, mu_home, tid, H, ppm);
    for (t in 1:2) {
        target += normal_lpdf(X[t] | mu_games[t], sigma);
    }
}
