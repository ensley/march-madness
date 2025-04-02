#include functions.stan

data {
    int N;
    int T;
    array[2,N] int Y;
    array[2,N] int tid;
    array[N] int minutes;
    array[N] int V;
}

transformed data {
    array[2,N] int H = append_array(rep_array(0, 1, N), rep_array(V, 1));
    real ppm = overall_ppm(Y, minutes);
    array[2] vector[N] X = convert_points_to_ppm(Y, minutes);
}

parameters {
    real<lower=0> sigma;
    vector[T] mu_off;
    vector[T] mu_def;
    vector[T] mu_home;
}

model {
    mu_off ~ std_normal();
    mu_def ~ std_normal();
    mu_home ~ std_normal();

    array[2] vector[N] mu_games = map_mu(mu_off, mu_def, mu_home, tid, H, ppm);
    for (t in 1:2) {
        target += normal_lpdf(X[t] | mu_games[t], sigma);
    }
}