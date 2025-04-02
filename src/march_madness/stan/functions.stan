functions {
    array[] vector map_mu(vector mu_off, vector mu_def, vector mu_home, array[,] int tid, array[,] int H, real ppm) {
        int N = num_elements(tid[1]);
        array[2] vector[N] mu_games;
        for (i in 1:2) {
            for (n in 1:N) {
                mu_games[i,n] = ppm + mu_off[tid[i,n]] - mu_def[tid[2-i+1,n]] + mu_home[tid[i,n]] * H[i,n];
            }
        }
        return mu_games;
    }

    real overall_ppm(array[,] int Y, array[] int minutes) {
        int N = num_elements(Y[1]);
        real ppm = (sum(Y[1]) + sum(Y[2])) / (2.0 * sum(minutes));
        return ppm;
    }

    array[] vector convert_points_to_ppm(array[,] int Y, array[] int minutes) {
        int N = num_elements(Y[1]);
        array[2] vector[N] X;
        for (t in 1:2) {
            for (n in 1:N) {
                X[t,n] = Y[t,n] * 1.0 / minutes[n];
            }
        }
        return X;
    }

    array[,] int simulate_bracket_rng(array[] int tid, vector mu_off, vector mu_def, vector mu_home, real sigma) {
        int T = num_elements(tid);
        int R = to_int(log2(T)) + 1;
        array[R,T] int wid = rep_array(0, R, T);
        wid[1] = tid;
        array[R-1,T] int advance = rep_array(0, R-1, T);

        for (r in 2:R) {
            int G = to_int(pow(2, R-r));
            array[2,G] int H = rep_array(0, 2, G);
            array[2*G] int winners = wid[r-1,1:(2*G)];
            array[2,G] int gid;

            for (g in 1:(2*G)) {
                int i = ((g+1) % 2) + 1;
                int j = to_int(ceil(g/2.0));
                gid[i,j] = winners[g];
            }

            array[2] vector[G] mu_games = map_mu(mu_off, mu_def, mu_home, gid, H, 0.0);

            for (g in 1:G) {
                if (wid[r,g] == 0) {
                    wid[r,g] = normal_rng(mu_games[1,g], sigma) > normal_rng(mu_games[2,g], sigma) ? gid[1,g] : gid[2,g];
                }
                for (t in 1:T) {
                    if (wid[r,g] == tid[t]) {
                        advance[r-1,t] += 1;
                        break;
                    }
                }
            }
        }

        return advance;
    }
}