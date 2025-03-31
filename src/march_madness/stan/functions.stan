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
}