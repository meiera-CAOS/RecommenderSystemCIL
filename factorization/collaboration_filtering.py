from fancyimpute import KNN, SoftImpute
from scipy.optimize import minimize
from sklearn.impute._iterative import IterativeImputer

from factorization.util import *

'''
run with python3
##################################################################################
apply methods to data
'''

DEFAULT_DATA_FILE = os.path.join(os.path.dirname(__file__),'../data/linear_comb.csv')


def impute(linear_comb_file=DEFAULT_DATA_FILE):
    # coefficients
    split = True
    x0 = [1, 1, 1, 1]

    for iterations in range(2):

        if split:
            X_incomplete, val_data = load_data_train(split)
            positions_val = val_data[:, 0]
            ratings_val = val_data[:, 1]
        else:
            X_incomplete = load_data_train()

        # method: softImpute
        X_filled_softimpute = SoftImpute(min_value=1, max_value=5, convergence_threshold=10e-7, max_iters=1000,
                                         max_rank=400) \
            .fit_transform(X_incomplete)
        if split:
            evaluate(X_filled_softimpute, val_data, "softimpute")
            predictions_softimpute = prediction_set(positions_val, X_filled_softimpute)

        # method: iterativeImputer
        X_filled_ii = IterativeImputer(max_iter=5, min_value=1, max_value=5).fit_transform(X_incomplete)

        if split:
            evaluate(X_filled_ii, val_data, "iterativeImputer")
            predictions_it_imputer = prediction_set(positions_val, X_filled_ii)

        # method: kNN

        X_filled_knn = KNN(k=5, min_value=1, max_value=5).fit_transform(X_incomplete)

        if split:
            evaluate(X_filled_knn, val_data, "kNN")
            predictions_knn = prediction_set(positions_val, X_filled_knn)

        # uniform average of the methods: softImpute, iterativeImputer and kNN
        average = (X_filled_ii + X_filled_knn + X_filled_softimpute) / 3

        if split:
            evaluate(average, val_data, "average")
            predictions_average = prediction_set(positions_val, average)

        # linear combination f the methods: softImpute, iterativeImputer and kNN
        if split:
            def objective(x):
                x1 = x[0]
                x2 = x[1]
                x3 = x[2]
                x4 = x[3]
                return np.sqrt((np.square(
                    x1 * predictions_softimpute + x2 * predictions_knn + x3 * predictions_it_imputer + x4 * predictions_average - ratings_val)).mean())

            b = (0.0, 1.0)
            bnds = (b, b, b, b)

            for iterations in range(1):
                sol = minimize(objective, x0, method='SLSQP', bounds=bnds)

                x0 = sol.x

                linear_comb = x0[0] * X_filled_softimpute + x0[1] * X_filled_knn + x0[2] * X_filled_ii + x0[3] * average
                evaluate(linear_comb, val_data, "linear_combination")
        else:
            linear_comb = x0[0] * X_filled_softimpute + x0[1] * X_filled_knn + x0[2] * X_filled_ii + x0[3] * average

            # extract matri to .csv file to pass to NN
            matrix_to_csv(linear_comb, linear_comb_file)

        split = False


'''
##################################################################################
output from submission run:
##################################################################################

FINISHED: libraries loaded
FINISHED: Data loaded into matrix
[SoftImpute] Max Singular Value of X_init = 1940.395707
[SoftImpute] Iter 1: observed MAE=1.326920 rank=400
[SoftImpute] Iter 2: observed MAE=1.122176 rank=400
[SoftImpute] Iter 3: observed MAE=0.970744 rank=400
...
[SoftImpute] Iter 581: observed MAE=0.691605 rank=130
[SoftImpute] Iter 582: observed MAE=0.691605 rank=130
[SoftImpute] Stopped after iteration 582 for lambda=38.807914
softimpute RMSE: 0.993504
iterativeImputer RMSE: 1.064902
Imputing row 1/10000 with 979 missing, elapsed time: 760.235
Imputing row 101/10000 with 918 missing, elapsed time: 765.223
...
Imputing row 9801/10000 with 758 missing, elapsed time: 1283.869
Imputing row 9901/10000 with 883 missing, elapsed time: 1289.208
kNN RMSE: 1.187626
average RMSE: 1.020123
linear_combination RMSE: 0.983435
FINISHED: Data loaded into matrix
[SoftImpute] Max Singular Value of X_init = 2154.850381
[SoftImpute] Iter 1: observed MAE=1.332503 rank=400
[SoftImpute] Iter 2: observed MAE=1.127163 rank=400
[SoftImpute] Iter 3: observed MAE=0.978854 rank=400
...
[SoftImpute] Iter 540: observed MAE=0.711913 rank=115
[SoftImpute] Iter 541: observed MAE=0.711913 rank=115
[SoftImpute] Stopped after iteration 541 for lambda=43.097008
Imputing row 1/10000 with 977 missing, elapsed time: 801.012
Imputing row 101/10000 with 910 missing, elapsed time: 806.696
Imputing row 201/10000 with 951 missing, elapsed time: 812.299
...
Imputing row 9801/10000 with 736 missing, elapsed time: 1332.296
Imputing row 9901/10000 with 871 missing, elapsed time: 1337.634
FINISHED: Predictions read from matrix
'''
