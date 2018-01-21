import argparse
import pathlib

import numpy as np
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.ensemble import VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from fitmodel.fitmodel import do_grid_search
from gridSearch.gridSearch import GridSearchCustomModel
from processing.processing import create_dataframe, drop_column, join_dfs, drop_original_values, \
    apply_macd, get_random_list, \
    apply_bollinger_band, drop_columns, apply_ichimo, apply_stochastic, apply_diff_on, \
    create_target_ahead, AHEAD, apply_momentum, apply_rsi
from reporting.reporting import CustomReport
from utility.utility import get_len_dfs

TARGET_VARIABLE = "Close__diff"
PERCENTAGE_CHANGE = 0.04
crossList = []

if __name__ == "__main__":

    # Getting path where CSVs are stored
    parser = argparse.ArgumentParser("Ml Forex")
    parser.add_argument("--datapath", help="complete path to CSV file directory")
    parser.add_argument("--target", help="target variable")
    parser.add_argument("--train_len", help="train len")

    args = parser.parse_args()

    notification_file = "/home/edge7/Desktop/notification_file_" + args.target
    data_file = "/home/edge7/Desktop/price_data_" + args.target
    result_file = "/home/edge7/Desktop/result_file_" + args.target
    while True:
        TARGET_VARIABLE = args.target + "_Body_in_pips"

        flist = [p for p in pathlib.Path(args.datapath).iterdir() if p.is_file()]
        crossList.append(args.target)
        # Creating n dataframe where n is the number of files
        dfs = create_dataframe(flist, "Gmt time", crossList)

        # Dropping volume columns
        dfs = drop_column(dfs, "Volume")
        dfs = drop_column(dfs, "index")
        dfs_len = get_len_dfs(dfs)

        # Inner join on GMT time
        df = join_dfs(dfs, "Gmt time")

        df = apply_ichimo(df, args.target)

        df = apply_stochastic(df, args.target)

        df = apply_rsi(df, args.target)
        # Applying MAC
        df = apply_macd(df, 26, 12)

        df = apply_bollinger_band(df, "Close_" + args.target, window=50)

        df = apply_diff_on(df, ["Volume_" + args.target])
        df = apply_momentum(df, "Close_" + args.target, window=5)

        df = create_target_ahead(df, "Close_" + args.target, AHEAD, PERCENTAGE_CHANGE)

        df['target_in_pips'] = df["Close_" + args.target].diff().shift(-1)
        df = drop_original_values(df, crossList)
        df = drop_columns(df, ["Close_" + args.target + "_diff", "Open_" + args.target + "_diff",
                               "Low_" + args.target + "_diff",
                               "High_" + args.target + "_diff"])
        df_prediction = df.copy()
        df = df.replace([np.inf, -np.inf], np.nan)
        df = df.dropna(how='any')
        target_in_pips = df['target_in_pips']
        t = df['target']
        gmt = df['Gmt time']
        df = drop_column([df], "Gmt time")[0]
        # df = drop_column([df], "diff")[0]
        df = drop_column([df], 'target_in_pips')[0]

        # Preparing reporting object
        report = CustomReport(args.datapath, df, TARGET_VARIABLE, args.train_len, args.predict, target_in_pips.cumsum(),
                              gmt, args.target)
        report.init()

        total_length = df.shape[0]
        start = 0
        train_len = int(args.train_len)
        test_len = 1

        old_best_models = None
        df = df.tail(args.train_len + 1)
        train_set_live = df.head(args.train_len)
        to_predict = df.tail(1)

        X_train = train_set_live.ix[:, train_set_live.columns != 'target']
        y_train = train_set_live.ix[:, train_set_live.columns == 'target']

        X_test = to_predict.ix[:, to_predict.columns != 'target']
        y_test = to_predict.ix[:, to_predict.columns == 'target']

        # Scaling X
        sc = StandardScaler()
        X_train = sc.fit_transform(X_train)
        X_test = sc.transform(X_test)

        param_grid_log_reg = {'C': 2.0 ** np.arange(-4, 8)}
        gdLog = GridSearchCustomModel(LogisticRegression(penalty='l1', max_iter=2000, random_state=42),
                                      param_grid_log_reg)

        param_grid_rf = {'n_estimators': [15, 30, 50, 100, 120], 'max_depth': [4, 5, 7, 12, 15]
                         }
        gdRf = GridSearchCustomModel(RandomForestClassifier(n_jobs=-1, random_state=42), param_grid_rf)

        param_grid_svm = [
            {'C': 10.0 ** np.arange(-3, 4), 'kernel': ['linear']}
            # {'C': 10.0 ** np.arange(-3, 4), 'gamma': 8.5 ** np.arange(-3, 3), 'kernel': ['rbf']
            # }
        ]
        gdSVM = GridSearchCustomModel(SVC(probability=True, random_state=42), param_grid_svm)

        param_grid_GB = {'learning_rate': [0.1, 0.2, 0.5, 0.05], 'n_estimators': [10, 20, 50, 100],
                         'max_depth': [3, 5, 7, 10, 15]
                         }

        gdGB = GridSearchCustomModel(GradientBoostingClassifier(random_state=42, max_features='auto'), param_grid_GB)

        param_grid_ANN = {"hidden_layer_sizes": [(20, 20), (15,), (8, 3), (5, 5), (10, 10), (40, 40), (10, 5),
                                                 (15, 13), (5, 10), (5, 5, 5), (10, 10, 10)],
                          'activation': ['tanh', 'relu'],
                          'alpha': 10.0 ** -np.arange(1, 7)}

        gdANN = GridSearchCustomModel(MLPClassifier(solver='lbfgs', random_state=42, verbose=False, max_iter=12000),
                                      param_grid_ANN)

        best_models = do_grid_search([gdLog, gdRf, gdGB], X_train, y_train.values.ravel(), report, old_best_models)

        # for model in best_models:
        #   report.write_score(model, X_train, y_train, X_test, y_test)
        #  res = model.predict(X_test)
        #  report.write_result_in_pips_single_model(res.tolist(),
        #                                          gmt[start + train_len: start + train_len + test_len].tolist(),
        #                                         target_in_pips[
        #                                        start + train_len: start + train_len + test_len].tolist(),
        #                                       model.best_estimator_)

        #y_test_pred, voting_classifier = report.write_combined_results(best_models, X_train, y_train, X_test, y_test)
        #report.write_result_in_pips(y_test_pred.tolist(),
        #                            gmt[start + train_len: start + train_len + test_len].tolist(),
        #                            target_in_pips[start + train_len: start + train_len + test_len].tolist())

        report.file_descriptor.write("\n\n\n *** FINAL PREDICTION ***")
        m = [(str(model.best_estimator_), model.best_estimator_) for model in best_models]
        voting_classifier = VotingClassifier(estimators=m, voting='soft', n_jobs=-1)
        voting_classifier.fit(X_train, y_train.values.ravel())
        best_models.append(voting_classifier)
        voting_classifier.predict(X_test)

    report.close()
    print('end')
