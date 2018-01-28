import argparse
import os
import pathlib
import time

import numpy as np
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.ensemble import VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from feature_selection.feature_importance import FeatureSelection
from fitmodel.fitmodel import do_grid_search
from gridSearch.gridSearch import GridSearchCustomModel
from processing.processing import create_dataframe, drop_column, join_dfs, drop_original_values, \
    apply_macd, apply_bollinger_band, drop_columns, apply_ichimo, apply_stochastic, apply_diff_on, \
    create_target_ahead, AHEAD, apply_momentum, apply_rsi, apply_distance_from_max, apply_distance_from_min, \
    apply_support_and_resistance, apply_linear_regression, sup_and_res, apply_williams, apply_PROC, apply_HeikenAshi, \
    apply_CCI, applyWCP, apply_GARCH
from utility.utility import get_len_dfs

CHANGE_IN_PIPS = 0.0105
crossList = []
prefix = "C:\\Users\\Administrator\\AppData\\Roaming\\MetaQuotes\\Terminal\\1DAFD9A7C67DC84FE37EAA1FC1E5CF75\\tester\\files\\"

if __name__ == "__main__":

    # Getting path where CSVs are stored
    parser = argparse.ArgumentParser("Ml Forex")
    parser.add_argument("--datapath", help="complete path to CSV file directory")
    parser.add_argument("--target", help="target variable")
    parser.add_argument("--train_len", help="train len")

    args = parser.parse_args()

    notification_file = prefix + "notificationfile_" + args.target
    data_file = prefix + "pricefile_" + args.target
    result_file = prefix + "resultfile_" + args.target
    counter = 0
    crossList.append(args.target)
    old_best_models = None
    fs = None
    fs_counter = 0
    # Infinite loop
    while True:

        # Wait for notification file
        while not os.path.isfile(notification_file):
            time.sleep(1)

        # res = "HOLD"
        # if counter != 0:
        #      counter += 1
        #      counter %= AHEAD
        #      lines = []
        #      with open(data_file, 'r') as f:
        #          lines = f.readlines()
        #      with open(data_file, 'w') as f:
        #          f.writelines(lines[:1] + lines[2:])
        # #     with open(result_file, 'w') as f:
        # #         f.write(res)
        # #     os.remove(notification_file)
        # #     continue

        flist = [pathlib.Path(data_file)]

        # Creating n dataframe where n is the number of files
        dfs = create_dataframe(flist, "Gmt time", crossList)

        # Dropping volume columns
        dfs = drop_column(dfs, "Volume")
        dfs = drop_column(dfs, "index")
        dfs_len = get_len_dfs(dfs)

        # Inner join on GMT time
        df = join_dfs(dfs, "Gmt time")

        #df = apply_ichimo(df, args.target)


        df = apply_rsi(df, "")

        df = sup_and_res(df, args.target, window=100)
        last_close = df.tail(1)["Close_" + args.target]
        last_close = last_close.tolist()[0]
        c = df.tail(1)["Close_" + args.target].tolist()[0]
        sup = c + df.tail(1)["closest_sup"].tolist()[0]
        res = c + df.tail(1)["closest_res"].tolist()[0]
        with open(prefix + "SR" + args.target, 'w') as f:
            f.write(str(sup) + "\n" + (str(res) + "\n"))


        #df = apply_support_and_resistance(df, args.target)
        df = apply_diff_on(df, ["Volume_" + args.target])

        #           ---- Momentum ---- #
        df = apply_momentum(df, "Close_" + args.target, window=5)
        df = apply_momentum(df, "Close_" + args.target, window=4)
        df = apply_momentum(df, "Close_" + args.target, window=3)
        df = apply_momentum(df, "Close_" + args.target, window=10)
        df = apply_momentum(df, "Close_" + args.target, window=9)
        df = apply_momentum(df, "Close_" + args.target, window=8)

        #           ---- Stochastic ---- #
        df = apply_stochastic(df, args.target, window=5, mean=1)
        df = apply_stochastic(df, args.target, window=4, mean=1)
        df = apply_stochastic(df, args.target, window=3, mean=1)
        df = apply_stochastic(df, args.target, window=5, mean=2)
        df = apply_stochastic(df, args.target, window=4, mean=2)
        df = apply_stochastic(df, args.target, window=3, mean=2)
        df = apply_stochastic(df, args.target, window=10, mean=1)
        df = apply_stochastic(df, args.target, window=9, mean=1)
        df = apply_stochastic(df, args.target, window=8, mean=1)
        df = apply_stochastic(df, args.target, window=10, mean=2)
        df = apply_stochastic(df, args.target, window=9, mean=2)
        df = apply_stochastic(df, args.target, window=8, mean=2)

        #           ---- Williams ---- #
        df = apply_williams(df, args.target, window = 10)
        df = apply_williams(df, args.target, window=9)
        df = apply_williams(df, args.target, window=8)
        df = apply_williams(df, args.target, window=7)
        df = apply_williams(df, args.target, window=6)

        #           ---- Williams ---- #
        df = apply_PROC(df, args.target, window=15)
        df = apply_PROC(df, args.target, window=14)
        df = apply_PROC(df, args.target, window=13)
        df = apply_PROC(df, args.target, window=12)

        #           ---- Applying MACD ---#
        df = apply_macd(df, 26, 12)

        #           ---- Bollinger Band ---- #
        df = apply_bollinger_band(df, "Close_" + args.target, window=50)
        df = apply_bollinger_band(df, "Close_" + args.target, window=100)
        df = apply_bollinger_band(df, "Close_" + args.target, window=15)

        #           ----  Heiken Ashi
        df = apply_HeikenAshi(df, args.target)

        #           ----  CCI ------ #
        df = apply_CCI(df, args.target, window = 15)

        #           ----  WCP ------ #
        df = applyWCP(df, args.target)

        #           ---- GARCH ------ #
        df = apply_GARCH(df, args.target, window = 15)



        #df = apply_distance_from_max(df, "Close_" + args.target, window=50)
        #df = apply_distance_from_min(df, "Close_" + args.target, window=50)
        df = apply_linear_regression(df, "Close_" + args.target, window=75)

        df = create_target_ahead(df, "Close_" + args.target, AHEAD, CHANGE_IN_PIPS)

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
        with open(prefix + " " + args.target, 'a') as f:
            f.write(str(t.tolist()[-int(args.train_len):]) + "\n")
        gmt = df['Gmt time']
        df = drop_column([df], "Gmt time")[0]
        # df = drop_column([df], "diff")[0]
        df = drop_column([df], 'target_in_pips')[0]
        if fs is None:
            fs = FeatureSelection(df, prefix + "_" + args.target)
        total_length = df.shape[0]
        start = 0
        train_len = int(args.train_len)
        test_len = 1

        df = df.tail(int(args.train_len) + 1).reset_index(drop=True)
        train_set_live = df.head(int(args.train_len))
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
        try:
            best_models = do_grid_search([gdRf], X_train, y_train.values.ravel(), 0, old_best_models)
            for i in best_models:
                if "RandomForest" in str(i.best_estimator_):
                    rf = i.best_estimator_
            fs.write_feature_importance(rf)
            fs_counter += 1
        except Exception as e:
            print(e)
            best_models = old_best_models
            with open(prefix + "_" + args.target + "_LOG", 'a') as f:
                f.write(str(e))

            os.remove(notification_file)
            with open(result_file, 'w') as f:
                f.write("HOLD")
            continue

        tp = df_prediction.tail(1)
        gmt = tp['Gmt time']
        m = [(str(model.best_estimator_), model.best_estimator_) for model in best_models]
        voting_classifier = VotingClassifier(estimators=m, voting='soft', n_jobs=-1)
        voting_classifier.fit(X_train, y_train.values.ravel())
        res = voting_classifier.predict(X_test)[0]
        print("vot")
        print(res)
        print(voting_classifier.predict_proba(X_test)[0])
        # print(m[0][0])
        # print(m[0][1].predict_proba(X_test)[0])
        # print(m[1][1].predict_proba(X_test)[0])
        with open(result_file, 'w') as f:
            f.write(res)

        os.remove(notification_file)
        old_best_models = best_models

        if fs_counter % 30 == 0:
            fs_counter = 0
            fs.consolidate_feature_importance()

        counter += 1
        counter %= AHEAD

    print('end')
