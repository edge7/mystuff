import argparse
import os

mingw_path = 'C:\\Program Files\\mingw-w64\\x86_64-7.2.0-posix-seh-rt_v5-rev1\\mingw64\\bin'
os.environ['PATH'] = mingw_path + ';' + os.environ['PATH']

import pathlib
import time
import math
import numpy as np
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.ensemble import VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from xgboost.sklearn import XGBClassifier
from feature_selection.feature_importance import FeatureSelection
from fitmodel.fitmodel import do_grid_search
from gridSearch.gridSearch import GridSearchCustomModel
from processing.processing import create_dataframe, drop_column, join_dfs, drop_original_values, \
    apply_macd, apply_bollinger_band, drop_columns, apply_ichimo, apply_stochastic, create_target_ahead, AHEAD, \
    apply_momentum, apply_rsi, apply_distance_from_max, apply_distance_from_min, \
    apply_support_and_resistance, apply_linear_regression, sup_and_res, apply_williams, apply_PROC, apply_HeikenAshi, \
    apply_CCI, applyWCP, applyFourier, apply_diff, apply_poly, apply_linear_regression_on_min, \
    apply_linear_regression_on_max, apply_df_test, create_dataframe_for_stacking, apply_fourier_on_min, \
    apply_fourier_on_max, apply_poly_on_max, apply_poly_on_min, apply_shift
from utility.utility import get_len_dfs

CHANGE_IN_PIPS = 0.0000
last_pred = "OUT"
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
    best_score_file = prefix + "best_score_" + args.target
    counter = 0
    crossList.append(args.target)
    old_best_models = None
    old_best_models_ = None

    RETRAIN = 0
    fs = None
    fs_counter = 0
    # Infinite loop
    while True:

        # Wait for notification file
        while not os.path.isfile(notification_file):
            time.sleep(1)

        res = "HOLD"
        if counter != 0:
            counter += 1
            counter %= round(AHEAD)
            RETRAIN += 1
            lines = []
            with open(data_file, 'r') as f:
                lines = f.readlines()
            with open(data_file, 'w') as f:
                f.writelines(lines[:1] + lines[2:])
            with open(result_file, 'w') as f:
                f.write(res)
            os.remove(notification_file)
            continue

        flist = [pathlib.Path(data_file)]

        # Creating n dataframe where n is the number of files
        dfs = create_dataframe(flist, "Gmt time", crossList)

        # Dropping volume columns
        dfs = drop_column(dfs, "Volume")
        dfs = drop_column(dfs, "index")
        dfs_len = get_len_dfs(dfs)

        # Inner join on GMT time
        df = join_dfs(dfs, "Gmt time")

        df = apply_ichimo(df, args.target)

        df = apply_rsi(df, "")
        df = apply_rsi(df, "", window=25)
        df = apply_rsi(df, "", window=35)

        df = sup_and_res(df, args.target, window=100)
        last_close = df.tail(1)["Close_" + args.target]
        last_close = last_close.tolist()[0]
        c = df.tail(1)["Close_" + args.target].tolist()[0]
        sup = c + df.tail(1)["closest_sup"].tolist()[0]
        res = c + df.tail(1)["closest_res"].tolist()[0]
        with open(prefix + "SR" + args.target, 'w') as f:
            f.write(str(sup) + "\n" + (str(res) + "\n"))

        df = apply_support_and_resistance(df, args.target)
        df = apply_support_and_resistance(df, args.target, window=50)
        df = apply_support_and_resistance(df, args.target, window=100)
        df = apply_support_and_resistance(df, args.target, window=25)
        df = apply_support_and_resistance(df, args.target, window=200)

        # df = apply_diff_on(df, ["Volume_" + args.target])

        #           ---- Momentum ---- #
        # df = apply_momentum(df, "Close_" + args.target, window=5)
        df = apply_momentum(df, "Close_" + args.target, window=50)
        df = apply_momentum(df, "Close_" + args.target, window=10)
        # df = apply_momentum(df, "Close_" + args.target, window=15)
        df = apply_momentum(df, "Close_" + args.target, window=25)
        df = apply_momentum(df, "Close_" + args.target, window=100)
        df = apply_momentum(df, "Close_" + args.target, window=200)

        #           ---- Stochastic ---- #
        df = apply_stochastic(df, args.target, window=50, mean=5)
        df = apply_stochastic(df, args.target, window=10, mean=1)
        # df = apply_stochastic(df, args.target, window=9, mean=1)
        df = apply_stochastic(df, args.target, window=80, mean=1)
        df = apply_stochastic(df, args.target, window=200, mean=20)
        df = apply_stochastic(df, args.target, window=400, mean=25)
        #  df = apply_stochastic(df, args.target, window=8, mean=2)

        #           ---- Williams ---- #
        df = apply_williams(df, args.target, window=20)
        df = apply_williams(df, args.target, window=9)
        df = apply_williams(df, args.target, window=45)
        # df = apply_williams(df, args.target, window=7)
        df = apply_williams(df, args.target, window=105)

        #           ---- PROC ---- #
        df = apply_PROC(df, args.target, window=15)
        df = apply_PROC(df, args.target, window=75)
        df = apply_PROC(df, args.target, window=25)
        df = apply_PROC(df, args.target, window=50)
        df = apply_PROC(df, args.target, window=150)
        df = apply_PROC(df, args.target, window=250)

        #           ---- Applying MACD ---#
        df = apply_macd(df, 100, 50)

        #           ---- Bollinger Band ---- #
        df = apply_bollinger_band(df, "Close_" + args.target, window=50)
        df = apply_bollinger_band(df, "Close_" + args.target, window=100)
        df = apply_bollinger_band(df, "Close_" + args.target, window=25)
        df = apply_bollinger_band(df, "Close_" + args.target, window=205)
        df = apply_bollinger_band(df, "Close_" + args.target, window=300)
        df = apply_bollinger_band(df, "Close_" + args.target, window=400)

        #           ----  Heiken Ashi
        df = apply_HeikenAshi(df, args.target)

        #           ----  CCI ------ #
        # df = apply_CCI(df, args.target, window=15)
        df = apply_CCI(df, args.target, window=25)
        df = apply_CCI(df, args.target, window=50)
        df = apply_CCI(df, args.target, window=100)

        #           ----  WCP ------ #
        df = applyWCP(df, args.target)

        #           ---- GARCH ------ #
        # df = apply_GARCH(df, args.target, window=5)
        # df = apply_GARCH(df, args.target, window=10)
        # df = apply_GARCH(df, args.target, window=60)

        #          ---- FOURIER ----- #
        df = apply_fourier_on_min(df, args.target, window=300, window_m=5)
        df = apply_fourier_on_min(df, args.target, window=15, window_m=150)
        df = apply_fourier_on_min(df, args.target, window=400, window_m=10)
        df = apply_fourier_on_min(df, args.target, window=150, window_m=3)

        df = apply_fourier_on_max(df, args.target, window=300, window_m=5)
        df = apply_fourier_on_max(df, args.target, window=400, window_m=10)
        df = apply_fourier_on_max(df, args.target, window=150, window_m=3)
        df = apply_fourier_on_max(df, args.target, window=15, window_m=150)

        df = applyFourier(df, args.target, window=20)
        df = applyFourier(df, args.target, window=15)
        df = applyFourier(df, args.target, window=10)
        df = applyFourier(df, args.target, window=50)
        df = applyFourier(df, args.target, window=100)
        df = applyFourier(df, args.target, window=200)
        df = applyFourier(df, args.target, window=300)
        df = applyFourier(df, args.target, window=500)
        df = applyFourier(df, args.target, window=1000)
        #
        df = apply_poly(df, args.target, windows=5)
        df = apply_poly(df, args.target, windows=10)
        df = apply_poly(df, args.target, windows=50)
        df = apply_poly(df, args.target, windows=100)
        df = apply_poly(df, args.target, windows=200)
        df = apply_poly(df, args.target, windows=500)
        df = apply_poly(df, args.target, windows=1000)

        df = apply_poly_on_max(df, args.target, windows_m=5, windows=5)
        df = apply_poly_on_max(df, args.target, windows_m=10, windows=5)
        df = apply_poly_on_max(df, args.target, windows_m=5, windows=15)
        df = apply_poly_on_max(df, args.target, windows_m=5, windows=50)
        df = apply_poly_on_max(df, args.target, windows_m=15, windows=5)
        df = apply_poly_on_max(df, args.target, windows_m=10, windows=50)
        df = apply_poly_on_max(df, args.target, windows_m=10, windows=500)
        df = apply_poly_on_max(df, args.target, windows_m=10, windows=200)

        df = apply_poly_on_min(df, args.target, windows_m=5, windows=5)
        df = apply_poly_on_min(df, args.target, windows_m=10, windows=5)
        df = apply_poly_on_min(df, args.target, windows_m=5, windows=15)
        df = apply_poly_on_min(df, args.target, windows_m=5, windows=50)
        df = apply_poly_on_min(df, args.target, windows_m=15, windows=5)
        df = apply_poly_on_min(df, args.target, windows_m=10, windows=50)
        df = apply_poly_on_min(df, args.target, windows_m=10, windows=500)
        df = apply_poly_on_min(df, args.target, windows_m=10, windows=200)

        df = apply_distance_from_max(df, "Close_" + args.target, window=500)
        df = apply_distance_from_min(df, "Close_" + args.target, window=500)
        df = apply_distance_from_max(df, "Close_" + args.target, window=750)
        df = apply_distance_from_min(df, "Close_" + args.target, window=750)
        df = apply_distance_from_max(df, "Close_" + args.target, window=1500)
        df = apply_distance_from_min(df, "Close_" + args.target, window=1500)

        df = apply_linear_regression(df, "Close_" + args.target, window=30)
        df = apply_linear_regression(df, "Close_" + args.target, window=75)
        df = apply_linear_regression(df, "Close_" + args.target, window=250)
        df = apply_linear_regression(df, "Close_" + args.target, window=125)
        #
        df = apply_linear_regression_on_min(df, "Close_" + args.target, window_m=10, window=75)
        df = apply_linear_regression_on_min(df, "Close_" + args.target, window_m=5, window=15)
        df = apply_linear_regression_on_min(df, "Close_" + args.target, window_m=20, window=250)
        df = apply_linear_regression_on_min(df, "Close_" + args.target, window_m=120, window=5)
        df = apply_linear_regression_on_min(df, "Close_" + args.target, window_m=30, window=125)
        df = apply_linear_regression_on_min(df, "Close_" + args.target, window_m=240, window=5)
        df = apply_linear_regression_on_min(df, "Close_" + args.target, window_m=240, window=10)
        df = apply_linear_regression_on_min(df, "Close_" + args.target, window_m=480, window=10)
        df = apply_linear_regression_on_min(df, "Close_" + args.target, window_m=480, window=5)
        df = apply_linear_regression_on_min(df, "Close_" + args.target, window_m=5, window=15)
        df = apply_linear_regression_on_min(df, "Close_" + args.target, window_m=5, window=5)
        df = apply_linear_regression_on_min(df, "Close_" + args.target, window_m=10, window=5)
        #
        df = apply_linear_regression_on_max(df, "Close_" + args.target, window_m=5, window=15)
        df = apply_linear_regression_on_max(df, "Close_" + args.target, window_m=120, window=5)
        df = apply_linear_regression_on_max(df, "Close_" + args.target, window_m=5, window=5)
        df = apply_linear_regression_on_max(df, "Close_" + args.target, window_m=10, window=75)
        df = apply_linear_regression_on_max(df, "Close_" + args.target, window_m=10, window=5)
        df = apply_linear_regression_on_max(df, "Close_" + args.target, window_m=20, window=250)
        df = apply_linear_regression_on_max(df, "Close_" + args.target, window_m=30, window=125)
        df = apply_linear_regression_on_max(df, "Close_" + args.target, window_m=240, window=5)
        df = apply_linear_regression_on_max(df, "Close_" + args.target, window_m=240, window=10)
        df = apply_linear_regression_on_max(df, "Close_" + args.target, window_m=480, window=5)
        df = apply_linear_regression_on_max(df, "Close_" + args.target, window_m=480, window=10)
        df = apply_df_test(df, args.target)

        df = apply_diff(df, ["Gmt time", "SIN", "COS", "poly"], shift=24)

        df = create_target_ahead(df, "Close_" + args.target, AHEAD, CHANGE_IN_PIPS)

        df['target_in_pips'] = df["Close_" + args.target]
        df = drop_original_values(df, crossList)
        df = drop_columns(df, ["Close_" + args.target + "_diff", "Open_" + args.target + "_diff",
                               "Low_" + args.target + "_diff",
                               "High_" + args.target + "_diff"])

        # ------------ Apply shift -------------------
        df = apply_shift(df, ["Gmt time", "diff", "target", "SIN", "COS", "poly", "shift", "SHIFT"])

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
        train_set_live = df.head(int(args.train_len) - 30)
        to_predict = df.tail(1)

        X_train = train_set_live.ix[:, train_set_live.columns != 'target']
        y_train = train_set_live.ix[:, train_set_live.columns == 'target']
        X_for_col = X_train
        X_test = to_predict.ix[:, to_predict.columns != 'target']
        y_test = to_predict.ix[:, to_predict.columns == 'target']

        # Scaling X
        sc = StandardScaler()
        X_train = sc.fit_transform(X_train)
        X_test = sc.transform(X_test)

        # Just DO FEATURE SELECTION
        param_grid_rf = {'n_estimators': [15, 30, 50, 100, 120], 'max_depth': [4, 5, 7, 12, 15, 20]
                         }
        gdRf = GridSearchCustomModel(RandomForestClassifier(n_jobs=-1, random_state=42), param_grid_rf)
        best_models_, best_score = do_grid_search([gdRf], X_train, y_train.values.ravel(), 100, old_best_models_,
                                                  prefix,
                                                  args.target)
        old_best_models_ = best_models_
        few_features_for_stacking = None
        for i in best_models_:
            if "RandomForest" in str(i.best_estimator_):
                rf = i.best_estimator_
            feature_import = rf.feature_importances_.tolist()
            new_list = [(importance, name) for name, importance in zip(X_for_col.columns.tolist(), feature_import)]
            sorted_by_importance = sorted(new_list, key=lambda tup: tup[0], reverse=True)
            featureToUse = sorted_by_importance[:round(math.sqrt(int(args.train_len) * 2.5))]
            few_features_for_stacking = sorted_by_importance[:round(math.sqrt(int(args.train_len) / 2))]
            with open(prefix + "FS" + args.target, 'a') as f:
                f.write(str(featureToUse) + "\n")
            featureToUse = [x[1] for x in featureToUse]
            few_features_for_stacking = [x[1] for x in few_features_for_stacking]
            newDF = df[featureToUse + ["target"]]
            dfForStacking = df[few_features_for_stacking + ["target"]]

        df = newDF
        df = df.tail(int(args.train_len) + 1)
        train_set_live = df.head(int(args.train_len) - 30)
        to_predict = df.tail(1)

        X_train = train_set_live.ix[:, train_set_live.columns != 'target']
        y_train = train_set_live.ix[:, train_set_live.columns == 'target']
        X_test = to_predict.ix[:, to_predict.columns != 'target']
        y_test = to_predict.ix[:, to_predict.columns == 'target']

        X_train_stacking = train_set_live[few_features_for_stacking]
        y_train_stacking = train_set_live["target"]
        X_test_stacking = to_predict[few_features_for_stacking]

        # Scaling X
        sc = StandardScaler()
        X_train = sc.fit_transform(X_train)
        X_test = sc.transform(X_test)
        # END FEATURE SELECTION

        param_grid_log_reg = {'C': 2.0 ** np.arange(-4, 8)}
        gdLog = GridSearchCustomModel(LogisticRegression(penalty='l1', max_iter=2000, random_state=42),
                                      param_grid_log_reg)

        param_grid_rf = {'n_estimators': [15, 30, 50, 100, 120, 200, 300], 'max_depth': [4, 5, 7, 12, 15, 20]
                         }
        gdRf = GridSearchCustomModel(RandomForestClassifier(n_jobs=-1, random_state=42), param_grid_rf)

        param_grid_svm = [
            {'C': 10.0 ** np.arange(-3, 4), 'kernel': ['linear']}
            # {'C': 10.0 ** np.arange(-3, 4), 'gamma': 8.5 ** np.arange(-3, 3), 'kernel': ['rbf']
            # }
        ]
        gdSVM = GridSearchCustomModel(SVC(probability=True, random_state=42), param_grid_svm)

        param_grid_GB = {'learning_rate': [0.05, 0.03, 0.08, 0.1, 0.2], 'n_estimators': [20, 50, 100, 200],
                         'max_depth': [3, 7, 10, 15]
                         }

        gdGB = GridSearchCustomModel(GradientBoostingClassifier(random_state=42, max_features='auto'), param_grid_GB)

        n_estimators = [15, 50, 100]
        learning_rate = [0.001, 0.01, 0.1]
        param_grid_XGB = dict(learning_rate=learning_rate, n_estimators=n_estimators, max_depth=[3, 7, 10])
        gbXGB = GridSearchCustomModel(XGBClassifier(random_state=42, n_jobs=-1), param_grid_XGB)
        try:
            best_models, best_score = do_grid_search([gdRf, gbXGB], X_train, y_train.values.ravel(), 10000,
                                                     old_best_models, prefix,
                                                     args.target)
            with open(best_score_file, 'a') as f:
                f.write("\n")
                for b in best_score:
                    f.write(str(b) + " ")
                f.write("\n")
            for i in best_models:
                if "RandomForest" in str(i.best_estimator_):
                    rf = i.best_estimator_
            fs.write_feature_importance(rf, featureToUse)
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
        RETRAIN = RETRAIN % 30
        RETRAIN += 1

        tp = df_prediction.tail(1)
        gmt = tp['Gmt time']
        m = [(str(model.best_estimator_), model.best_estimator_) for model in best_models]
        # X_train_stacking_with_predictions = create_dataframe_for_stacking(X_train_stacking, m, X_train)
        voting_classifier = VotingClassifier(estimators=m, voting='soft', n_jobs=-1)
        voting_classifier.fit(X_train, y_train.values.ravel())
        res = voting_classifier.predict(X_test)[0]

        print("vot")
        print(res)
        print(voting_classifier.predict_proba(X_test)[0])
        print(m[0][0])
        print(m[0][1].predict_proba(X_test)[0])
        print(m[1][0])
        print(m[1][1].predict_proba(X_test)[0])
        classes = list(voting_classifier.classes_)
        probs = voting_classifier.predict_proba(X_test).tolist()[0]
        # print("\n\n --- TRAINING STACK MODEL   \n\n")
        # X_test_stacking_with_predictions = create_dataframe_for_stacking(X_test_stacking.reset_index(drop=True), m,
        # X_test)

        # Scaling X
        # sc = StandardScaler()
        # X_train_s = sc.fit_transform(X_train_stacking_with_predictions)
        # X_test_s = sc.transform(X_test_stacking_with_predictions)
        # best_models, best_score = do_grid_search([gdRf, gbXGB], X_train_s, y_train_stacking.values.ravel(), 10000, old_best_models,
        #                             prefix,
        #                             args.target)

        # m = [(str(model.best_estimator_), model.best_estimator_) for model in best_models]
        # voting_classifier = VotingClassifier(estimators=m, voting='soft', n_jobs=-1)
        # voting_classifier.fit(X_train_s, y_train_stacking.values.ravel())
        # res = voting_classifier.predict(X_test_s)[0]

        # classes_s = voting_classifier.classes_
        # probs_s = voting_classifier.predict_proba(X_test_s).tolist()[0]
        # final_probs = []
        # for a, b in zip(classes_s, classes):
        #    assert a == b

        # for p, c in zip(probs, probs_s):
        #    final_probs.append((p + c) / 2.0)

        # final_res = None
        # high_prob = 0
        # for p, c in zip(final_probs, classes_s):
        #    if p > high_prob:
        #        high_prob = p
        #        final_res = c

        print("vot_stacking")
        print(res)
        # print(voting_classifier.predict_proba(X_test_s)[0])
        # print(m[0][0])
        # print(m[0][1].predict_proba(X_test_s)[0])
        # print(m[1][0])
        # print(m[1][1].predict_proba(X_test_s)[0])

        # for i in best_models:
        #     if "RandomForest" in str(i.best_estimator_):
        #         rf = i.best_estimator_
        #     else:
        #         continue
        #     feature_import = rf.feature_importances_.tolist()
        #     new_list = [(importance, name) for name, importance in
        #                 zip(X_train_stacking_with_predictions.columns.tolist(), feature_import)]
        #     sorted_by_importance = sorted(new_list, key=lambda tup: tup[0], reverse=True)
        #     featureToUse = sorted_by_importance
        #     with open(prefix + "FS" + args.target, 'a') as f:
        #         f.write("STACK\n")
        #         f.write(str(featureToUse) + "\n")

        with open(result_file, 'w') as f:
            f.write(res)

        os.remove(notification_file)
        old_best_models = best_models

        if fs_counter % 30 == 0:
            fs_counter = 0
            fs.consolidate_feature_importance()

        counter += 1
        if res == "OUT":
            counter += 1
            counter %= round(AHEAD / 4)
        else:
            counter %= round(AHEAD)

    print('end')
