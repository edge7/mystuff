import argparse
import os

from sklearn.metrics import f1_score, accuracy_score
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier

from economic_calendar.economic_calendar import check_if_available, apply_calendar
from feature_selection.feature_extraction import do_feature_extraction

mingw_path = 'C:\\Program Files\\mingw-w64\\x86_64-7.2.0-posix-seh-rt_v5-rev1\\mingw64\\bin'
os.environ['PATH'] = mingw_path + ';' + os.environ['PATH']

import pathlib
import time
import numpy as np
from sklearn.ensemble import RandomForestClassifier, BaggingClassifier
from sklearn.ensemble import VotingClassifier
from sklearn.linear_model import LogisticRegression
from xgboost.sklearn import XGBClassifier
from feature_selection.feature_importance import FeatureSelection
from fitmodel.fitmodel import do_grid_search
from gridSearch.gridSearch import GridSearchCustomModel
from processing.processing import create_dataframe, drop_column, join_dfs, drop_original_values, \
    apply_macd, apply_bollinger_band, drop_columns, apply_stochastic, create_target_ahead, AHEAD, \
    apply_momentum, apply_rsi, apply_diff, subset_df, apply_support_and_resistance, apply_df_test, apply_shift_just_at, \
    apply_mayority, apply_mv_avg_to, apply_mv_avg_just_to, applyFourier
from utility.utility import get_len_dfs

CHANGE_IN_PIPS = 0.0
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
    test_file = prefix + "test_Score_" + args.target
    counter = 0
    crossList.append(args.target)
    old_best_models = None
    old_best_models_ = None
    old_calendar = None

    RETRAIN = 0
    fs = None
    fs_counter = 0

    # Economic Calendar
    file = "C:\\Users\\Administrator\\Desktop\\CalendarioEconomico"
    calendars = check_if_available(file, args.target)
    import warnings

    warnings.filterwarnings("ignore")
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
            # with open(data_file, 'r') as f:
            #     lines = f.readlines()
            # with open(data_file, 'w') as f:
            #     f.writelines(lines[:1] + lines[2:])
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



        #           ---- Applying MACD ---#
        df = apply_macd(df, 26, 9)

        # df = apply_ichimo(df, args.target)

        df = apply_rsi(df, "")
        df = apply_rsi(df, "", window=25)
        # df = apply_rsi(df, "", window=35)

        # df = sup_and_res(df, args.target, window=100)
        df = applyFourier(df, args.target, window=48)
        df = applyFourier(df, args.target, window=150)
        df = applyFourier(df, args.target, window=300)
        df = apply_support_and_resistance(df, args.target, window=50)
        df = apply_support_and_resistance(df, args.target, window=100)
        df = apply_support_and_resistance(df, args.target, window=25)
        # df = apply_support_and_resistance(df, args.target, window=200)

        # df = apply_diff_on(df, ["Volume_" + args.target])

        #           ---- Momentum ---- #
        df = apply_momentum(df, "Close_" + args.target, window=5)
        df = apply_momentum(df, "Close_" + args.target, window=3)
        df = apply_momentum(df, "Close_" + args.target, window=8)
        df = apply_momentum(df, "Close_" + args.target, window=50)
        df = apply_momentum(df, "Close_" + args.target, window=10)
        # df = apply_momentum(df, "Close_" + args.target, window=15)
        # df = apply_momentum(df, "Close_" + args.target, window=25)
        df = apply_momentum(df, "Close_" + args.target, window=100)
        # df = apply_momentum(df, "Close_" + args.target, window=200)

        #           ---- Stochastic ---- #
        df = apply_stochastic(df, args.target, window=50, mean=10)
        df = apply_stochastic(df, args.target, window=10, mean=3)

        #           ---- Bollinger Band ---- #
        df = apply_bollinger_band(df, "Close_" + args.target, window=50)
        df = apply_bollinger_band(df, "Close_" + args.target, window=100)
        df = apply_bollinger_band(df, "Close_" + args.target, window=400)
        df = apply_df_test(df, args.target)

        # df = apply_coefs_behind(df, "Close_" + args.target)
        # df = apply_std(df, args.target, window=30)
        df = apply_diff(df, ["Gmt time", "SIN", "COS", "poly"], shift=3)
        df = apply_diff(df, ["Gmt time", "SIN", "COS", "poly"], shift=6)
        #df = apply_diff(df, ["Gmt time", "SIN", "COS", "poly"], shift=12)
        #df = apply_diff(df, ["Gmt time", "SIN", "COS", "poly", "diff"], shift=24)
        # df = apply_diff(df, ["Gmt time", "SIN", "COS", "poly", "_diff_shift_"], shift=6 * 3)
         #df = apply_mayority(df, args.target, window=100)

        df['target_in_pips'] = df["Close_" + args.target]

        df = drop_columns(df, ["Close_" + args.target + "_diff", "Open_" + args.target + "_diff",
                               "Low_" + args.target + "_diff",
                               "High_" + args.target + "_diff"])

        df_prediction = df.copy()
        df = df.replace([np.inf, -np.inf], np.nan)
        df = df.dropna(how='any').reset_index(drop=True)
        df, old_calendar = apply_calendar(df, calendars, old_calendar=old_calendar)
        df = df.replace([np.inf, -np.inf], np.nan)
        df = df.dropna(how='any').reset_index(drop=True)
        df = apply_shift_just_at(df, ["LAST"])
        df = df.replace([np.inf, -np.inf], np.nan)
        df = df.dropna(how='any')
        df = apply_mv_avg_to(df, ["target"], window=5)
        df = df.replace([np.inf, -np.inf], np.nan)
        df = df.dropna(how='any').reset_index(drop=True)
        df = apply_mv_avg_to(df, ["target", "AVG"], window=10)
        df = df.replace([np.inf, -np.inf], np.nan)
        df = df.dropna(how='any').reset_index(drop=True)
        df = apply_mv_avg_to(df, ["target", "AVG"], window=60)
        df = df.replace([np.inf, -np.inf], np.nan)
        df = df.dropna(how='any').reset_index(drop=True)
        df = apply_mv_avg_to(df, ["target", "AVG"], window=120)
        df = df.replace([np.inf, -np.inf], np.nan)
        df = df.dropna(how='any').reset_index(drop=True)
        df = apply_mv_avg_just_to(df, ["LAST"], window=240)
        df = df.replace([np.inf, -np.inf], np.nan)
        df = df.dropna(how='any').reset_index(drop=True)
        df = apply_mv_avg_just_to(df, ["LAST"], window=6*120)
        df = create_target_ahead(df, "Close_" + args.target, AHEAD, CHANGE_IN_PIPS)

        df = df.replace([np.inf, -np.inf], np.nan)
        df = df.dropna(how='any').reset_index(drop=True)
        #df = subset_df(df, args.target, window=100)
        df = df.replace([np.inf, -np.inf], np.nan)
        df = df.dropna(how='any').reset_index(drop=True)
        target_in_pips = df['target_in_pips']
        t = df['target']
        with open(prefix + " " + args.target, 'a') as f:
            f.write(str(t.tolist()[-int(args.train_len):]) + "\n")
        gmt = df['Gmt time']
        df = drop_column([df], "Gmt time")[0]
        df = drop_original_values(df, crossList)
        df = drop_column([df], 'target_in_pips')[0]
        if fs is None:
            fs = FeatureSelection(df, prefix + "_" + args.target)

        total_length = df.shape[0]
        start = 0
        train_len = int(args.train_len)
        df = df.replace([np.inf, -np.inf], np.nan)
        df = df.dropna(how='any').reset_index(drop=True)
        dfs = do_feature_extraction(df, args.target, prefix, train_len)
        if len(dfs) == 0:
            print("No good model Skipping !!!")
            with open(result_file, 'w') as f:
                f.write("OUT")
                counter += 1
                counter %= round(AHEAD)
            os.remove(notification_file)
            continue

        param_grid_log_reg = {'C': 2.0 ** np.arange(-3, 7)}
        gdLog = GridSearchCustomModel(LogisticRegression(penalty='l1', max_iter=2000, random_state=42),
                                      param_grid_log_reg)

        param_grid_rf = {'n_estimators': [ 100, 120, 200, 300, 500], 'max_depth': [4, 5, 7, 12, 15, 20]
                         }
        gdRf = GridSearchCustomModel(
            RandomForestClassifier(n_jobs=-1, random_state=42, class_weight="balanced", min_weight_fraction_leaf=0.05,
                                   criterion="entropy"),
            param_grid_rf)

        n_estimators = [15, 50, 100]
        learning_rate = [0.001, 0.01, 0.1]
        param_grid_XGB = dict(learning_rate=learning_rate, n_estimators=n_estimators, max_depth=[3, 7, 10])
        gbXGB = GridSearchCustomModel(XGBClassifier(random_state=42, n_jobs=-1), param_grid_XGB)

        bbc = GridSearchCustomModel(
            BaggingClassifier(base_estimator=DecisionTreeClassifier(criterion="entropy", class_weight="balanced",
                                                                    max_features="auto"),
                              random_state=42, n_jobs=-1, max_features=1),
            {'n_estimators': [100, 200, 500]})

        rfbc = GridSearchCustomModel(
            BaggingClassifier(
                base_estimator=RandomForestClassifier(criterion="entropy", class_weight="balanced_subsample",
                                                      max_features="auto", n_estimators=1, bootstrap=True),
                random_state=42, n_jobs=-1, max_features=1),
            {'n_estimators': [100, 200, 500], 'max_features': [1, 3, 5, 7]}
        )

        param_grid_ANN = {"hidden_layer_sizes": [(20, 20), (15,), (5,), (10, 10), (10, 5),
                                                 (5, 10), (5, 5, 5), (10, 10, 10)],
                          'activation': ['tanh'],
                          'alpha': np.logspace(-5, 3, 5)}

        gdANN = GridSearchCustomModel(
            MLPClassifier(solver='lbfgs', random_state=42, verbose=False, max_iter=1200, early_stopping=True),
            param_grid_ANN)

        toAverage = []
        for cool in dfs:
            X_train = cool.X_train

            y_train = cool.y_train
            X_test = cool.X_test
            l = cool.train_len
            toTestX = cool.toTestX
            toTestY = cool.toTestY
            print("Training train_len = " + str(l))

            try:
                best_models, best_score = do_grid_search([gdRf, gdANN, gbXGB ], X_train, y_train.values.ravel(),
                                                         10000,
                                                         old_best_models, prefix,
                                                         args.target)

                for i in best_models:
                    if "RandomForest" in str(i.best_estimator_):
                        rf = i.best_estimator_

            except Exception as e:
                print(e)
                best_models = old_best_models
                with open(prefix + "_" + args.target + "_LOG", 'a') as f:
                    f.write(str(e))

                os.remove(notification_file)
                with open(result_file, 'w') as f:
                    f.write("HOLD")
                continue
            good_ones = []
            for model in best_models:
                res_test = model.predict(toTestX)
                f1_test = accuracy_score(toTestY.values.ravel(), res_test)
                if f1_test < 0.58:
                    print("Test before voting low" + str(f1_test))
                    print("getting rid of " + (str(model.best_estimator_)))
                    continue
                else:
                    print("test for " + (str(model.best_estimator_)) + str(f1_test))
                    print("\n\n")
                    good_ones.append(model)

            if not good_ones:
                os.remove(notification_file)
                print("NO GOOD RESULT before votation skipping")
                with open(result_file, 'w') as f:
                    f.write("OUT")
                continue

            m = [(str(model.best_estimator_), model.best_estimator_) for model in good_ones]
            voting_classifier = VotingClassifier(estimators=m, voting='soft', n_jobs=-1)
            voting_classifier.fit(X_train, y_train.values.ravel())
            res = voting_classifier.predict(X_test)[0]

            # print("vot")
            # print(res)
            # print(voting_classifier.predict_proba(X_test)[0])
            # print(m[0][0])
            # print(m[0][1].predict_proba(X_test)[0])
            # print(m[1][0])
            # print(m[1][1].predict_proba(X_test)[0])
            classes = list(voting_classifier.classes_)
            probs = voting_classifier.predict_proba(X_test).tolist()[0]
            r = 0.0
            for i in best_score:
                r += i
            r = r / float(len(best_score))
            best_score = r
            if best_score < 0.45:
                print("Skipping as best score is: " + str(best_score))
            else:
                res_test = voting_classifier.predict(toTestX)
                f1_test = accuracy_score(toTestY.values.ravel(), res_test)
                if f1_test < 0.52:
                    print("Best TEST SCORE too low: SUCK!" + str(f1_test))
                    continue
                else:
                    with open(test_file, 'a') as fp:
                        fp.write("\n  Length: " + str(l) + " TEST SCORE: " + str(f1_test))
                    print("\n  Length: " + str(l) + " Classifier TEST SCORE: " + str(f1_test))
                    toAverage.append((classes, probs, l, f1_test))

        print("Averaging results ")
        if len(toAverage) == 0:
            res = "OUT"
            print("No Trade, no good models !!!")
        else:
            cl = toAverage[0][0]
            kl = []
            for _ in cl:
                kl.append(0.0)
            for c, p, lun, bs in toAverage:
                assert c == cl
                print("len: " + str(lun))
                print("Prob " + str(p))
                print("Score " + str(bs))
                for i, v in enumerate(p):
                    kl[i] += (bs * v)
                with open(best_score_file, 'a') as f:
                    f.write("\n")
                    f.write(str(bs) + " ")
                    f.write("\n")
            index_max = 0
            p_max = -1
            for i, k in enumerate(kl):

                if k > p_max:
                    p_max = k
                    index_max = i
            res = cl[index_max]
            print("RES: " + str(res))
            print("PROB: " + str(p_max))
            print("ALL PROBS:" + str(kl))

        with open(result_file, 'w') as f:
            f.write(res)

        os.remove(notification_file)
        old_best_models = best_models

        if fs_counter % 30 == 0:
            fs_counter = 0
            fs.consolidate_feature_importance()

        counter += 1
        if res == "OUT":
            counter = AHEAD - 3
            # counter %= round(AHEAD)
        else:
            counter %= round(AHEAD)

    print('end')
