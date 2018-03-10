import math
import random

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import RFECV
from sklearn.metrics import f1_score, accuracy_score
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler

from fitmodel.fitmodel import do_grid_search
from gridSearch.gridSearch import GridSearchCustomModel
from processing.processing import AHEAD


class CoolData(object):
    def __init__(self, X_train, X_test, y_train, best_score, train_len, toTestX, toTestY):
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.best_score = best_score
        self.train_len = train_len
        self.toTestX = toTestX
        self.toTestY = toTestY


def do_feature_extraction(df_, target, prefix, train_len_original):
    to_return = []
    rows = df_.shape[0]
    print("NUMBER OF ROWS: " + str(rows))
    NUMBER_TO_PREDICT = 1
    trains = [ 150, 300, 400, 600, 800, 1000, 1200, 1500, 2000]
    # Let's prepare N DF with different lengths
    for train_len in trains:

        train_len = int(train_len)

        # calculate test size
        samples_to_test = int(0.2 * train_len) + 1

        if train_len + samples_to_test + AHEAD + 10 > rows:
            continue

        print("Preparing DF for train len = " + str(train_len))

        # take the tail except the last AHEAD
        df = df_.tail(train_len + samples_to_test + AHEAD).reset_index(drop=True)
        newdf = df
        sell = newdf[newdf["target"] == "SELL"].shape[0]
        buy = newdf[newdf["target"] == "BUY"].shape[0]
        out = newdf[newdf["target"] == "OUT"].shape[0]
        print("SELL " + str(sell))
        print("BUY " + str(buy))
        print("OUT " + str(out))
        if sell == 0 or buy == 0:
            continue
        train_set_live = df.head(train_len).reset_index(drop=True)
        for_testing = df.tail(samples_to_test + AHEAD)[: -(AHEAD + 1)].reset_index(drop=True)

        # train_set_live = train_set_live[train_set_live["target"] != "OUT"]
        # for_testing = for_testing[for_testing["target"] != "OUT"]
        for_testing_X = for_testing.ix[:, for_testing.columns != 'target']
        for_testing_y = for_testing.ix[:, for_testing.columns == 'target']

        keep = [c for c
                in list(for_testing_X)
                if len(for_testing_X[c].unique()) > 1]

        #for_testing_X = for_testing_X[keep]
        #train_set_live = train_set_live[keep]

        X_train = train_set_live.ix[:, train_set_live.columns != 'target']
        y_train = train_set_live.ix[:, train_set_live.columns == 'target']
        X_for_col = X_train
        # Scaling X
        sc = StandardScaler()
        try:

            X_train = sc.fit_transform(X_train)
            for_testing_X_scaled = sc.transform(for_testing_X)
            for_testing_X_scaled = for_testing_X_scaled[~np.isnan(for_testing_X_scaled).any(axis=1)]
            for_testing_X_scaled = for_testing_X_scaled[np.isfinite(for_testing_X_scaled).all(axis=1)]
            # Just DO FEATURE SELECTION
            param_grid_rf = {'n_estimators': [100, 200, 500], 'max_depth': [4, 5, 7, 12, 15, 20]
                             }
            gdRf = GridSearchCustomModel(
                RandomForestClassifier(n_jobs=-1, random_state=42, min_weight_fraction_leaf=0.05, max_features=1),
                param_grid_rf)
            best_models_, best_score = do_grid_search([gdRf], X_train, y_train.values.ravel(), 100, None,
                                                      prefix,
                                                      target)
            best_score = best_score[0]
            if best_score < 0.48:
                print("Best score too low: " + str(best_score))
                continue
            bm = best_models_[0].best_estimator_

            for_testing_y_pred = bm.predict(for_testing_X_scaled)
        except Exception as e:
            print(e)
            print("fuckoff scaler")
            continue
        f1_test = accuracy_score(for_testing_y.values.ravel(), for_testing_y_pred)

        if f1_test < 0.48:
            print("Best TEST SCORE too low: " + str(f1_test))
            continue

        print("F1 Test is good: " + str(f1_test))
        for i in best_models_:
            if "RandomForest" in str(i.best_estimator_):
                rf = i.best_estimator_
            feature_import = rf.feature_importances_.tolist()
            new_list = [(importance, name) for name, importance in zip(X_for_col.columns.tolist(), feature_import)]
            sorted_by_importance = sorted(new_list, key=lambda tup: tup[0], reverse=True)
            featureToUse = sorted_by_importance[:round(math.sqrt(int(train_len) * 5))]
            with open(prefix + "FS" + target + "_IT", 'a') as f:
                f.write(str(train_len) + "\n")
                f.write(str(featureToUse) + "\n")
            featureToUse = [x[1] for x in featureToUse]
            # Running additional feature selection
            # Create the RFE object and compute a cross-validated score.
            # model = rf
            # # The "accuracy" scoring is proportional to the number of correct
            # # classifications
            # print("Starting new Feature Selection!")
            # rfecv = RFECV(estimator=model, step=0.01, cv=TimeSeriesSplit(n_splits=5).split(X_train),
            #               scoring='accuracy', n_jobs=-1)
            # rfecv.fit(X_train, y_train.values.ravel())
            # supports = rfecv.get_support(indices=True)
            # rfecv_new_fs = []
            # for index_retention in supports:
            #     if X_for_col.columns.tolist()[index_retention] in featureToUse :
            #         rfecv_new_fs.append(X_for_col.columns.tolist()[index_retention])
            #     else:
            #         value = random.randint(1, 101)
            #         if value < 3:
            #             to_add = X_for_col.columns.tolist()[index_retention]
            #             print(" Ripescata come Bersani: " + str(to_add))
            #             rfecv_new_fs.append(to_add)
            #
            # print("Old #Features: " + str(len(featureToUse)))
            # print("New #Features: " + str(len(rfecv_new_fs)))
            #
            # featureToUse = rfecv_new_fs
            # with open(prefix + "FS" + target + "_IT", 'a') as f:
            #     f.write("features selection")
            #     f.write(str(train_len) + "\n")
            #     f.write(str(featureToUse) + "\n")
            newDF = df_[featureToUse + ["target"]]

        df = newDF
        if not featureToUse:
            continue

        # df now is the original dataframe with less columns included
        df = df.tail(train_len + samples_to_test + AHEAD).reset_index(drop=True)
        train_set_live = df.head(train_len).reset_index(drop=True)
        to_test = df.tail(samples_to_test + AHEAD)[: -(AHEAD + 1)].reset_index(drop=True)
        to_predict = df.tail(NUMBER_TO_PREDICT).reset_index(drop=True)
        assert to_predict["target"][NUMBER_TO_PREDICT - 1] == "OUT"
        X_train = train_set_live.ix[:, train_set_live.columns != 'target']
        y_train = train_set_live.ix[:, train_set_live.columns == 'target']
        X_test = to_predict.ix[:, to_predict.columns != 'target']
        to_test_X = to_test.ix[:, to_test.columns != "target"]
        to_test_y = to_test.ix[:, to_test.columns == "target"]
        # Scaling X
        sc = StandardScaler()
        X_train = sc.fit_transform(X_train)
        X_test = sc.transform(X_test)
        to_test_X_scaled = sc.transform(to_test_X)
        to_test_X_scaled = to_test_X_scaled[~np.isnan(to_test_X_scaled).any(axis=1)]
        to_test_X_scaled = to_test_X_scaled[np.isfinite(to_test_X_scaled.all(axis=1))]
        to_return.append(CoolData(X_train, X_test, y_train, best_score, train_len, to_test_X_scaled, to_test_y))

    return to_return
