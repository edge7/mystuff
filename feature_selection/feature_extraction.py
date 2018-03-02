import math
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, accuracy_score
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
    samples_to_test = 6 * 50
    rows = df_.shape[0]
    trains = [ rows, int(rows/2)]
    # Let's prepare N DF with different lengths
    for train_len in trains:

        print("Preparing DF for train len = " + str(train_len))
        df = df_.tail(int(train_len) + samples_to_test).reset_index(drop=True)
        train_set_live = df.head(int(train_len) - AHEAD - 1)[:-samples_to_test].reset_index(drop=True)
        for_testing = df.head(int(train_len) - AHEAD - 1).tail(samples_to_test).reset_index(drop=True)
        # train_set_live = train_set_live[train_set_live["target"] != "OUT"]
        # for_testing = for_testing[for_testing["target"] != "OUT"]
        for_testing_X = for_testing.ix[:, for_testing.columns != 'target']
        for_testing_y = for_testing.ix[:, for_testing.columns == 'target']

        X_train = train_set_live.ix[:, train_set_live.columns != 'target']
        y_train = train_set_live.ix[:, train_set_live.columns == 'target']
        X_for_col = X_train
        # Scaling X
        sc = StandardScaler()
        X_train = sc.fit_transform(X_train)
        for_testing_X_scaled = sc.transform(for_testing_X)
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
        if best_score < 0.30:
            print("Best score too low: " + str(best_score))
            continue
        bm = best_models_[0].best_estimator_

        for_testing_y_pred = bm.predict(for_testing_X_scaled)
        f1_test = accuracy_score(for_testing_y.values.ravel(), for_testing_y_pred)
        if f1_test < 0.20:
            print("Best TEST SCORE too low: " + str(f1_test))
            continue
        print("F1 Test is good: " + str(f1_test))
        for i in best_models_:
            if "RandomForest" in str(i.best_estimator_):
                rf = i.best_estimator_
            feature_import = rf.feature_importances_.tolist()
            new_list = [(importance, name) for name, importance in zip(X_for_col.columns.tolist(), feature_import)]
            sorted_by_importance = sorted(new_list, key=lambda tup: tup[0], reverse=True)
            featureToUse = sorted_by_importance[:round(math.sqrt(int(train_len) * 15))]
            with open(prefix + "FS" + target + "_IT", 'a') as f:
                f.write(str(train_len) + "\n")
                f.write(str(featureToUse) + "\n")
            featureToUse = [x[1] for x in featureToUse]
            newDF = df[featureToUse + ["target"]]

        df = newDF
        df = df.tail(int(train_len) + 1)
        train_set_live = df.head(int(train_len + samples_to_test) - AHEAD)[:-samples_to_test]
        to_test = df.head(int(train_len + samples_to_test) - AHEAD).tail(samples_to_test)[:-1]
        to_predict = df.tail(1).reset_index(drop=True)
        assert to_predict["target"][0] == "OUT"
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

        to_return.append(CoolData(X_train, X_test, y_train, best_score, train_len, to_test_X_scaled, to_test_y))

    return to_return
