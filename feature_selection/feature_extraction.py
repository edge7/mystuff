import math
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler

from fitmodel.fitmodel import do_grid_search
from gridSearch.gridSearch import GridSearchCustomModel


class CoolData(object):
    def __init__(self, X_train, X_test, y_train, best_score):
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.best_score = best_score


def do_feature_extraction(df_, target, prefix, train_len_original):
    to_return = []

    # Let's prepare N DF with different lengths
    for train_len in [round(train_len_original * 2), round(train_len_original / 2.0), round(train_len_original / 3.0),
                      train_len_original]:
        print("Preparing DF for train len = " + train_len)
        df = df_.tail(int(train_len) + 1).reset_index(drop=True)
        train_set_live = df_.head(int(train_len) - 30)

        X_train = train_set_live.ix[:, train_set_live.columns != 'target']
        y_train = train_set_live.ix[:, train_set_live.columns == 'target']
        X_for_col = X_train
        # Scaling X
        sc = StandardScaler()
        X_train = sc.fit_transform(X_train)

        # Just DO FEATURE SELECTION
        param_grid_rf = {'n_estimators': [15, 30, 50, 100, 120], 'max_depth': [4, 5, 7, 12, 15, 20]
                         }
        gdRf = GridSearchCustomModel(RandomForestClassifier(n_jobs=-1, random_state=42), param_grid_rf)
        best_models_, best_score = do_grid_search([gdRf], X_train, y_train.values.ravel(), 100, None,
                                                  prefix,
                                                  target)
        if best_score < 55.0:
            print("Best score too low: " + best_score)
            continue

        for i in best_models_:
            if "RandomForest" in str(i.best_estimator_):
                rf = i.best_estimator_
            feature_import = rf.feature_importances_.tolist()
            new_list = [(importance, name) for name, importance in zip(X_for_col.columns.tolist(), feature_import)]
            sorted_by_importance = sorted(new_list, key=lambda tup: tup[0], reverse=True)
            featureToUse = sorted_by_importance[:round(math.sqrt(int(train_len) * 2.5))]
            few_features_for_stacking = sorted_by_importance[:round(math.sqrt(int(train_len) / 2))]
            with open(prefix + "FS" + target + "_" + str(train_len), 'a') as f:
                f.write(str(featureToUse) + "\n")
            featureToUse = [x[1] for x in featureToUse]
            newDF = df[featureToUse + ["target"]]

        df = newDF
        df = df.tail(int(train_len) + 1)
        train_set_live = df.head(int(train_len) - 30)
        to_predict = df.tail(1)

        X_train = train_set_live.ix[:, train_set_live.columns != 'target']
        y_train = train_set_live.ix[:, train_set_live.columns == 'target']
        X_test = to_predict.ix[:, to_predict.columns != 'target']

        # Scaling X
        sc = StandardScaler()
        X_train = sc.fit_transform(X_train)
        X_test = sc.transform(X_test)
        to_return.append(CoolData(X_train, X_test, y_train, best_score))

    return to_return
