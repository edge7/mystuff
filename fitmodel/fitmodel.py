from time import time
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit


def fit_models(list_models, X, y):
    for model in list_models:
        print("training model \n")
        print(model)
        model.fit(X, y)
    return list_models


def do_grid_search(list_models, X, y, counter, old_best_models, prefix=None, t=None):
    to_return = []
    best_score = []
    print("Shape X \n")
    print(X.shape)
    print("#Columns Y \n")
    print(y.shape)
    c = -1

    for model in list_models:
        c += 1
        if counter < 29 and old_best_models is not None:
            to_return.append(old_best_models[c])
            print("\n\n ***** NO MODEL GENERATED: " + str(model.model)[0:10] + "\n")
            continue
        my_cv = TimeSeriesSplit(n_splits=5).split(X)
        print("\n\n\n ***** NEW MODEL FOR MODEL " + str(model.model)[0:10] +
              " is being generated ****\n\n\n")
        print("Grid Search running for model \n")
        print(model.model)
        print("  ...  \n")
        start = time()
        clf = GridSearchCV(model.model, model.params, refit=True, cv=my_cv, n_jobs=-1, verbose=0, scoring='f1_macro')
        clf.fit(X, y)
        end = time()
        elapsed = (end - start) / 60.0
        print("Grid Search done in " + str(elapsed) + " minutes\n")
        print(" printing best parameters:")
        print(clf.best_params_)
        print(" Print best score:")
        print(clf.best_score_)
        best_score.append(abs(clf.best_score_))
        if prefix is not None:
            with open(prefix + "_" + t + "_SCORE", 'a') as f:
                f.write("\n *** " + str(clf.best_score_))
                f.write("\nShape X \n")
                f.write(str(X.shape))

        to_return.append(clf)

    return to_return, best_score


def modify_res_in_according_to(res, model):
    tmp = res.tolist()
    best_score = model.best_score_
    to_return = []
    for i in tmp:
        if best_score > 0.52:
            to_return.append(i)
        elif best_score < 0.48:
            to_return.append((i * -1.0))
        else:
            to_return.append(0.0)
    return to_return
