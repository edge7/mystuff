from time import time
from sklearn.model_selection import GridSearchCV


def fit_models(list_models, X, y):
    for model in list_models:
        print("training model \n")
        print(model)
        model.fit(X, y)
    return list_models


def do_grid_search(list_models, X, y):
    to_return = []
    print("Shape X \n")
    print(X.shape)
    print("#Columns Y \n")
    print(y.shape)
    for model in list_models:
        print("Grid Search running for model \n")
        print(model.model)
        print("  ...  \n")
        start = time()
        clf = GridSearchCV(model.model, model.params, cv=5, n_jobs=-1, verbose=1, scoring='f1')
        clf.fit(X, y)
        end = time()
        elapsed = (end - start) / 60.0
        print("Grid Search done in " + str(elapsed) + " minutes\n")
        print(" printing best parameters: \n")
        print(clf.best_params_)
        print(" Print best score \n")
        print(clf.best_score_)
        to_return.append(clf)

    return to_return
