import argparse
import pathlib
import numpy as np
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from gridSearch.gridSearch import GridSearchCustomModel
from processing.processing import create_dataframe, drop_column, join_dfs, apply_diff, create_y, drop_original_values, \
    apply_macd
from reporting.reporting import CustomReport
from utility.utility import get_len_dfs
from fitmodel.fitmodel import do_grid_search

TARGET_VARIABLE = "Close__diff"
crossList = []

# Stock version 
if __name__ == "__main__":

    # Getting path where CSVs are stored
    parser = argparse.ArgumentParser("Ml Forex")
    parser.add_argument("--datapath", help="complete path to CSV file directory")
    parser.add_argument("--target", help="target variable")
    parser.add_argument("--train_len", help="train len")
    parser.add_argument("--predict", help="predict next output")

    args = parser.parse_args()
    TARGET_VARIABLE = "Close_" + args.target + "_diff"
    flist = [p for p in pathlib.Path(args.datapath).iterdir() if p.is_file()]
    crossList.append(args.target)
    # Creating n dataframe where n is the number of files
    dfs = create_dataframe(flist, "Gmt time", crossList)

    # Dropping volume columns
    # dfs = drop_column(dfs, "Volume")
    dfs = drop_column(dfs, "index")
    dfs_len = get_len_dfs(dfs)

    # Inner join on GMT time
    df = join_dfs(dfs, "Gmt time")
    # if args.predict == 'yes':
    #    df = df.tail(int(args.train_len)*2)
    # Check that len is the same (inner join validation)
    # check_len_is_same(dfs_len, len(df.index))

    # Applying MAC
    df = apply_macd(df, 26, 12)

    # Apply diff to the column except for Gmt time
    df = apply_diff(df, "Gmt time")

    # Close_xdiff is the difference between Close_i - Close_i-1
    df['target'] = df.apply(lambda row: create_y(row, TARGET_VARIABLE), axis=1)

    # We don't want to have target in the same row as Close_xdiff, we want to move it up (shifting)
    # When we will make predictions in realtime, we want to predict Close_xdiff starting from previous row
    df['target'] = df['target'].shift(-1)
    df['target_in_pips'] = df[TARGET_VARIABLE].shift(-1)
    # df = drop_column([df], "diff")[0]
    df = drop_original_values(df, crossList)
    # Apply percentage excluding the diff calculated above, gtm time and target column
    # df = apply_percentage(df, ["diff", "Gmt time", "target", "pips"]).ix[1:][:-1]
    # df = apply_mvavg(df, ["target", "_macdhist", "_signalline", "_macdline"], 5)
    # df = apply_mvavg(df, ["target", "_mv_avg", "_macdhist", "_signalline", "_macdline"], 10)
    df_prediction = df.copy()
    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.dropna(how='any')
    target_in_pips = df['target_in_pips']
    t = df['target']
    gmt = df['Gmt time']
    df = drop_column([df], "Gmt time")[0]
    #df = drop_column([df], "diff")[0]
    df = drop_column([df], 'target_in_pips')[0]
    # df = modify_time([df])[0]
    # df = df[df.target != 0]
    # plt.hist(df["target"].values.ravel())
    # plt.show()

    # Preparing reporting object
    report = CustomReport(args.datapath, df, TARGET_VARIABLE, args.train_len, args.predict, target_in_pips.cumsum(), gmt)
    report.init()
    # Splitting
    # train_set, test_set = train_test_split(df, test_size=0.2, random_state=42)
    total_length = df.shape[0]
    start = 0
    train_len = int(args.train_len)

    test_len = 1

    while start + train_len + test_len <= total_length:

        train_set = df.iloc[start: start + train_len]
        test_set = df.iloc[start + train_len: start + train_len + test_len]
        train_set = train_set.reset_index(drop=True)
        test_set = test_set.reset_index(drop=True)
        report.write_step(start, train_len, test_len)
        # poly_features = PolynomialFeatures(degree=1, include_bias=False)

        X_train = train_set.ix[:, train_set.columns != 'target']
        y_train = train_set.ix[:, train_set.columns == 'target']
        # X_train = poly_features.fit_transform(X_train)
        X_test = test_set.ix[:, test_set.columns != 'target']
        # X_test = poly_features.fit_transform(X_test)
        y_test = test_set.ix[:, test_set.columns == 'target']

        # Scaling X
        sc = StandardScaler()
        X_train = sc.fit_transform(X_train)
        X_test = sc.transform(X_test)

        # Need to remove 0 label

        y_train_0 = y_train[y_train['target'] == 0].index.tolist()
        y_test_0 = y_test[y_test['target'] == 0].index.tolist()

        y_train = y_train[y_train["target"] != 0]
        y_test = y_test[y_test["target"] != 0]

        X_train = np.delete(X_train, y_train_0, axis=0)
        X_test = np.delete(X_test, y_test_0, axis=0)

        # Starting training

        param_grid_log_reg = {'C': 2.0 ** np.arange(-3, 9)}
        gdLog = GridSearchCustomModel(LogisticRegression(penalty='l2', max_iter=2000, random_state=42), param_grid_log_reg)

        param_grid_rf = {'n_estimators': [15, 30, 50, 100, 120], 'max_depth': [4, 5, 7, 12, 15]
                         }
        gdRf = GridSearchCustomModel(RandomForestClassifier(n_jobs=-1, random_state=42), param_grid_rf)

        param_grid_svm = [
            {'C': 10.0 ** np.arange(-2, 3), 'kernel': ['linear']},
            {'C': 10.0 ** np.arange(-3, 4), 'gamma': 8.5 ** np.arange(-3, 3), 'kernel': ['rbf']},
        ]
        gdSVM = GridSearchCustomModel(SVC(probability=True, random_state=42), param_grid_svm)

        param_grid_GB = {'learning_rate': [0.1, 0.2, 0.5, 0.05], 'n_estimators': [10, 20, 50, 100], 'max_depth': [3, 5, 7, 10, 15]
                         }

        gdGB = GridSearchCustomModel(GradientBoostingClassifier(random_state=42, max_features='auto'), param_grid_GB)

        param_grid_ANN = {"hidden_layer_sizes": [(20,20), (15,), (8, 3), (5, 5), (10, 10), (40, 40), (10, 5),
                                                 (15, 13), (5, 10), (5,5,5), (10,10,10)],
                          'activation': ['tanh', 'relu'],
                          'alpha': 10.0 ** -np.arange(1, 7)}

        gdANN = GridSearchCustomModel(MLPClassifier(solver='lbfgs', random_state=42, verbose=False, max_iter=12000), param_grid_ANN)

        best_models = do_grid_search([ gdRf], X_train, y_train.values.ravel())

        for model in best_models:
            report.write_score(model, X_train, y_train, X_test, y_test)
            res = model.predict(X_test)
            report.write_result_in_pips_single_model(res.tolist(),
                                        gmt[start + train_len: start + train_len + test_len].tolist(),
                                        target_in_pips[start + train_len: start + train_len + test_len].tolist(), model.best_estimator_)

        report.write_result_in_pips_single_model([1] * test_len,
                                                 gmt[start + train_len: start + train_len + test_len].tolist(),
                                                 target_in_pips[
                                                 start + train_len: start + train_len + test_len].tolist(),
                                                 'real_price')
        y_test_pred, voting_classifier = report.write_combined_results(best_models, X_train, y_train, X_test, y_test)
        report.write_result_in_pips(y_test_pred.tolist(),
                                    gmt[start + train_len: start + train_len + test_len].tolist(),
                                    target_in_pips[start + train_len: start + train_len + test_len].tolist())
        start += test_len

    if args.predict == 'yes':
        to_predict = df_prediction.tail(1)
        to_predict = drop_column([to_predict], 'target_in_pips')[0]
        to_predict = drop_column([to_predict], 'target')[0]
        gmt = to_predict['Gmt time']
        to_predict = drop_column([to_predict], 'Gmt time')[0]
        X = sc.transform(to_predict)
        best_models = [model.best_estimator_ for model in best_models]
        best_models.append(voting_classifier)
        for model in best_models:
            report.write_predictions_next(model, X, gmt)
        report.write_prob_voting(voting_classifier, X)

    report.consolidate_feature_importance()
    report.close()
    print('end')
