import time

import os

from sklearn.ensemble import VotingClassifier
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score

from Ensemble.ensemble import EnsembleClassifier
from processing.processing import THRESHOLD
import datetime as dt
import matplotlib.pyplot as plt
import matplotlib.dates as mdates


class CustomReport(object):
    def __init__(self, pathToWrite, df, target_variable, train_len, pred):
        self.list_pips_cum = []
        self.path = pathToWrite
        self.df = df
        self.target_variable = target_variable
        self.total_pips = 0
        self.list_pips = []
        self.train_len = train_len
        self.pred = pred

    def close(self):
        self.file_descriptor.close()

    def write_step(self, start, train_len, test_len):
        self.file_descriptor.write(
            "\n\n\n   ******** Starting step: " + str(start) + "  --  " + str(train_len) + " ---- " + str(test_len))

    def init(self):
        # Creating file
        now = time.strftime("%c")
        directory = os.path.join(self.path, "reporting_train_len_" + str(self.train_len) + '_' + str(self.pred), now)
        if not os.path.exists(directory):
            os.makedirs(directory)
        self.file_path = os.path.join(directory, "reporting")
        self.file_descriptor = open(self.file_path, "a")
        self.file_descriptor.write("Reporting GridSearch: + " + now + "\n\n")
        self.file_descriptor.write("*** TARGET VARIABLE IS *** \n" + self.target_variable + "\n\n")
        self.file_descriptor.write("Dataframe has the following dimensions:\n")
        self.file_descriptor.write(str(self.df.shape))
        self.file_descriptor.write("\nIt contains the following features: \n")
        self.file_descriptor.write("\n".join(self.df.columns.tolist()))
        self.file_descriptor.write("\n\n\n Printing scores/best parameters for models\n")

    def write_feature_importance(self, model):
        self.file_descriptor.write("\n\n **** Writing out feature importance **** \n\n")
        cols = self.df.columns.tolist()
        feature_import = model.feature_importances_.tolist()
        cols.remove("target")
        new_list = [(importance, name) for name, importance in zip(cols, feature_import)]
        sorted_by_importance = sorted(new_list, key=lambda tup: tup[0], reverse=True)
        to_write = ["Feature: " + str(elem[1]) + "  " + str(elem[0]) for elem in sorted_by_importance]
        self.file_descriptor.write("\n".join(to_write))

    def write_combined_results(self, models, X_train, y_train, X_test, y_test):
        m = [(str(model.best_estimator_), model.best_estimator_) for model in models]
        model = VotingClassifier(estimators=m, voting='soft', n_jobs=-1)
        modelE = EnsembleClassifier(m)
        self.file_descriptor.write("\n\n Voting Classifier  \n\n")
        self.file_descriptor.write(" Writing prop:\n")
        self.file_descriptor.write(" ".join(str(x) for x in modelE.predict(X_test)))
        self.file_descriptor.write("\n\n Writing y test \n\n")
        self.file_descriptor.write(" ".join(str(x) for x in y_test.values.tolist()))
        self.file_descriptor.write("\n\n")
        # model = EnsembleClassifier([model.best_estimator_ for model in models])
        # Prediction on train set
        model.fit(X_train, y_train.values.ravel())
        y_train_pred = model.predict(X_train)
        conf_train = confusion_matrix(y_train.values.ravel(), y_train_pred)
        acc_train = accuracy_score(y_train.values.ravel(), y_train_pred)
        ps_train = precision_score(y_train.values.ravel(), y_train_pred)
        rec_train = recall_score(y_train.values.ravel(), y_train_pred)
        f1_train = f1_score(y_train.values.ravel(), y_train_pred)

        self.file_descriptor.write("\nconf train: \n")
        self.file_descriptor.write(str(conf_train))
        self.file_descriptor.write("\nacc_train: \t")
        self.file_descriptor.write(str(acc_train))
        self.file_descriptor.write("\n --------------- \n")
        self.file_descriptor.write("\nprecision_train: \t")
        self.file_descriptor.write(str(ps_train))
        self.file_descriptor.write("\n --------------- \n")
        self.file_descriptor.write("recall_train : \t")
        self.file_descriptor.write(str(rec_train))
        self.file_descriptor.write("\n --------------- \n")
        self.file_descriptor.write("\nf1_train : \t")
        self.file_descriptor.write(str(f1_train))

        # Evaluating on test set
        y_test_pred = model.predict(X_test)

        conf = confusion_matrix(y_test, y_test_pred)
        acc = accuracy_score(y_test, y_test_pred)
        ps = precision_score(y_test, y_test_pred)
        rec = recall_score(y_test, y_test_pred)
        f1 = f1_score(y_test, y_test_pred)
        self.file_descriptor.write("\n --------------- \n")
        self.file_descriptor.write("conf test: \n")
        self.file_descriptor.write(str(conf))
        self.file_descriptor.write("\n --------------- \n")
        self.file_descriptor.write("acc_test: \t")
        self.file_descriptor.write(str(acc))
        self.file_descriptor.write("\n --------------- \n")
        self.file_descriptor.write("precision_test: \t")
        self.file_descriptor.write(str(ps))
        self.file_descriptor.write("\n --------------- \n")
        self.file_descriptor.write("recall_test : \t")
        self.file_descriptor.write(str(rec))
        self.file_descriptor.write("\n --------------- \n")
        self.file_descriptor.write("f1_test : \t")
        self.file_descriptor.write(str(f1))
        self.file_descriptor.write("\n --------------- \n")
        return y_test_pred, model

    def write_result_in_pips(self, y_pred, gmt, target_in_pips):
        th = THRESHOLD
        for idx, val in enumerate(y_pred):
            y = val
            gm = gmt[idx]
            gm = gm.split(" ")[0]
            gm = dt.datetime.strptime(gm, '%Y-%m-%d').date()
            pips = target_in_pips[idx]

            if y > 0 and pips >= th or (y < 0 and pips < th):
                self.total_pips += pips
                self.list_pips.append((gm, abs(pips)))
                self.list_pips_cum.append((gm, self.total_pips))
            else:
                self.total_pips -= pips
                self.list_pips.append((gm, - abs(pips)))
                self.list_pips_cum.append((gm, self.total_pips))

            self.write_chart()

    def write_predictions_next(self, model, X, time):
        self.file_descriptor.write("\n\n\n Writing predictions for time: " + str(time))
        preds = model.predict(X)
        self.file_descriptor.write("Model: " + str(model))
        self.file_descriptor.write("\n Pred: " + str(preds))
        self.file_descriptor.write("\n Prob: " + str(model.predict_proba(X)))


    def write_score(self, model, X_train, y_train, X_test, y_test):

        self.file_descriptor.write("\n " + str(model.best_estimator_))

        if "RandomForest" in str(model.best_estimator_):
            self.write_feature_importance(model.best_estimator_)

        self.file_descriptor.write("\n\n The metrics for GridSearch is: " + model.scoring + "\n\n")
        self.file_descriptor.write("\nBest Score " + str(model.best_score_) + "\n")
        self.file_descriptor.write("\nBest Params:\t " + str(model.best_params_))

        # Prediction on train set
        y_train_pred = model.predict(X_train)
        conf_train = confusion_matrix(y_train.values.ravel(), y_train_pred)
        acc_train = accuracy_score(y_train.values.ravel(), y_train_pred)
        ps_train = precision_score(y_train.values.ravel(), y_train_pred)
        rec_train = recall_score(y_train.values.ravel(), y_train_pred)
        f1_train = f1_score(y_train.values.ravel(), y_train_pred)

        self.file_descriptor.write("\nconf train: \n")
        self.file_descriptor.write(str(conf_train))
        self.file_descriptor.write("\nacc_train: \t")
        self.file_descriptor.write(str(acc_train))
        self.file_descriptor.write("\n --------------- \n")
        self.file_descriptor.write("\nprecision_train: \t")
        self.file_descriptor.write(str(ps_train))
        self.file_descriptor.write("\n --------------- \n")
        self.file_descriptor.write("recall_train : \t")
        self.file_descriptor.write(str(rec_train))
        self.file_descriptor.write("\n --------------- \n")
        self.file_descriptor.write("\nf1_train : \t")
        self.file_descriptor.write(str(f1_train))

        # Evaluating on test set
        y_test_pred = model.predict(X_test)

        conf = confusion_matrix(y_test, y_test_pred)
        acc = accuracy_score(y_test, y_test_pred)
        ps = precision_score(y_test, y_test_pred)
        rec = recall_score(y_test, y_test_pred)
        f1 = f1_score(y_test, y_test_pred)
        self.file_descriptor.write("\n --------------- \n")
        self.file_descriptor.write("conf test: \n")
        self.file_descriptor.write(str(conf))
        self.file_descriptor.write("\n --------------- \n")
        self.file_descriptor.write("acc_test: \t")
        self.file_descriptor.write(str(acc))
        self.file_descriptor.write("\n --------------- \n")
        self.file_descriptor.write("precision_test: \t")
        self.file_descriptor.write(str(ps))
        self.file_descriptor.write("\n --------------- \n")
        self.file_descriptor.write("recall_test : \t")
        self.file_descriptor.write(str(rec))
        self.file_descriptor.write("\n --------------- \n")
        self.file_descriptor.write("f1_test : \t")
        self.file_descriptor.write(str(f1))
        self.file_descriptor.write("\n --------------- \n")

    def write_chart(self):
        x = [item[0] for item in self.list_pips_cum]
        y = [item[1] for item in self.list_pips_cum]
        plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%m.%d.%Y'))
        plt.gca().xaxis.set_major_locator(mdates.YearLocator())

        plt.plot(x, y)
        plt.ylabel('Gain (in pips)')
        plt.xlabel('Time')
        plt.gcf().autofmt_xdate()
        plt.savefig(self.file_path + " pict.png")
