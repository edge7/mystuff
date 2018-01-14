import time

import os

import pandas as pd
from sklearn.decomposition import PCA
from sklearn.ensemble import VotingClassifier
from sklearn.lda import LDA
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score

from Ensemble.ensemble import EnsembleClassifier
from processing.processing import THRESHOLD, AHEAD
import datetime as dt
import matplotlib.pyplot as plt
import matplotlib.dates as mdates


class CustomReport(object):
    def __init__(self, pathToWrite, df, target_variable, train_len, pred, pips, t, equity):
        self.counter = {}
        self.how_many_behind = {}
        self.last_prediction = {}
        self.list_pips_cum = []
        self.path = pathToWrite
        self.df = df
        self.target_variable = target_variable
        self.total_pips = 0
        self.list_pips = []
        self.list_pips_cum_s = {}
        self.total_pips_s = {}
        self.list_pips_s = {}
        self.train_len = train_len
        self.pred = pred
        self.p = pips
        self.t = t
        self.legend_vot = False
        self.legend_s = {}
        self.list_colors = ['k', 'g', 'r', 'c', 'y', 'm']
        self.dict_colors = {}
        self.feature_importance = {}
        self.equity = equity
        self.last_y_test_classifier = None
        self.last_mode_classifier = None

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

        x = self.t.tolist()
        try:
            x = [dt.datetime.strptime(gm, '%Y-%m-%d').date() for gm in x]
        except Exception as e:
            x = [dt.datetime.strptime(gm, '%m-%d-%Y').date() for gm in x]

            # y = self.p
            # plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%m.%d.%Y'))
            # plt.gca().xaxis.set_major_locator(mdates.YearLocator())
            # #
            # plt.plot(x, y, color='r', linewidth=0.2, linestyle='-', label='price')
            # plt.legend(loc='upper left')
            # plt.ylabel('Gain (in pips)')
            # plt.xlabel('Time')
            # plt.gcf().autofmt_xdate()
            # plt.savefig(self.file_path + " chart.png")

    def write_feature_importance(self, model):
        self.file_descriptor.write("\n\n **** Writing out feature importance **** \n\n")
        cols = self.df.columns.tolist()
        feature_import = model.feature_importances_.tolist()
        cols.remove("target")
        new_list = [(importance, name) for name, importance in zip(cols, feature_import)]
        sorted_by_importance = sorted(new_list, key=lambda tup: tup[0], reverse=True)
        to_write = ["Feature: " + str(elem[1]) + "  " + str(elem[0]) for elem in sorted_by_importance]
        self.file_descriptor.write("\n".join(to_write))
        for importance, name in sorted_by_importance:
            res = self.feature_importance.get(name, [])
            res.append(float(importance))
            self.feature_importance[name] = res

    def write_combined_results(self, models, X_train, y_train, X_test, y_test):

        exp = not self.newGenerated()
        # if exp is True, models have not been generated recently, so just return old predictions
        if exp and self.last_y_test_classifier is not None:
            self.file_descriptor.write("Model are not new, skipping classifier phase and return old results")
            return self.last_y_test_classifier, self.last_mode_classifier

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
        ps_train = precision_score(y_train.values.ravel(), y_train_pred, average="weighted")
        rec_train = recall_score(y_train.values.ravel(), y_train_pred, average="weighted")
        f1_train = f1_score(y_train.values.ravel(), y_train_pred, average="weighted")

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
        ps = precision_score(y_test, y_test_pred, average="weighted")
        rec = recall_score(y_test, y_test_pred, average="weighted")
        f1 = f1_score(y_test, y_test_pred, average="weighted")
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

        self.write_prob_voting(model, X_test)
        self.last_y_test_classifier = y_test_pred
        self.last_mode_classifier = model
        return y_test_pred, model

    # Used by classifier
    def write_result_in_pips(self, y_pred, gmt, target_in_pips):
        for idx, val in enumerate(y_pred):
            y = val
            gm = gmt[idx]
            try:
                tt = gm.split(" ")[0]
                gm = dt.datetime.strptime(tt, '%Y-%m-%d').date()
            except Exception as e:
                gm = dt.datetime.strptime(gm, '%m-%d-%Y').date()

            pips = target_in_pips[idx]
            counter = self.counter.get("VOTING", 0.0)
            counter += 1.0
            self.counter["VOTING"] = counter

            # OutOfMarket?
            if y != "OUT":

                if y == "BUY":
                    self.total_pips += pips
                    self.list_pips.append((gm, pips))

                if y == "SELL":
                    self.total_pips -= pips
                    self.list_pips.append((gm, -pips))

                self.list_pips_cum.append((gm, self.total_pips))
            else:
                self.list_pips.append((gm, 0.0))
                self.list_pips_cum.append((gm, self.total_pips))

        rate = self.total_pips / counter
        self.write_chart(label=" " + "{0:.4f}".format(rate) + "" + str(y))

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
        ps_train = precision_score(y_train.values.ravel(), y_train_pred, average="weighted")
        rec_train = recall_score(y_train.values.ravel(), y_train_pred, average="weighted")
        f1_train = f1_score(y_train.values.ravel(), y_train_pred, average="weighted")

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
        y_test_prob = model.predict_proba(X_test)

        conf = confusion_matrix(y_test, y_test_pred)
        acc = accuracy_score(y_test, y_test_pred)
        ps = precision_score(y_test, y_test_pred, average="weighted")
        rec = recall_score(y_test, y_test_pred, average="weighted")
        f1 = f1_score(y_test, y_test_pred, average="weighted")
        self.file_descriptor.write("\n --------------- \n")
        self.file_descriptor.write("PROBA TEST: " + str(y_test_prob))
        self.file_descriptor.write("\nconf test: \n")
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

    def write_chart(self, label=""):

        x = [item[0] for item in self.list_pips_cum]
        y = [item[1] for item in self.list_pips_cum]
        plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%m.%d.%Y'))
        plt.gca().xaxis.set_major_locator(mdates.YearLocator())

        plt.plot(x, y, 'b', linewidth=0.5, label='vot' + label)

        if not self.legend_vot:
            plt.legend(loc='upper left')
            self.legend_vot = True

        plt.ylabel('Gain (in pips)')
        plt.xlabel('Time')
        plt.gcf().autofmt_xdate()
        plt.legend(loc='upper left')
        plt.savefig(self.file_path + self.equity + ".png")
        # plt.close()
        plt.clf()

    def write_prob_voting(self, voting_classifier, X):
        probs = voting_classifier._collect_probas(X)
        self.file_descriptor.write("\n\n\n Writing Prob. voting classifier: \n")
        self.file_descriptor.write(str(probs))

    def hasExpired(self, model):
        res = self.how_many_behind.get(str(model.model)[0:5], 0)
        if res == 1:
            return True
        return False

    def newGenerated(self):
        for key in self.how_many_behind:
            res = self.how_many_behind.get(key, 0)
            if res == AHEAD:
                return True
        return False

    def write_result_in_pips_single_model(self, y_pred, gmt, target_in_pips, model):
        th = THRESHOLD
        for idx, val in enumerate(y_pred):
            y = val
            gm = gmt[idx]
            try:
                tt = gm.split(" ")[0]
                gm = dt.datetime.strptime(tt, '%Y-%m-%d').date()
            except Exception as e:
                gm = dt.datetime.strptime(gm, '%m-%d-%Y').date()

            pips = target_in_pips[idx]
            total_pips = self.total_pips_s.get(str(model)[0:5], 0.0)
            list_pips = self.list_pips_s.get(str(model)[0:5], [])
            list_pips_cum = self.list_pips_cum_s.get(str(model)[0:5], [])
            last_prediction = self.last_prediction.get(str(model)[0:5], 0.0)
            how_many_behind = self.how_many_behind.get(str(model)[0:5], 0)
            counter = self.counter.get(str(model)[0:5], 0.0)
            counter += 1.0
            self.counter[str(model)[0:5]] = counter
            how_many_behind -= 1

            # If timeout has expired use old prediction
            if how_many_behind <= 0:
                self.last_prediction[str(model)[0:5]] = y
                self.how_many_behind[str(model)[0:5]] = AHEAD
                self.file_descriptor.write('\n\n\n Prediction for: ' + str(model)[0:5] +
                                           ' has expired')
            else:
                self.how_many_behind[str(model)[0:5]] = how_many_behind
                y = last_prediction
                self.file_descriptor.write('\n\n\n Prediction for: ' + str(model)[0:5] +
                                           ' has not expired, still  ' + str(how_many_behind) + ' times')

            # OutOfMarket?
            if y != "OUT":

                # if y == "BUY" and pips >= th or (y == "SELL" and pips < th):
                #     total_pips += abs(pips)
                #     list_pips.append((gm, abs(pips)))
                #     list_pips_cum.append((gm, total_pips))
                # else:
                #     total_pips -= abs(pips)
                #     list_pips.append((gm, - abs(pips)))
                #     list_pips_cum.append((gm, total_pips))
                if y == "BUY":
                    total_pips += pips
                    list_pips.append((gm, pips))

                if y == "SELL":
                    total_pips -= pips
                    list_pips.append((gm, -pips))

                list_pips_cum.append((gm, total_pips))
            else:
                list_pips.append((gm, 0.0))
                list_pips_cum.append((gm, total_pips))

            pips_a_time = total_pips / counter
            k = str(model)[0:5]

            self.file_descriptor.write("\n\nStats per  " + k + ":\n")
            self.file_descriptor.write("PIPS A TIME: " + str(pips_a_time) + "\n")
            self.file_descriptor.write("TOTAL PIPS: " + str(total_pips) + "\n\n")

            self.total_pips_s[k] = total_pips
            self.list_pips_s[k] = list_pips
            self.list_pips_cum_s[k] = list_pips_cum

        self.write_chart_single_model(k, label=" " + "{0:.4f}".format(pips_a_time) + "" + str(y))

    def write_chart_single_model(self, k, label=""):
        color = self.dict_colors.get(k, None)
        if color is None:
            color = self.list_colors[0]
            del self.list_colors[0]
            self.dict_colors[k] = color

        x = [item[0] for item in self.list_pips_cum_s[k]]
        y = [item[1] for item in self.list_pips_cum_s[k]]
        plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%m.%d.%Y'))
        plt.gca().xaxis.set_major_locator(mdates.YearLocator())

        plt.plot(x, y, color, linewidth=0.5, label=k + label)

        if not self.legend_s.get(k, False):
            self.legend_s[k] = True

        plt.ylabel('Gain (in pips)')
        plt.xlabel('Time')
        plt.gcf().autofmt_xdate()
        # plt.savefig(self.file_path + " pict.png")

    def consolidate_feature_importance(self):
        l = []
        for name, lis in self.feature_importance.items():
            tot = float(sum(lis))
            count = float(len(lis))
            mean = tot / count
            l.append((name, mean))

        tw = sorted(l, key=lambda tup: tup[1], reverse=True)
        to_write = ''
        for t in tw:
            to_write += str(t[0]) + ':   ' + str(t[1])
            to_write += '\n'
        self.file_descriptor.write('\n\n\n\n\n\n\n\n Writing Feature IMPORTANCE:\n\n')
        self.file_descriptor.write(to_write)

    def write_2d(self, X_train, y_train, X_test):

        X = X_train
        y = y_train
        lda = LDA(n_components=2)  # 2-dimensional LDA
        lda_transformed = pd.DataFrame(lda.fit_transform(X_train, y))

        # Plot all three series
        plt.scatter(lda_transformed[(y == "OUT")["target"].tolist()][0],
                    lda_transformed[(y == "OUT")["target"].tolist()][1], label='OUT', c='red')
        plt.scatter(lda_transformed[(y == "BUY")["target"].tolist()][0],
                    lda_transformed[(y == "BUY")["target"].tolist()][1], label='BUY', c='blue')
        plt.scatter(lda_transformed[(y == "SELL")["target"].tolist()][0],
                    lda_transformed[(y == "SELL")["target"].tolist()][1], label='SELL',
                    c='lightgreen')

        xx = lda.transform(X_test)
        plt.scatter(xx[0][0],
                    xx[0][1], label='AHEAD',
                    c='black')

        # Display legend and show plot
        plt.legend(loc=3)
        plt.savefig(self.file_path + " lda.png")
        plt.clf()
        lda = PCA(n_components=2)  # 2-dimensional PCA
        lda_transformed = pd.DataFrame(lda.fit_transform(X_train, y))

        # Plot all three series
        plt.scatter(lda_transformed[(y == "OUT")["target"].tolist()][0],
                    lda_transformed[(y == "OUT")["target"].tolist()][1], label='OUT', c='red')
        plt.scatter(lda_transformed[(y == "BUY")["target"].tolist()][0],
                    lda_transformed[(y == "BUY")["target"].tolist()][1], label='BUY', c='blue')
        plt.scatter(lda_transformed[(y == "SELL")["target"].tolist()][0],
                    lda_transformed[(y == "SELL")["target"].tolist()][1], label='SELL',
                    c='lightgreen')

        xx = lda.transform(X_test)
        plt.scatter(xx[0][0],
                    xx[0][1], label='AHEAD',
                    c='black')
        # Display legend and show plot
        plt.legend(loc=3)
        plt.savefig(self.file_path + "PCA.png")
