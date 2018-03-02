import random

import numpy as np
import pandas as pd
from arch import arch_model
from numpy import nan, array
from sklearn import linear_model
from sklearn.preprocessing import StandardScaler

from algos.algos import adf

SUBSET_SIZE = 1000
THRESHOLD = 0
AHEAD = 6 * 2


def apply_candle(row, toUse):
    close = row["Close_" + toUse]
    high = row["High_" + toUse]
    low = row["Low_" + toUse]
    open = row["Open_" + toUse]

    # It is green?
    if close - open >= 0:
        high_in_pips = high - close
        body_in_pips = close - open
        low_in_pips = open - low
    else:
        high_in_pips = open - high
        body_in_pips = close - open
        low_in_pips = low - close

    return [abs(high_in_pips), body_in_pips, abs(low_in_pips)]


def create_cols(l):
    body = []
    high = []
    low = []
    body_before = []
    high_before = []
    low_before = []
    body_before_before = []
    high_before_before = []
    low_before_before = []
    body_before.append(nan)
    high_before.append(nan)
    low_before.append(nan)
    body_before_before.append(nan)
    high_before_before.append(nan)
    low_before_before.append(nan)
    body_before_before.append(nan)
    high_before_before.append(nan)
    low_before_before.append(nan)
    # Iterate from the first to the last
    old_i = None
    for idx, i in enumerate(l):
        if old_i is not None:
            high_before.append(old_i[0])
            low_before.append(old_i[2])
            body_before.append(old_i[1])

        old_i = i

        if idx >= 2:
            high_before_before.append(l[idx - 2][0])
            low_before_before.append(l[idx - 2][2])
            body_before_before.append(l[idx - 2][1])
        high.append(i[0])
        body.append(i[1])
        low.append(i[2])

    if old_i is not None:
        high_before.append(old_i[0])
        low_before.append(old_i[2])
        body_before.append(old_i[1])
    if idx >= 2:
        high_before_before.append(l[idx - 2][0])
        low_before_before.append(l[idx - 2][2])
        body_before_before.append(l[idx - 2][1])

    return [(pd.Series(high), pd.Series(high_before), pd.Series(high_before_before)),
            (pd.Series(body), pd.Series(body_before), pd.Series(body_before_before)), (pd.Series(low),
                                                                                       pd.Series(low_before),
                                                                                       pd.Series(low_before_before))]


def apply_low(row, toUse):
    body = row["BodyInPips"]
    open = row["Open_" + toUse]
    low = row["Low_" + toUse]
    close = row["Close_" + toUse]
    if body > 0:
        low_in_pips = open - low
    else:
        low_in_pips = close - low
    return low_in_pips


def apply_mv_avg_to(df, l, window=5):
    for col in df:
        if any(ext in col for ext in l):
            continue
        df[col + "_AVG_" + str(window)] = df[col].ewm(span=window, adjust=False).mean()
    return df


def apply_mv_avg_just_to(df, l, window=5):
    for col in df:
        if any(ext in col for ext in l) and "AVG" not in col:
            df[col + "_AVG_" + str(window)] = df[col].ewm(span=window, adjust=False).mean()
    return df


def apply_lowH(row, toUse):
    body = row["BodyInPipsHeiken"]
    open = row["Open_" + toUse]
    low = row["Low_" + toUse]
    close = row["Close_" + toUse]
    if body > 0:
        low_in_pips = open - low
    else:
        low_in_pips = close - low
    return low_in_pips


def apply_highH(row, toUse):
    body = row["BodyInPipsHeiken"]
    open = row["Open_" + toUse]
    close = row["Close_" + toUse]
    high = row["High_" + toUse]
    if body > 0:
        high_in_pips = high - close
    else:
        high_in_pips = high - open
    return high_in_pips


def apply_high(row, toUse):
    body = row["BodyInPips"]
    open = row["Open_" + toUse]
    close = row["Close_" + toUse]
    high = row["High_" + toUse]
    if body > 0:
        high_in_pips = high - close
    else:
        high_in_pips = high - open
    return high_in_pips


def add_candlestick_columns_2(df, toUse):
    df["BodyInPips"] = - df["Open_" + toUse] + df["Close_" + toUse]
    df["LowInPips"] = df.apply(lambda row: apply_low(row, toUse), axis=1)
    df["HighInPips"] = df.apply(lambda row: apply_high(row, toUse), axis=1)
    df["High/Body"] = df["HighInPips"] / df["BodyInPips"]
    df["Low/Body"] = df["LowInPips"] / df["BodyInPips"]
    df["High/BodyP"] = df["High/Body"].shift(1)
    df["Low/BodyP"] = df["Low/Body"].shift(1)
    df["BodyInPipsP"] = df["BodyInPips"].shift(1)
    df["Body/BodyP"] = pd.Series.abs(df["BodyInPips"] / df["BodyInPipsP"])
    df["Body/BodyPP"] = df["Body/BodyP"].shift(1)
    return df


def add_candlestick_columns(df, toUse):
    ret = df.apply(lambda row: apply_candle(row, toUse), axis=1)
    ret = create_cols(ret)
    high_in_pips = ret[0][0]
    body_in_pips = ret[1][0]
    low_in_pips = ret[2][0]
    df[toUse + "_High_in_pips"] = high_in_pips
    df[toUse + "_Body_in_pips"] = body_in_pips
    df[toUse + "_Low_in_pips"] = low_in_pips
    df[toUse + "_High/Body"] = high_in_pips / body_in_pips
    df[toUse + "_Low/body"] = low_in_pips / body_in_pips
    # df[toUse + "_Low/High"] = abs(low_in_pips / high_in_pips)
    # df[toUse + "_High-Low"] = abs(high_in_pips - low_in_pips)
    # df[toUse + "_High-body"] = abs(high_in_pips) - abs(body_in_pips)
    # df[toUse + "_Low-body"] = abs(low_in_pips) - abs(body_in_pips)
    high_in_pips = ret[0][1]
    body_in_pips = ret[1][1]
    low_in_pips = ret[2][1]
    df[toUse + "_High_in_pips_bef"] = high_in_pips
    df[toUse + "_Body_in_pips_bef"] = body_in_pips
    df[toUse + "_Low_in_pips_bef"] = low_in_pips
    # df[toUse + "_High-body_bef"] = abs(high_in_pips) - abs(body_in_pips)
    # df[toUse + "_Low-body_bef"] = abs(low_in_pips) - abs(body_in_pips)
    df[toUse + "_High/Body_bef"] = high_in_pips / body_in_pips
    df[toUse + "_Low/body_bef"] = low_in_pips / body_in_pips
    # df[toUse + "_Low/High_bef"] = abs(low_in_pips / high_in_pips)
    # df[toUse + "_High-Low_bef"] = abs(high_in_pips - low_in_pips)

    high_in_pips = ret[0][2]
    body_in_pips = ret[1][2]
    low_in_pips = ret[2][2]
    df[toUse + "_High_in_pips_bef_bef"] = high_in_pips
    df[toUse + "_Body_in_pips_bef_bef"] = body_in_pips
    df[toUse + "_Low_in_pips_bef_bef"] = low_in_pips

    df[toUse + "_CurrentBody/Body_bef"] = abs(df[toUse + "_Body_in_pips"]) / abs(df[toUse + "_Body_in_pips_bef"])
    df[toUse + "_CurrentHigh/High_bef"] = abs(df[toUse + "_High_in_pips"]) / abs(df[toUse + "_High_in_pips_bef"])
    df[toUse + "_CurrentLow/Low_bef"] = abs(df[toUse + "_Low_in_pips"]) / abs(df[toUse + "_Low_in_pips_bef"])
    return df


def transform_columns_names(df, crossList, path, excluded, keep_names=False):
    toUse = crossList[0]
    for j in crossList:
        if j in str(path):
            toUse = j

    for col in df:
        if excluded not in col:
            df[col + "_" + toUse] = df[col]
            # df[col + "_" + toUse] = df[col + "_" + toUse].rolling(window=10).mean()
            del df[col]

    df = add_candlestick_columns_2(df, toUse)

    return df


def drop_original_values(df, cross_list):
    for toUse in cross_list:
        del df["Open_" + toUse]
        del df["Close_" + toUse]
        del df["High_" + toUse]
        del df["Low_" + toUse]
    return df


def der(x):
    l = x.tolist()
    values = []
    acc = 0
    for index, value in enumerate(l):
        if index == 0:
            continue
        values.append(l[index] - l[index - 1])
        acc += l[index] - l[index - 1]

    return abs(acc / len(l))


def merge_close_values(values):
    new_values = []
    values.sort()
    for index, v in enumerate(values):
        if index == 0:
            continue
        if values[index] - values[index - 1] < 2:
            new_values = new_values[:-1]

        new_values.append(v)
    return new_values


def sup_and_res(df, target, window=30):
    rows = df.shape[0]
    start = 0
    done = False
    while not done:
        x = df.iloc[start: start + window]
        res = apply_supp_and_res(x, target)
        start += 1
        index = res[0]
        closest_res = res[1]
        closest_supp = res[2]
        if start + window > rows:
            done = True
        df.set_value(index, 'closest_res', closest_res)
        df.set_value(index, 'closest_sup', closest_supp)
        df.set_value(index, 'diff_clos_sup', closest_supp - closest_res)
    return df


def apply_supp_and_res(df, target):
    last_close = df.tail(1)["Close_" + target]
    index = last_close.index.values[0]
    last_close = last_close.tolist()[0]

    close = df["Close_" + target]

    close_smooth = close.rolling(window=1).mean()

    d_1 = close_smooth.rolling(window=2).apply(lambda x: der(x))
    d_2 = close_smooth.rolling(window=3).apply(lambda x: der(x))

    values_1 = d_1.sort_values().index.tolist()[0:10]
    values_2 = d_2.sort_values().index.tolist()[0:10]
    values = list(set(values_1 + values_2))
    values.sort()
    values = merge_close_values(values)

    real_values = []
    for v in values:
        real_values.append(close[v])

    closest_supp = -10000000000
    closest_res = 1000000000000
    for v in real_values:
        # possible res
        if last_close < v < closest_res:
            closest_res = v
        if last_close > v > closest_supp:
            closest_supp = v

    if closest_supp == -10000000000:
        closest_supp = last_close

    if closest_res == 1000000000000:
        closest_res = last_close
    assert closest_res >= last_close
    assert closest_supp <= last_close
    return index, closest_res - last_close, closest_supp - last_close


def apply_df_test(df, target):
    df["Close_" + target + 'adf_100'] = df["Close_" + target].rolling(window=100).apply(lambda x: adf(x))
    df["Close_" + target + 'adf_40'] = df["Close_" + target].rolling(window=40).apply(lambda x: adf(x))
    df["Close_" + target + 'adf_20'] = df["Close_" + target].rolling(window=20).apply(lambda x: adf(x))
    # df[target + 'adf_25'] = df[target].rolling(window=25).apply(lambda x: adf(x))
    return df


def create_dataframe(flist, excluded, crossList, keep_names=True):
    l = list()
    for path in flist:
        df = pd.read_csv(str(path), delimiter=";")
        if 'Adj Close' in df:
            del df['Adj Close']
        cols = list(df.columns.values)
        for i in cols:
            if i == "Gmt time":
                continue
                # if i == "Close":
                # df[i + "_original - avg"] = df[i] - df[i].ewm(span=5, adjust=False).mean()
                # df[i + "_original - avg"] = df[i] - df[i].rolling(window=4).mean()
            if "Close" in i:
                df[i] = df[i].rolling(window=6).mean()
        else:
            df[i] = df[i].rolling(window=2).mean()

        df['Gmt time'] = df['Gmt time'].apply(lambda x: x.replace(",", " "))
        # df = df.head(75).reset_index()
        if 'Volume' in df:
            df = df[df["Volume"] != 0.000].reset_index()
        df = transform_columns_names(df, crossList, path, excluded, keep_names=keep_names)
        l.append(df)
    return l


def apply_distance_from_max(df, target, window=50):
    df['dist_from_max_' + str(window)] = df[target] - df[target].rolling(window=window).apply(lambda x: np.max(x))
    return df


def momentum(x):
    return 100.0 * x[-1] / x[0]


def apply_momentum(df, target, window=8):
    df['momentum_' + str(window)] = df[target].rolling(window=window).apply(lambda x: momentum(x))
    return df


def PROC(x):
    return x[-1] - x[0]


def apply_PROC(df, target, window=10):
    df["PROC_" + str(window)] = df["Close_" + target].rolling(window=window).apply(lambda x: PROC(x))
    return df


def apply_distance_from_min(df, target, window=50):
    df['dist_from_min_' + str(window)] = df[target] - df[target].rolling(window=window).apply(lambda x: np.min(x))
    return df


def modify_time(list_dataframes):
    for l in list_dataframes:
        l.ix[:, 0] = l.ix[:, 0].map(lambda a: float(a.split(" ")[1].split(":")[0]))
    return list_dataframes


def print_types(df):
    print(df.columns.to_series().groupby(df.dtypes).groups)


def join_dfs(dfs, column):
    res = None
    for i in dfs:
        if res is None:
            res = i
        else:
            res = pd.merge(res, i, on=column, how='inner')
    return res


def drop_column(dfs, column):
    for i in dfs:
        for col in i:
            if column in col:
                del i[col]
    return dfs


def drop_columns(df, columns):
    for col in df:
        for ic in columns:
            if ic == col:
                del df[ic]
    return df


def apply_diff(df, excluded, shift=80):
    for col in df:
        if any(ext in col for ext in excluded):
            continue
        df[col + "_diff_shift_" + str(shift)] = df[col].diff(periods=shift)
    return df


def apply_shift(df, excluded):
    for i in [6 * 5]:
        for col in df:
            if any(ext in col for ext in excluded):
                continue
            df[col + "_SHIFT_" + str(i)] = df[col].shift(i)
    return df


def apply_shift_just_at(df, included):
    it = df.copy()
    for i in [6 * 5, 6 * 10]:
        for col in it:
            if any(ext in col for ext in included):
                df[col + "_SHIFT_" + str(i)] = df[col].shift(i)
                df[col + "_SHIFT_DIFF_" + str(i)] = df[col].diff(periods=i)
    return df


def apply_ichimo(df, t):
    # Kijun
    # averaging the highest high and the lowest low for the past 26/9 periods
    for col in df:
        if 'High_' + t == col:
            high26 = df[col].rolling(window=26).max()
            high9 = df[col].rolling(window=9).max()

        if 'Low_' + t == col:
            low26 = df[col].rolling(window=26).min()
            low9 = df[col].rolling(window=9).min()

    kijun = (high26 + low26) / 2.0
    tenkan = (high9 + low9) / 2.0

    avg = (kijun + tenkan) / 2.0

    # df["kijun"] = df["Close_" + t] - kijun
    # df["tenkan"] = df["Close_" + t] - tenkan
    # df["kijun_tenkan_avg"] = df["Close_" + t] - avg
    df["kijun-tenkan"] = kijun - tenkan
    return df


def lin_reg_dist(x, window):
    sc = StandardScaler()
    x = sc.fit_transform(x.reshape(-1, 1))
    line = linear_model.LinearRegression(fit_intercept=False)
    f = line.fit(array(range(1, window + 1)).reshape(-1, 1), x)
    # y = f.predict(x[-1])
    return f.coef_[0]


def apply_linear_regression(df, target, window=50):
    df["dist_from_lin_reg_" + str(window)] = df[target].rolling(window=window).apply(lambda x: lin_reg_dist(x, window))
    df["dist_from_lin_reg_" + str(window * 2)] = df[target].rolling(window=window * 2).apply(
        lambda x: lin_reg_dist(x, window * 2))

    df["dist_from_lin_reg_" + str(window * 3)] = df[target].rolling(window=window * 3).apply(
        lambda x: lin_reg_dist(x, window * 3))
    return df


def apply_linear_regression_on_max(df, target, window_m=10, window=50):
    df["dist_from_lin_reg_max_" + str(window) + "_" + str(window_m)] = df[target].rolling(
        window=window_m).max().rolling(window=window).apply(
        lambda x: lin_reg_dist(x, window))
    df["dist_from_lin_reg_max_" + str(window * 2) + "_" + str(window_m)] = df[target].rolling(
        window=window_m).max().rolling(
        window=window * 2).apply(lambda x: lin_reg_dist(x, window * 2))

    df["dist_from_lin_reg_max_" + str(window * 3) + "_" + str(window_m)] = df[target].rolling(
        window=window_m).max().rolling(
        window=window * 3).apply(lambda x: lin_reg_dist(x, window * 3))
    return df


def apply_linear_regression_on_min(df, target, window_m=10, window=50):
    df["dist_from_lin_reg_min_" + str(window) + "_" + str(window_m)] = df[target].rolling(
        window=window_m).min().rolling(window=window).apply(
        lambda x: lin_reg_dist(x, window))
    df["dist_from_lin_reg_min_" + str(window * 2) + "_" + str(window_m)] = df[target].rolling(
        window=window_m).min().rolling(
        window=window * 2).apply(lambda x: lin_reg_dist(x, window * 2))

    df["dist_from_lin_reg_min_" + str(window * 3) + "_" + str(window_m)] = df[target].rolling(
        window=window_m).min().rolling(
        window=window * 3).apply(lambda x: lin_reg_dist(x, window * 3))
    return df


def create_dataframe_for_stacking(X_train_stacking, m, X_train):
    len = X_train.shape[0]
    len_stacking = X_train_stacking.shape[0]
    assert len == len_stacking
    for i in range(0, len):
        for model in m:
            model_ = model[1]
            name_model = model[0][0:5]
            features = X_train[i].reshape(1, -1)
            predictions = model_.predict_proba(features).tolist()[0]
            classes = list(model_.classes_)
            for p, c in zip(predictions, classes):
                X_train_stacking.set_value(i, str(name_model) + "_" + str(c), float(p))
    return X_train_stacking


def create_target_ahead(df, CLOSE_VARIABLE, AHEAD, threshold):
    number_rows = df.shape[0]
    counter = 0
    # Iterating row by row, check change after AHEAD candles
    for index in range(number_rows):
        start = df.iloc[[index]][CLOSE_VARIABLE][index]
        to_set = 0
        to_set_ = "OUT"
        end = min(number_rows, index + AHEAD + 1)
        if counter == 0:
            for next in range(index + 1, end):
                value = df.iloc[[next]][CLOSE_VARIABLE][next]
                pctg = value - start
                if pctg > threshold:
                    to_set += pctg
                if pctg < - threshold:
                    to_set += pctg

            if to_set > 0:
                to_set_ = "BUY"
            if to_set < 0:
                to_set_ = "SELL"
            df.set_value(index, 'target', to_set_)
            counter = round(AHEAD / 6)
        else:
            counter -= 1
            df.set_value(index, 'target', nan)

    df.set_value(number_rows - 1, 'target', "OUT")
    print("Ended Target AHEAD")
    print(df.shape)
    df = df[df["target"].notnull()]
    print(df.shape)
    return df


def normalized(x, axis=-1, order=2):
    sc = StandardScaler()
    x = sc.fit_transform(x.reshape(-1, 1))
    return x


def poly(x, coef, degree):
    import numpy.polynomial.polynomial as poly
    coefs = poly.polyfit(range(x.size), normalized(x), degree)
    return coefs[coef]


def apply_coefs_behind(df, target, window=50):
    window = ""
    df["coef_behind_500_400_1" + str(window)] = df[target].rolling(window=500).apply(lambda x: x).head(100).apply(
        lambda x: poly(x, 1, 2))
    df["coef_behind_400_300_1" + str(window)] = df[target].rolling(window=400).head(100).apply(lambda x: poly(x, 1, 2))
    df["coef_behind_300_200_1" + str(window)] = df[target].rolling(window=300).head(100).apply(lambda x: poly(x, 1, 2))
    df["coef_behind_200_100_1" + str(window)] = df[target].rolling(window=200).head(100).apply(lambda x: poly(x, 1, 2))
    df["coef_behind_100_0_1" + str(window)] = df[target].rolling(window=100).head(100).apply(lambda x: poly(x, 1, 2))

    df["coef_behind_500_400_2" + str(window)] = df[target].rolling(window=500).head(100).apply(lambda x: poly(x, 2, 2))
    df["coef_behind_400_300_2" + str(window)] = df[target].rolling(window=400).head(100).apply(lambda x: poly(x, 2, 2))
    df["coef_behind_300_200_2" + str(window)] = df[target].rolling(window=300).head(100).apply(lambda x: poly(x, 2, 2))
    df["coef_behind_200_100_2" + str(window)] = df[target].rolling(window=200).head(100).apply(lambda x: poly(x, 2, 2))
    df["coef_behind_100_0_2" + str(window)] = df[target].rolling(window=100).head(100).apply(lambda x: poly(x, 2, 2))
    return df


def apply_poly_on_max(df, t, windows_m=5, windows=5):
    df["poly_1_3" + str(windows) + "_MAX_" + str(windows_m)] = df["Close_" + t].rolling(window=windows_m).max().rolling(
        window=windows).apply(
        lambda x: poly(x, 1, 3))
    df["poly_2_3" + str(windows) + "_MAX_" + str(windows_m)] = df["Close_" + t].rolling(window=windows_m).max().rolling(
        window=windows).apply(
        lambda x: poly(x, 2, 3))
    df["poly_3_3" + str(windows) + "_MAX_" + str(windows_m)] = df["Close_" + t].rolling(window=windows_m).max().rolling(
        window=windows).apply(
        lambda x: poly(x, 3, 3))

    df["poly_1_4" + str(windows) + "_MAX_" + str(windows_m)] = df["Close_" + t].rolling(window=windows_m).max().rolling(
        window=windows).apply(
        lambda x: poly(x, 1, 4))
    df["poly_2_4" + str(windows) + "_MAX_" + str(windows_m)] = df["Close_" + t].rolling(window=windows_m).max().rolling(
        window=windows).apply(
        lambda x: poly(x, 2, 4))
    df["poly_3_4" + str(windows) + "_MAX_" + str(windows_m)] = df["Close_" + t].rolling(window=windows_m).max().rolling(
        window=windows).apply(
        lambda x: poly(x, 3, 4))
    return df


def apply_poly_on_min(df, t, windows_m=5, windows=5):
    df["poly_1_3" + str(windows) + "_min_" + str(windows_m)] = df["Close_" + t].rolling(window=windows_m).min().rolling(
        window=windows).apply(
        lambda x: poly(x, 1, 3))
    df["poly_2_3" + str(windows) + "_min_" + str(windows_m)] = df["Close_" + t].rolling(window=windows_m).min().rolling(
        window=windows).apply(
        lambda x: poly(x, 2, 3))
    df["poly_3_3" + str(windows) + "_min_" + str(windows_m)] = df["Close_" + t].rolling(window=windows_m).min().rolling(
        window=windows).apply(
        lambda x: poly(x, 3, 3))

    df["poly_1_4" + str(windows) + "_min_" + str(windows_m)] = df["Close_" + t].rolling(window=windows_m).min().rolling(
        window=windows).apply(
        lambda x: poly(x, 1, 4))
    df["poly_2_4" + str(windows) + "_min_" + str(windows_m)] = df["Close_" + t].rolling(window=windows_m).min().rolling(
        window=windows).apply(
        lambda x: poly(x, 2, 4))
    df["poly_3_4" + str(windows) + "_min_" + str(windows_m)] = df["Close_" + t].rolling(window=windows_m).min().rolling(
        window=windows).apply(
        lambda x: poly(x, 3, 4))
    return df


def apply_poly(df, t, windows=5):
    df["poly_1_3" + str(windows)] = df["Close_" + t].rolling(window=windows).apply(lambda x: poly(x, 1, 3))
    df["poly_2_3" + str(windows)] = df["Close_" + t].rolling(window=windows).apply(lambda x: poly(x, 2, 3))
    df["poly_3_3" + str(windows)] = df["Close_" + t].rolling(window=windows).apply(lambda x: poly(x, 3, 3))

    df["poly_1_4" + str(windows)] = df["Close_" + t].rolling(window=windows).apply(lambda x: poly(x, 1, 4))
    df["poly_2_4" + str(windows)] = df["Close_" + t].rolling(window=windows).apply(lambda x: poly(x, 2, 4))
    df["poly_3_4" + str(windows)] = df["Close_" + t].rolling(window=windows).apply(lambda x: poly(x, 3, 4))
    return df


def apply_williams(df, t, window=14):
    for col in df:
        if 'High_' + t == col:
            high = df[col].rolling(window=window).max()
        if 'Low_' + t == col:
            low = df[col].rolling(window=window).min()

    df["Williams_" + str(window)] = -100.0 * (high - df["Close_" + t]) / (high - low)
    return df


def apply_stochastic(df, t, window=14, mean=3):
    for col in df:
        if 'High_' + t == col:
            high = df[col].rolling(window=window).max()
        if 'Low_' + t == col:
            low = df[col].rolling(window=window).min()

    fast_stoc = 100.0 * (df["Close_" + t] - low) / (high - low)
    slow_stoc = fast_stoc.rolling(window=mean).mean()
    df["fast_stoc_" + str(window)] = fast_stoc
    if mean != 1:
        df["slow_stoc_" + str(window)] = slow_stoc
    # df["fast-slow_stoc"] = fast_stoc - slow_stoc
    return df


def gain(x, window):
    sum = 0.0
    for i in x.tolist():
        if i > 0:
            sum += i
    return sum / window


def loss(x, window):
    loss = 0.0
    for i in x.tolist():
        if i < 0:
            loss += abs(i)
    return loss / window


def apply_rsi(df, t, window=14):
    g = df[t + "BodyInPips"].rolling(window=window).apply(lambda x: gain(x, window))
    l = df[t + "BodyInPips"].rolling(window=window).apply(lambda x: loss(x, window))
    df["RSI" + str(window)] = 100.0 - (100 / (1 + (g / l)))
    return df


def apply_CCI(df, t, window=10):
    typicalPrice = (df["High_" + t] + df["Low_" + t] + df["Close_" + t]) / 3.0
    typicalPriceSma = typicalPrice.rolling(window=window).mean()
    meanDeviation = typicalPrice.rolling(window=window).std()
    df["CCI_" + str(window)] = (typicalPrice - typicalPriceSma) / (0.015 * meanDeviation)
    return df


def apply_HeikenAshi(df, t):
    df["Close_Heiken"] = (df["Open_" + t] + df["High_" + t] + df["Low_" + t] + df["Close_" + t]) / 4.0
    df["Open_Heiken"] = (df["Open_" + t].shift(1) + df["Close_" + t].shift(1)) / 2.0
    df["High_Heiken"] = df[["Close_" + t, "Open_Heiken", "High_" + t]].max(axis=1)
    df["Low_Heiken"] = df[["Low_" + t, "Open_Heiken", "High_Heiken"]].min(axis=1)
    df["BodyInPipsHeiken"] = - df["Open_Heiken"] + df["Close_Heiken"]
    df["LowInPipsHeiken"] = df.apply(lambda row: apply_lowH(row, "Heiken"), axis=1)
    df["HighInPipsHeiken"] = df.apply(lambda row: apply_highH(row, "Heiken"), axis=1)
    del df["Close_Heiken"]
    del df["Open_Heiken"]
    del df["High_Heiken"]
    del df["Low_Heiken"]
    return df


def fourier_series_coeff_numpy(f, T, N, return_complex=False):
    """Calculates the first 2*N+1 Fourier series coeff. of a periodic function.

    Given a periodic, function f(t) with period T, this function returns the
    coefficients a0, {a1,a2,...},{b1,b2,...} such that:

    f(t) ~= a0/2+ sum_{k=1}^{N} ( a_k*cos(2*pi*k*t/T) + b_k*sin(2*pi*k*t/T) )

    If return_complex is set to True, it returns instead the coefficients
    {c0,c1,c2,...}
    such that:

    f(t) ~= sum_{k=-N}^{N} c_k * exp(i*2*pi*k*t/T)

    where we define c_{-n} = complex_conjugate(c_{n})

    Refer to wikipedia for the relation between the real-valued and complex
    valued coeffs at http://en.wikipedia.org/wiki/Fourier_series.

    Parameters
    ----------
    f : the periodic function, a callable like f(t)
    T : the period of the function f, so that f(0)==f(T)
    N_max : the function will return the first N_max + 1 Fourier coeff.

    Returns
    -------
    if return_complex == False, the function returns:

    a0 : float
    a,b : numpy float arrays describing respectively the cosine and sine coeff.

    if return_complex == True, the function returns:

    c : numpy 1-dimensional complex-valued array of size N+1

    """
    # From Shanon theoreom we must use a sampling freq. larger than the maximum
    # frequency you want to catch in the signal.
    f_sample = 2 * N
    # we also need to use an integer sampling frequency, or the
    # points will not be equispaced between 0 and 1. We then add +2 to f_sample
    t, dt = np.linspace(0, T, f_sample + 2, endpoint=False, retstep=True)

    y = np.fft.rfft(f(t)) / t.size

    if return_complex:
        return y
    else:
        y *= 2
        return y[0].real, y[1:-1].real, -y[1:-1].imag


def fourierSeries(period, N, toReturn):
    """Calculate the Fourier series coefficients up to the Nth harmonic"""
    sin = []
    cos = []
    T = len(period)
    t = np.arange(T)
    for n in range(N + 1):
        an = 2 / T * (period * np.cos(2 * np.pi * n * t / T)).sum()
        bn = 2 / T * (period * np.sin(2 * np.pi * n * t / T)).sum()
        sin.append(an)
        cos.append(bn)
    w = toReturn.split(".")

    if w[0] == "SIN":
        harmonic = int(w[1])
        return sin[harmonic]

    if w[0] == "COS":
        harmonic = int(w[1])
        return cos[harmonic]


def apply_fourier_on_min(df, t, window=10, window_m=5):
    df["SIN_1_" + str(window) + "_" + str(window_m)] = df["Close_" + t].rolling(window=window_m).min().rolling(
        window=window).apply(lambda x: fourierSeries(x, 5, "SIN.0"))
    df["SIN_2_" + str(window) + "_" + str(window_m)] = df["Close_" + t].rolling(window=window_m).min().rolling(
        window=window).apply(lambda x: fourierSeries(x, 5, "SIN.1"))
    df["SIN_3_" + str(window) + "_" + str(window_m)] = df["Close_" + t].rolling(window=window_m).min().rolling(
        window=window).apply(lambda x: fourierSeries(x, 5, "SIN.2"))
    df["SIN_4_" + str(window) + "_" + str(window_m)] = df["Close_" + t].rolling(window=window_m).min().rolling(
        window=window).apply(lambda x: fourierSeries(x, 5, "SIN.3"))
    df["SIN_5_" + str(window) + "_" + str(window_m)] = df["Close_" + t].rolling(window=window_m).min().rolling(
        window=window).apply(lambda x: fourierSeries(x, 5, "SIN.4"))

    df["COS_1_" + str(window) + "_" + str(window_m)] = df["Close_" + t].rolling(window=window_m).min().rolling(
        window=window).apply(lambda x: fourierSeries(x, 5, "COS.0"))
    df["COS_2_" + str(window) + "_" + str(window_m)] = df["Close_" + t].rolling(window=window_m).min().rolling(
        window=window).apply(lambda x: fourierSeries(x, 5, "COS.1"))
    df["COS_3_" + str(window) + "_" + str(window_m)] = df["Close_" + t].rolling(window=window_m).min().rolling(
        window=window).apply(lambda x: fourierSeries(x, 5, "COS.2"))
    df["COS_4_" + str(window) + "_" + str(window_m)] = df["Close_" + t].rolling(window=window_m).min().rolling(
        window=window).apply(lambda x: fourierSeries(x, 5, "COS.3"))
    df["COS_5_" + str(window) + "_" + str(window_m)] = df["Close_" + t].rolling(window=window_m).min().rolling(
        window=window).apply(lambda x: fourierSeries(x, 5, "COS.4"))
    return df


def apply_fourier_on_max(df, t, window=10, window_m=5):
    df["SIN_1_" + str(window) + "_" + str(window_m)] = df["Close_" + t].rolling(window=window_m).max().rolling(
        window=window).apply(lambda x: fourierSeries(x, 5, "SIN.0"))
    df["SIN_2_" + str(window) + "_" + str(window_m)] = df["Close_" + t].rolling(window=window_m).max().rolling(
        window=window).apply(lambda x: fourierSeries(x, 5, "SIN.1"))
    df["SIN_3_" + str(window) + "_" + str(window_m)] = df["Close_" + t].rolling(window=window_m).max().rolling(
        window=window).apply(lambda x: fourierSeries(x, 5, "SIN.2"))
    df["SIN_4_" + str(window) + "_" + str(window_m)] = df["Close_" + t].rolling(window=window_m).max().rolling(
        window=window).apply(lambda x: fourierSeries(x, 5, "SIN.3"))
    df["SIN_5_" + str(window) + "_" + str(window_m)] = df["Close_" + t].rolling(window=window_m).max().rolling(
        window=window).apply(lambda x: fourierSeries(x, 5, "SIN.4"))

    df["COS_1_" + str(window) + "_" + str(window_m)] = df["Close_" + t].rolling(window=window_m).max().rolling(
        window=window).apply(lambda x: fourierSeries(x, 5, "COS.0"))
    df["COS_2_" + str(window) + "_" + str(window_m)] = df["Close_" + t].rolling(window=window_m).max().rolling(
        window=window).apply(lambda x: fourierSeries(x, 5, "COS.1"))
    df["COS_3_" + str(window) + "_" + str(window_m)] = df["Close_" + t].rolling(window=window_m).max().rolling(
        window=window).apply(lambda x: fourierSeries(x, 5, "COS.2"))
    df["COS_4_" + str(window) + "_" + str(window_m)] = df["Close_" + t].rolling(window=window_m).max().rolling(
        window=window).apply(lambda x: fourierSeries(x, 5, "COS.3"))
    df["COS_5_" + str(window) + "_" + str(window_m)] = df["Close_" + t].rolling(window=window_m).max().rolling(
        window=window).apply(lambda x: fourierSeries(x, 5, "COS.4"))
    return df


def applyFourier(df, t, window=10):
    df["SIN_1_" + str(window)] = df["Close_" + t].rolling(window=window).apply(lambda x: fourierSeries(x, 5, "SIN.0"))
    df["SIN_2_" + str(window)] = df["Close_" + t].rolling(window=window).apply(lambda x: fourierSeries(x, 5, "SIN.1"))

    df["COS_1_" + str(window)] = df["Close_" + t].rolling(window=window).apply(lambda x: fourierSeries(x, 5, "COS.0"))
    df["COS_2_" + str(window)] = df["Close_" + t].rolling(window=window).apply(lambda x: fourierSeries(x, 5, "COS.1"))

    df["SIN_1_D_" + str(window)] = df["Close_" + t].diff().rolling(window=window).apply(
        lambda x: fourierSeries(x, 5, "SIN.0"))
    df["SIN_2_D_" + str(window)] = df["Close_" + t].diff().rolling(window=window).apply(
        lambda x: fourierSeries(x, 5, "SIN.1"))

    df["COS_1_D_" + str(window)] = df["Close_" + t].diff().rolling(window=window).apply(
        lambda x: fourierSeries(x, 5, "COS.0"))
    df["COS_2_D_" + str(window)] = df["Close_" + t].diff().rolling(window=window).apply(
        lambda x: fourierSeries(x, 5, "COS.1"))
    df["COS_3_D_" + str(window)] = df["Close_" + t].diff().rolling(window=window).apply(
        lambda x: fourierSeries(x, 5, "COS.2"))
    df["COS_4_D_" + str(window)] = df["Close_" + t].diff().rolling(window=window).apply(
        lambda x: fourierSeries(x, 5, "COS.3"))
    df["COS_5_D_" + str(window)] = df["Close_" + t].diff().rolling(window=window).apply(
        lambda x: fourierSeries(x, 5, "COS.4"))
    return df


def garch(x, toReturn):
    model = arch_model(x)
    results = model.fit(disp="off", show_warning=False)
    if toReturn == "mu":
        return results.params["mu"]
    if toReturn == "omega":
        return results.params["omega"]
    if toReturn == "alpha":
        return results.params["alpha[1]"]
    if toReturn == "beta":
        return results.params["beta[1]"]


def apply_GARCH(df, t, window=15):
    df["GARCH_MU_" + str(window)] = df["Close_" + t] - df["Close_" + t].rolling(window=window).apply(
        lambda x: garch(x, "mu"))
    df["GARCH_OMEGA_" + str(window)] = df["Close_" + t].rolling(window=window).apply(lambda x: garch(x, "omega"))
    df["GARCH_ALPHA_" + str(window)] = df["Close_" + t].rolling(window=window).apply(lambda x: garch(x, "alpha"))
    df["GARCH_beta_" + str(window)] = df["Close_" + t].rolling(window=window).apply(lambda x: garch(x, "beta"))
    return df


def applyWCP(df, t):
    close = df["Close_" + t]
    open = df["Open_" + t]
    low = df["Low_" + t]
    df["WCP"] = close - ((close * 2.0 + open + low) / 4.0)
    return df


def apply_macd(df, slow, fast):
    applyTo = "Close"
    for col in df:
        if applyTo in col and 'adf' not in col:
            macd_line = df[col].rolling(window=fast).mean() - df[col].rolling(window=slow).mean()
            signal_line = macd_line.rolling(window=15).mean()
            macd_hist = macd_line - signal_line
            df[col + "_macdline"] = macd_line
            # df[col + "_signalline"] = signal_line
            # df[col + "_macdhist"] = macd_hist
            df[col + "_price-mean25"] = df[col].rolling(window=10).mean() - df[col]
            df[col + "_price-mean25"] = df[col].rolling(window=25).mean() - df[col]
            df[col + "_price-mean50"] = df[col].rolling(window=50).mean() - df[col]
            df[col + "_price-mean100"] = df[col].rolling(window=100).mean() - df[col]
            df[col + "_price-mean200"] = df[col].rolling(window=200).mean() - df[col]
            df[col + "_price-mean300"] = df[col].rolling(window=300).mean() - df[col]
            df[col + "_price-mean400"] = df[col].rolling(window=400).mean() - df[col]

            df["mean25-mean50"] = df[col + "_price-mean25"] - df[col + "_price-mean50"]
            df["mean50-mean100"] = df[col + "_price-mean50"] - df[col + "_price-mean100"]
            df["mean100-mean200"] = df[col + "_price-mean100"] - df[col + "_price-mean200"]
            df["mean200-mean300"] = df[col + "_price-mean200"] - df[col + "_price-mean300"]
            df["mean400-mean300"] = df[col + "_price-mean400"] - df[col + "_price-mean300"]
            df[col + "_price_log2"] = df[col].rolling(window=2).apply(lambda x: np.log(x[-1] / x[0]))
            df[col + "_price_log5"] = df[col].rolling(window=5).apply(lambda x: np.log(x[-1] / x[0]))
            df[col + "_price_log10"] = df[col].rolling(window=10).apply(lambda x: np.log(x[-1] / x[0]))
            df[col + "_price_log20"] = df[col].rolling(window=20).apply(lambda x: np.log(x[-1] / x[0]))
            df[col + "_price_log30"] = df[col].rolling(window=30).apply(lambda x: np.log(x[-1] / x[0]))
            df[col + "_price_log40"] = df[col].rolling(window=40).apply(lambda x: np.log(x[-1] / x[0]))

    return df


def discretize(df, bins=15):
    df = df.dropna(thresh=2, axis=1)
    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.dropna(how='any').reset_index(drop=True)
    for col in df:
        if col == "target":
            continue
        if len(df[col].unique()) == 1:
            df.drop(col, inplace=True, axis=1)
            continue
        df[col] = pd.qcut(df[col].values, bins, duplicates="drop").codes
    return df


def coefLinReg(x, window):
    sc = StandardScaler()
    x = sc.fit_transform(x.reshape(-1, 1))
    line = linear_model.LinearRegression()
    f = line.fit(array(range(1, window + 1)).reshape(-1, 1), x)
    return f.coef_[0]


def interceptLinReg(x, window):
    sc = StandardScaler()
    x = sc.fit_transform(x.reshape(-1, 1))
    line = linear_model.LinearRegression()
    f = line.fit(array(range(1, window + 1)).reshape(-1, 1), x)
    return f.intercept_[0]


def apply_std(df, t, window=20):
    df["STD_" + str(window)] = df["Close_" + t].rolling(window).std()
    window = window * 2
    df["STD_" + str(window)] = df["Close_" + t].rolling(window).std()
    return df


def apply_mayority(df, target, window=10):
    df["m"] = df["Close_" + target].rolling(window=window).apply(lambda x: coefLinReg(x, window))
    df = df.dropna(how='any').reset_index(drop=True)
    i = 0.1
    for j, row in df.iterrows():
        th = row["m"]
        up = th + i * abs(th)
        down = th - i * abs(th)
        newdf = df[(df["m"] < up) & (df["m"] > down)]
        sell = newdf[newdf["target"] == "SELL"].shape[0] + 1
        buy = newdf[newdf["target"] == "BUY"].shape[0] + 1
        sell = sell / (sell + buy) * 100.0
        df.set_value(j, "PCG_SELL_" + str(window), sell)

    return df


def dist_from_average(x, window):
    pass


def subset_df(df, target, window=150):
    print("Getting subset dataset using window: " + str(window))
    # df["THRESHOLD"] = df["Close_" + target].rolling(window=window).max() -df["Close_" + target].rolling(
    # window=window).min()
    # df["THRESHOLD"] = df["Close_" + target].rolling(window=window).apply(
    #    lambda x: dist_from_average(x, window))
    # df["THRESHOLD_2"] = df["Close_" + target].rolling(window=int(window)).apply(
    #    lambda x: dist_from_average(x, window))
    df["THRESHOLD"] = (df["Close_" + target].rolling(window=100).mean() - df["Close_" + target].rolling(
        window=50).mean()).rolling(
        window=20).sum()
    df["THRESHOLD_2"] = (df["Close_" + target].rolling(window=100).mean() - df["Close_" + target].rolling(
        window=200).mean()).rolling(
        window=20).sum()
    th = df.tail(1)["THRESHOLD"].tolist()[0]
    th2 = df.tail(1)["THRESHOLD_2"].tolist()[0]
    done = True
    i = 0.05
    while done:
        up = th + i * abs(th)
        down = th - i * abs(th)
        up2 = th2 + i * abs(th2)
        down2 = th2 - i * abs(th2)
        gmt = df.tail(1)["Gmt time"].tolist()[0]
        print("Cutting out values more than " + str(up))
        print("Cutting out values less than " + str(down))
        print("Cutting out values more than " + str(up2))
        print("Cutting out values less than " + str(down2))
        print("GMT: " + str(gmt))
        newdf = df[
            (df["THRESHOLD"] < up) & (df["THRESHOLD"] > down) & (df["THRESHOLD_2"] < up2) & (df["THRESHOLD_2"] > down2)]
        print(df.shape)
        print(newdf.shape)
        if newdf.shape[0] > SUBSET_SIZE:
            done = False
            print("Done With i: " + str(i))
        i = i + 0.05

    var = newdf.tail(1)["Gmt time"].tolist()[0]
    assert var == gmt
    sell = newdf[newdf["target"] == "SELL"].shape[0]
    buy = newdf[newdf["target"] == "BUY"].shape[0]
    out = newdf[newdf["target"] == "OUT"].shape[0]
    print("SELL " + str(sell))
    print("BUY " + str(buy))
    print("OUT " + str(out))
    return newdf

    col = df['Gmt time']
    month = col.apply(lambda x: int(x.split('-')[0]))
    df['month'] = month
    return df


def my_round(x):
    if x < 0.5:
        return -1
    else:
        return 1


def apply_support_and_resistance(df, target, window=50):
    close_label = "Close_" + target
    high_label = "High_" + target
    low_label = "Low_" + target
    close = df[close_label].rolling(window=window).mean()
    high = df[high_label].rolling(window=window).apply(lambda x: np.max(x))
    low = df[low_label].rolling(window=window).apply(lambda x: np.min(x))
    p = (high + low + close) / 3.0
    r1 = (2 * p) - low
    s1 = (2 * p) - high
    df["distanceFromR1_" + str(window)] = df["Close_" + target] - r1
    df["distanceFromS1_" + str(window)] = df["Close_" + target] - s1
    return df


def apply_bollinger_band(df, column, window=25):
    df['bollinger_band_up_' + str(window)] = \
        (df[column].rolling(window=window).mean() + 2.0 * df[column].rolling(window=window).std()) - df[column]

    df['bollinger_band_up_' + str(window) + "shift5"] = df['bollinger_band_up_' + str(window)].shift(5)

    df['bollinger_band_down_' + str(window)] = df[column] - (
            df[column].rolling(window=window).mean() - 2.0 * df[column].rolling(
        window=window).std())

    df['bollinger_band_down_' + str(window) + "shift5"] = df['bollinger_band_down_' + str(window)].shift(5)

    df['bollinger_band_up_' + str(window) + "3STD"] = \
        (df[column].rolling(window=window).mean() + 3.0 * df[column].rolling(window=window).std()) - df[column]

    df['bollinger_band_down_' + str(window) + "3STD"] = df[column] - (
            df[column].rolling(window=window).mean() - 3.0 * df[column].rolling(
        window=window).std())

    df['bollinger_band_up_' + str(window) + "4STD"] = \
        (df[column].rolling(window=window).mean() + 4.0 * df[column].rolling(window=window).std()) - df[column]

    df['bollinger_band_down_' + str(window) + "4STD"] = df[column] - (
            df[column].rolling(window=window).mean() - 4.0 * df[column].rolling(
        window=window).std())

    # df['bollinger_band_diff_' + str(window)] = df['bollinger_band_up_' + str(window)] - df['bollinger_band_down_' + str(window)]
    return df


def apply_diff_on(df, l):
    for i in l:
        if i in df:
            df[i + "_diff"] = df[i].diff()
    return df


def get_random_list(length):
    l = [my_round(random.random()) for _ in range(0, length, 1)]
    l = ["BUY" if x == 1 else "SELL" for x in l]
    return l


def apply_mvavg(df, excluded, window):
    for col in df:
        if any(substring in col for substring in excluded):
            continue
        df[col + "_mv_avg" + str(window)] = df[col].rolling(window=window).mean()
    return df


def apply_mvavg_to_target(df, param):
    df["target_" + str(param)] = df["target"].rolling(window=param).mean()
    return df


def apply_percentage(df, excluded):
    for col in df:
        if any(substring in col for substring in excluded):
            continue
        df[col + "_pctg"] = df[col].pct_change() * 100
        # del df[col]
    return df


def create_y(row, column_to_check):
    if row[column_to_check] >= THRESHOLD:
        return 1
    elif row[column_to_check] < THRESHOLD:
        return -1
    return 0
