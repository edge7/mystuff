import random

import pandas as pd
from numpy import nan
import numpy as np

from algos.algos import adf

THRESHOLD = 0
AHEAD = 5

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

    return [high_in_pips, body_in_pips, low_in_pips]


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


def add_candlestick_columns(df, toUse):
    ret = df.apply(lambda row: apply_candle(row, toUse), axis=1)
    ret = create_cols(ret)
    high_in_pips = ret[0][0]
    body_in_pips = ret[1][0]
    low_in_pips = ret[2][0]
    df[toUse + "_High_in_pips"] = high_in_pips
    df[toUse + "_Body_in_pips"] = body_in_pips
    df[toUse + "_Low_in_pips"] = low_in_pips
    # df[toUse + "_High/Body"] = abs(high_in_pips / body_in_pips)
    # df[toUse + "_Low/body"] = abs(low_in_pips / body_in_pips)
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
    # df[toUse + "_High/Body_bef"] = abs(high_in_pips / body_in_pips)
    # df[toUse + "_Low/body_bef"] = abs(low_in_pips / body_in_pips)
    # df[toUse + "_Low/High_bef"] = abs(low_in_pips / high_in_pips)
    # df[toUse + "_High-Low_bef"] = abs(high_in_pips - low_in_pips)

    high_in_pips = ret[0][2]
    body_in_pips = ret[1][2]
    low_in_pips = ret[2][2]
    df[toUse + "_High_in_pips_bef_bef"] = high_in_pips
    df[toUse + "_Body_in_pips_bef_bef"] = body_in_pips
    df[toUse + "_Low_in_pips_bef_bef"] = low_in_pips

    df[toUse + "_CurrentBody-Body_bef"] = abs(df[toUse + "_Body_in_pips"]) - abs(df[toUse + "_Body_in_pips_bef"])
    df[toUse + "_CurrentHigh-High_bef"] = abs(df[toUse + "_High_in_pips"]) - abs(df[toUse + "_High_in_pips_bef"])
    df[toUse + "_CurrentLow-Low_bef"] = abs(df[toUse + "_Low_in_pips"]) - abs(df[toUse + "_Low_in_pips_bef"])
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

    df = add_candlestick_columns(df, toUse)

    return df


def drop_original_values(df, cross_list):
    for toUse in cross_list:
        del df["Open_" + toUse]
        del df["Close_" + toUse]
        del df["High_" + toUse]
        del df["Low_" + toUse]
    return df


def apply_df_test(df, target):
    df[target + 'adf_50'] = df[target].rolling(window=50).apply(lambda x: adf(x))
    df[target + 'adf_100'] = df[target].rolling(window=100).apply(lambda x: adf(x))
    # df[target + 'adf_25'] = df[target].rolling(window=25).apply(lambda x: adf(x))
    return df


def create_dataframe(flist, excluded, crossList, keep_names=True):
    l = list()
    for path in flist:
        df = pd.read_csv(str(path))
        if 'Adj Close' in df:
            del df['Adj Close']
        # cols = list(df.columns.values)
        # for i in cols:
        #    df[i] = pd.to_numeric(df[i], errors='ignore')
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
        for c in columns:
            if c == col:
                del df[c]
    return df


def apply_diff(df, excluded):
    for col in df:
        if any(ext in col for ext in excluded):
            continue
        df[col + "_diff"] = df[col].diff()
    return df


def apply_ichimo(df, t):
    # Kijun
    # averaging the highest high and the lowest low for the past 26 periods
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


def create_target_ahead(df, CLOSE_VARIABLE, AHEAD, threshold):
    number_rows = df.shape[0]
    # Iterating row by row, check change after AHEAD candles
    for index in range(number_rows):
        start = df.iloc[[index]][CLOSE_VARIABLE][index]
        to_set = 0
        to_set_ = "OUT"
        end = min(number_rows, index + AHEAD +1)
        for next in range(index+1, end):
            value = df.iloc[[next]][CLOSE_VARIABLE][next]
            pctg = (value / start - 1 ) * 100.0
            if pctg > threshold:
                to_set += 1
            if pctg < - threshold:
                to_set -=1
        if to_set > 0:
            to_set_ = "BUY"
        if to_set < 0:
            to_set_ = "SELL"

        df.set_value(index, 'target', to_set_)

    return df

def apply_stochastic(df, t):
    for col in df:
        if 'High_' + t == col:
            high14 = df[col].rolling(window=14).max()
        if 'Low_' + t == col:
            low14 = df[col].rolling(window=14).min()

    fast_stoc = 100.0 * (df["Close_" + t] - low14) / (high14 - low14)
    slow_stoc = fast_stoc.rolling(window=3).mean()
    df["fast_stoc"] = fast_stoc
    df["slow_stoc"] = slow_stoc
    df["fast-slow_stoc"] = fast_stoc - slow_stoc
    return df


def apply_macd(df, slow, fast):
    applyTo = "Close"
    for col in df:
        if applyTo in col and 'adf' not in col:
            macd_line = df[col].rolling(window=fast).mean() - df[col].rolling(window=slow).mean()
            signal_line = macd_line.rolling(window=9).mean()
            macd_hist = macd_line - signal_line
            df[col + "_macdline"] = macd_line
            # df[col + "_signalline"] = signal_line
            # df[col + "_macdhist"] = macd_hist
            # df[col + "_price-mean25"] = df[col].rolling(window=25).mean() - df[col]
            df[col + "_price-mean50"] = df[col].rolling(window=50).mean() - df[col]
            df[col + "_price-mean100"] = df[col].rolling(window=100).mean() - df[col]

            df["mean50-mean100"] = df[col + "_price-mean50"] - df[col + "_price-mean100"]
            # df[col + "_price_log"] = df[col].rolling(window=2).apply(lambda x: np.log(x[1] / x[0]))

    return df


def create_month_column(df):
    col = df['Gmt time']
    month = col.apply(lambda x: int(x.split('-')[0]))
    df['month'] = month
    return df


def my_round(x):
    if x < 0.5:
        return -1
    else:
        return 1


def apply_bollinger_band(df, column, window=25):
    df['bollinger_band_up_' + str(window)] = \
        (df[column].rolling(window=window).mean() + 2.0 * df[column].rolling(window=window).std()) - df[column]

    df['bollinger_band_down_' + str(window)] = df[column] - (
        df[column].rolling(window=window).mean() - 2.0 * df[column].rolling(
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
    l = [ "BUY" if x == 1  else "SELL" for x in l ]
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
