import pandas as pd
from numpy import nan
import numpy as np

THRESHOLD = 0


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
    df[toUse + "_High-Low"] = abs(high_in_pips - low_in_pips)
    df[toUse + "_High-body"] = abs(high_in_pips) - abs(body_in_pips)
    df[toUse + "_Low-body"] = abs(low_in_pips) - abs(body_in_pips)
    high_in_pips = ret[0][1]
    body_in_pips = ret[1][1]
    low_in_pips = ret[2][1]
    df[toUse + "_High_in_pips_bef"] = high_in_pips
    df[toUse + "_Body_in_pips_bef"] = body_in_pips
    df[toUse + "_Low_in_pips_bef"] = low_in_pips
    df[toUse + "_High-body_bef"] = abs(high_in_pips) - abs(body_in_pips)
    df[toUse + "_Low-body_bef"] = abs(low_in_pips) - abs(body_in_pips)
    # df[toUse + "_High/Body_bef"] = abs(high_in_pips / body_in_pips)
    # df[toUse + "_Low/body_bef"] = abs(low_in_pips / body_in_pips)
    # df[toUse + "_Low/High_bef"] = abs(low_in_pips / high_in_pips)
    df[toUse + "_High-Low_bef"] = abs(high_in_pips - low_in_pips)

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


def transform_columns_names(df, crossList, path, excluded):
    toUse = crossList[0]
    for j in crossList:
        if j in str(path):
            toUse = j

    for col in df:
        if excluded not in col:
            df[col + "_" + toUse] = df[col]
            # df[col + "_" + toUse] = df[col  + "_" + toUse].rolling(window=10).mean()
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


def create_dataframe(flist, excluded, crossList):
    l = list()
    for path in flist:
        df = pd.read_csv(str(path))
        del df['Adj Close']
        # cols = list(df.columns.values)
        # for i in cols:
        #    df[i] = pd.to_numeric(df[i], errors='ignore')

        # df = df.head(75).reset_index()
        df = df[df["Volume"] != 0.000].reset_index()
        df = transform_columns_names(df, crossList, path, excluded)
        l.append(df)
    return l


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


def apply_diff(df, excluded):
    for col in df:
        if excluded in col:
            continue
        df[col + "_diff"] = df[col].diff()
    return df


def apply_macd(df, slow, fast):
    applyTo = "Close"
    for col in df:
        if applyTo in col:
            macd_line = df[col].rolling(window=fast).mean() - df[col].rolling(window=slow).mean()
            signal_line = macd_line.rolling(window=9).mean()
            macd_hist = macd_line - signal_line
            df[col + "_macdline"] = macd_line
            df[col + "_signalline"] = signal_line
            df[col + "_macdhist"] = macd_hist
            df[col + "_price-mean25"] = df[col].rolling(window=25).mean() - df[col]
            df[col + "_price-mean50"] = df[col].rolling(window=50).mean() - df[col]
            df[col + "_price_log"] = df[col].rolling(window=2).apply(lambda x: np.log(x[1] / x[0]))

    return df


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
