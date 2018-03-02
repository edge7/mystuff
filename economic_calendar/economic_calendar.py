import os

import pandas as pd
import datetime

old_calendar = None
old_max_row = None


def check_if_available(root_dir, target):
    toReturn = []
    files = [item for item in os.listdir(root_dir) if os.path.isfile(os.path.join(root_dir, item))]
    for f in files:
        if f in target:
            print("FOUND EC. Calendar: " + f)
            df = pd.read_csv(str(root_dir + "\\" + f), delimiter="@")
            df["Gmt time"] = df["Gmt time"].apply(lambda x: datetime.datetime.strptime(x, '%Y-%m-%d %H:%M:%S'))
            toReturn.append(df)
    return toReturn


def apply_calendar(df, calendars, old_calendar=None):

    if len(calendars) == 0:
        print("No calendars available")
        return df

    if old_calendar is not None:
        df = pd.merge(df, old_calendar, on="Gmt time", how='left')
        start = old_calendar.shape[0] - 1
        print("Merging as I CAN!")
        print("RIPARTO DA " + str(start))
        print(str(old_calendar["DIFF_LAST_SUMMARY"][start -5]))
        print(str(df["DIFF_LAST_SUMMARY"][start -5]))
        print(str(old_calendar["Gmt time"][start - 5]))
        print(str(df["Gmt time"][start - 5]))
    else:
        start = 0
        old_calendar = pd.DataFrame()
        print("CANNOT MERGE")

    end = df.shape[0]
    print("Found calendars .. iterating")
    last_summaries = []

    for c in calendars:
        columns = list(set(c["What"].tolist()))
        symbol = c["Symbol"][0]
        print("CALENDAR iteration")
        # Iterating row by row, check change after AHEAD candles
        for index in range(start, end):
            d = datetime.datetime.strptime(df["Gmt time"][index], '%Y.%m.%d %H:%M:%S')
            old_calendar.set_value(index, "Gmt time", df["Gmt time"][index])
            for col in columns:
                tmp = c[c["What"] == col]
                back = tmp[tmp["Gmt time"] <= d]
                last = back.tail(1)
                try:
                    actual = last["Actual"].tolist()[0]
                    forecast = last["Forecast"].tolist()[0]
                    a_f = last["A-F"].tolist()[0]
                    df.set_value(index, symbol + "_LAST_ACTUAL_" + col, float(actual))
                    df.set_value(index, symbol + "_LAST_A-F_" + col, float(a_f))
                    old_calendar.set_value(index, symbol + "_LAST_ACTUAL_" + col, float(actual))
                    old_calendar.set_value(index, symbol + "_LAST_A-F_" + col, float(a_f))
                except Exception as e:
                    # print(e)
                    # print(col)
                    # print(d)
                    # print("SKIPPPING")
                    df.set_value(index, symbol + "_LAST_ACTUAL_" + col, float(0))
                    # df.set_value(index, symbol + "_LAST_FORECAST_" + col, float(forecast))
                    df.set_value(index, symbol + "_LAST_A-F_" + col, float(0))
                    old_calendar.set_value(index, symbol + "_LAST_ACTUAL_" + col, -100000.0)
                    old_calendar.set_value(index, symbol + "_LAST_A-F_" + col, -100000.0)
                    pass
        # Sum up results:
        print("Summing up . . . ")
        for i, row in df.iterrows():
            good = 0
            bad = 0
            for j, column in row.iteritems():
                if "_LAST_A-F_" not in j or "UnemploymentChange" in j:
                    continue
                if column > 0:
                    good += 1
                if column < 0:
                    bad += 1
            res = good - bad
            df.set_value(i, symbol + "LAST_SUMMARY", res)
            old_calendar.set_value(i, symbol + "LAST_SUMMARY", res)
        last_summaries.append(symbol + "LAST_SUMMARY")
    df["DIFF_LAST_SUMMARY"] = df[last_summaries[0]] - df[last_summaries[1]]
    old_calendar["DIFF_LAST_SUMMARY"] = df[last_summaries[0]] - df[last_summaries[1]]
    return df, old_calendar
