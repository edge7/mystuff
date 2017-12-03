import os
import statistics
import pathlib

from ingestion.main import TARGET_VARIABLE
from processing.processing import create_dataframe, apply_diff, create_y, apply_bollinger_band

TESTDATA_FILENAME = os.path.join(os.path.dirname(__file__), 'testdata')


def get_value_row(d, item):
    return list(d[item])[0]


def test_create_dataframe():
    crossList = ["test_create"]
    flist = [p for p in pathlib.Path(TESTDATA_FILENAME).iterdir() if p.is_file()]
    # Creating n dataframe where n is the number of files
    dfs = create_dataframe(flist, "Gmt time", crossList)
    assert len(dfs) == 1
    df = dfs[0]
    # get schema
    schema = list(df)
    # Expected schema
    expected_schema = ['Gmt time', 'Close_test_create', 'Open_test_create', 'High_test_create', 'Low_test_create',
                       'test_create_High_in_pips', 'test_create_Body_in_pips', 'test_create_Low_in_pips',
                       'test_create_High-Low', 'test_create_High-body', 'test_create_Low-body',
                       'test_create_High_in_pips_bef', 'test_create_Body_in_pips_bef', 'test_create_Low_in_pips_bef',
                       'test_create_High-body_bef', 'test_create_Low-body_bef', 'test_create_High-Low_bef',
                       'test_create_High_in_pips_bef_bef', 'test_create_Body_in_pips_bef_bef',
                       'test_create_Low_in_pips_bef_bef', 'test_create_CurrentBody-Body_bef',
                       'test_create_CurrentHigh-High_bef', 'test_create_CurrentLow-Low_bef']

    assert sorted(expected_schema) == sorted(schema)

    last_row = df.tail(1)

    assert list(last_row['Gmt time'])[0] == '12-04-1990'

    assert get_value_row(last_row, 'Close_test_create') == 1000.1

    assert get_value_row(last_row, 'Open_test_create') == 1000.2

    assert get_value_row(last_row, 'High_test_create') == 2000.5

    assert get_value_row(last_row, 'Low_test_create') == 1

    assert get_value_row(last_row, 'test_create_High_in_pips') == - 2000.5 + 1000.2

    assert get_value_row(last_row, 'test_create_Body_in_pips') == - 1000.2 + 1000.1

    assert get_value_row(last_row, 'test_create_Low_in_pips') == 1 - 1000.1

    assert get_value_row(last_row, 'test_create_High-Low') == abs(- 2000.5 + 1000.2 - 1 + 1000.1)

    assert get_value_row(last_row, 'test_create_High-body') == abs(- 2000.5 + 1000.2) - abs(- 1000.2 + 1000.1)

    assert get_value_row(last_row, 'test_create_Low-body') == abs(1 - 1000.1) - abs(- 1000.2 + 1000.1)

    # Bef

    assert get_value_row(last_row, 'test_create_High_in_pips_bef') == 140.5 - 100

    assert get_value_row(last_row, 'test_create_Body_in_pips_bef') == 100 - 2

    assert get_value_row(last_row, 'test_create_Low_in_pips_bef') == 2 - 1.5

    assert get_value_row(last_row, 'test_create_High-body_bef') == abs(140.5 - 100) - abs(98)

    assert get_value_row(last_row, 'test_create_High-Low_bef') == abs(140.5 - 100) - abs(2 - 1.5)

    assert get_value_row(last_row, 'test_create_Low-body_bef') == abs(2 - 1.5) - abs(98)

    # Bef bef
    assert get_value_row(last_row, 'test_create_High_in_pips_bef_bef') == 3

    assert get_value_row(last_row, 'test_create_Body_in_pips_bef_bef') == 1

    assert get_value_row(last_row, 'test_create_Low_in_pips_bef_bef') == 2

    # Other
    assert get_value_row(last_row, 'test_create_CurrentBody-Body_bef') == abs(- 1000.2 + 1000.1) - abs(100 - 2)

    assert get_value_row(last_row, 'test_create_CurrentHigh-High_bef') == abs(- 2000.5 + 1000.2) - abs(140.5 - 100)

    assert get_value_row(last_row, 'test_create_CurrentLow-Low_bef') == abs(1 - 1000.1) - abs(2 - 1.5)


def test_apply_diff():
    crossList = ["test_create"]
    flist = [p for p in pathlib.Path(TESTDATA_FILENAME).iterdir() if p.is_file()]
    # Creating n dataframe where n is the number of files
    dfs = create_dataframe(flist, "Gmt time", crossList)
    df = dfs[0]
    df = apply_diff(df, ["Gmt time", "adf_", "dist_from_"])

    last = df.tail(1)

    assert get_value_row(last, 'Close_test_create_diff') == 1000.1 - 100

    df['target'] = df.apply(lambda row: create_y(row, "Close_" + "test_create" + "_diff"), axis=1)

    df['target'] = df['target'].shift(-1)

    print('end')


def test_bb():
    crossList = ["test_create"]
    flist = [p for p in pathlib.Path(TESTDATA_FILENAME).iterdir() if p.is_file()]
    # Creating n dataframe where n is the number of files
    dfs = create_dataframe(flist, "Gmt time", crossList)
    df = dfs[0]
    df = apply_bollinger_band(df, "Close_" + crossList[0], window=2)
    std = statistics.stdev([1000.1, 100])
    mean = statistics.mean([1000.1, 100])
    up = mean + 2.0*std - 1000.1
    down = 1000.1 - (mean - 2.0*std)
    assert get_value_row(df.tail(1), 'bollinger_band_up_2') == up
    assert get_value_row(df.tail(1), 'bollinger_band_down_2') == down
