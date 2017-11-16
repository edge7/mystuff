import pandas as pd
import statsmodels.tsa.stattools as ts
from numpy import polyfit, log10, sqrt, std, subtract


def adf(y):
    res = ts.adfuller(y, maxlag=1, autolag=None)
    adf = res[0]
    pvalue = res[1]
    return adf

def hurst_test(p):
    """Returns the Hurst Exponent of the time series vector ts"""
    p = log10(p)
    tau = []
    lagvec = []

    #  Step through the different lags
    for lag in range(5, 19):
        #  produce price difference with lag
        pp = subtract(p[lag:], p[:-lag])

        #  Write the different lags into a vector
        lagvec.append(lag)

        #  Calculate the variance of the difference vector
        tau.append(sqrt(std(pp)))

    # linear fit to double-log graph (gives power)
    m = polyfit(log10(lagvec), log10(tau), 1)

    # calculate hurst
    hurst = m[0] * 2

    # plot lag vs variance
    # py.plot(lagvec,tau,'o'); show()

    return hurst

#df = pd.read_csv(str('/home/toniotonia47/Desktop/stockML/data/CADJPY/CAD_JPY.csv'))
#hurst_test(df['Close'])