"""
.. module:: momentum
   :synopsis: Momentum Indicators.

.. moduleauthor:: Dario Lopez Padial (Bukosabino)

"""
import numpy as np
import pandas as pd

"""
.. module:: utils
   :synopsis: Utils classes and functions.

.. moduleauthor:: Dario Lopez Padial (Bukosabino)

"""
import math

import numpy as np
import pandas as pd


class IndicatorMixin:
    """Util mixin indicator class"""

    _fillna = False

    def _check_fillna(self, series: pd.Series, value: int = 0) -> pd.Series:
        """Check if fillna flag is True.

        Args:
            series(pandas.Series): dataset 'Close' column.
            value(int): value to fill gaps; if -1 fill values using 'backfill' mode.

        Returns:
            pandas.Series: New feature generated.
        """
        if self._fillna:
            series_output = series.copy(deep=False)
            series_output = series_output.replace([np.inf, -np.inf], np.nan)
            if isinstance(value, int) and value == -1:
                series = series_output.fillna(method="ffill").fillna(value=-1)
            else:
                series = series_output.fillna(method="ffill").fillna(value)
        return series

    @staticmethod
    def _true_range(
        high: pd.Series, low: pd.Series, prev_close: pd.Series
    ) -> pd.Series:
        tr1 = high - low
        tr2 = (high - prev_close).abs()
        tr3 = (low - prev_close).abs()
        true_range = pd.DataFrame(data={"tr1": tr1, "tr2": tr2, "tr3": tr3}).max(axis=1)
        return true_range


def dropna(df: pd.DataFrame) -> pd.DataFrame:
    """Drop rows with "Nans" values"""
    df = df.copy()
    number_cols = df.select_dtypes("number").columns.to_list()
    df[number_cols] = df[number_cols][df[number_cols] < math.exp(709)]  # big number
    df[number_cols] = df[number_cols][df[number_cols] != 0.0]
    df = df.dropna()
    return df


def _sma(series, periods: int, fillna: bool = False):
    min_periods = 0 if fillna else periods
    return series.rolling(window=periods, min_periods=min_periods).mean()


def _ema(series, periods, fillna=False):
    min_periods = 0 if fillna else periods
    return series.ewm(span=periods, min_periods=min_periods, adjust=False).mean()


def _get_min_max(series1: pd.Series, series2: pd.Series, function: str = "min"):
    """Find min or max value between two lists for each index"""
    series1 = np.array(series1)
    series2 = np.array(series2)
    if function == "min":
        output = np.amin([series1, series2], axis=0)
    elif function == "max":
        output = np.amax([series1, series2], axis=0)
    else:
        raise ValueError('"f" variable value should be "min" or "max"')

    return pd.Series(output)



class RSIIndicator(IndicatorMixin):
    """Relative Strength Index (RSI)

    Compares the magnitude of recent gains and losses over a specified time
    period to measure speed and change of price movements of a security. It is
    primarily used to attempt to identify overbought or oversold conditions in
    the trading of an asset.

    https://www.investopedia.com/terms/r/rsi.asp

    Args:
        close(pandas.Series): dataset 'Close' column.
        window(int): n period.
        fillna(bool): if True, fill nan values.
    """

    def __init__(self, close: pd.Series, window: int = 14, fillna: bool = False):
        self._close = close
        self._window = window
        self._fillna = fillna
        self._run()

    def _run(self):
        diff = self._close.diff(1)
        up_direction = diff.where(diff > 0, 0.0)
        down_direction = -diff.where(diff < 0, 0.0)
        min_periods = 0 if self._fillna else self._window
        emaup = up_direction.ewm(
            alpha=1 / self._window, min_periods=min_periods, adjust=False
        ).mean()
        emadn = down_direction.ewm(
            alpha=1 / self._window, min_periods=min_periods, adjust=False
        ).mean()
        relative_strength = emaup / emadn
        self._rsi = pd.Series(
            np.where(emadn == 0, 100, 100 - (100 / (1 + relative_strength))),
            index=self._close.index,
        )

    def rsi(self) -> pd.Series:
        """Relative Strength Index (RSI)

        Returns:
            pandas.Series: New feature generated.
        """
        rsi_series = self._check_fillna(self._rsi, value=50)
        return pd.Series(rsi_series, name="rsi")


class TSIIndicator(IndicatorMixin):
    """True strength index (TSI)

    Shows both trend direction and overbought/oversold conditions.

    https://school.stockcharts.com/doku.php?id=technical_indicators:true_strength_index

    Args:
        close(pandas.Series): dataset 'Close' column.
        window_slow(int): high period.
        window_fast(int): low period.
        fillna(bool): if True, fill nan values.
    """

    def __init__(
        self,
        close: pd.Series,
        window_slow: int = 25,
        window_fast: int = 13,
        fillna: bool = False,
    ):
        self._close = close
        self._window_slow = window_slow
        self._window_fast = window_fast
        self._fillna = fillna
        self._run()

    def _run(self):
        diff_close = self._close - self._close.shift(1)
        min_periods_r = 0 if self._fillna else self._window_slow
        min_periods_s = 0 if self._fillna else self._window_fast
        smoothed = (
            diff_close.ewm(
                span=self._window_slow, min_periods=min_periods_r, adjust=False
            )
            .mean()
            .ewm(span=self._window_fast, min_periods=min_periods_s, adjust=False)
            .mean()
        )
        smoothed_abs = (
            abs(diff_close)
            .ewm(span=self._window_slow, min_periods=min_periods_r, adjust=False)
            .mean()
            .ewm(span=self._window_fast, min_periods=min_periods_s, adjust=False)
            .mean()
        )
        self._tsi = smoothed / smoothed_abs
        self._tsi *= 100

    def tsi(self) -> pd.Series:
        """True strength index (TSI)

        Returns:
            pandas.Series: New feature generated.
        """
        tsi_series = self._check_fillna(self._tsi, value=0)
        return pd.Series(tsi_series, name="tsi")


class UltimateOscillator(IndicatorMixin):
    """Ultimate Oscillator

    Larry Williams' (1976) signal, a momentum oscillator designed to capture
    momentum across three different timeframes.

    http://stockcharts.com/school/doku.php?id=chart_school:technical_indicators:ultimate_oscillator

    BP = Close - Minimum(Low or Prior Close).
    TR = Maximum(High or Prior Close)  -  Minimum(Low or Prior Close)
    Average7 = (7-period BP Sum) / (7-period TR Sum)
    Average14 = (14-period BP Sum) / (14-period TR Sum)
    Average28 = (28-period BP Sum) / (28-period TR Sum)

    UO = 100 x [(4 x Average7)+(2 x Average14)+Average28]/(4+2+1)

    Args:
        high(pandas.Series): dataset 'High' column.
        low(pandas.Series): dataset 'Low' column.
        close(pandas.Series): dataset 'Close' column.
        window1(int): short period.
        window2(int): medium period.
        window3(int): long period.
        weight1(float): weight of short BP average for UO.
        weight2(float): weight of medium BP average for UO.
        weight3(float): weight of long BP average for UO.
        fillna(bool): if True, fill nan values with 50.
    """

    def __init__(
        self,
        high: pd.Series,
        low: pd.Series,
        close: pd.Series,
        window1: int = 7,
        window2: int = 14,
        window3: int = 28,
        weight1: float = 4.0,
        weight2: float = 2.0,
        weight3: float = 1.0,
        fillna: bool = False,
    ):
        self._high = high
        self._low = low
        self._close = close
        self._window1 = window1
        self._window2 = window2
        self._window3 = window3
        self._weight1 = weight1
        self._weight2 = weight2
        self._weight3 = weight3
        self._fillna = fillna
        self._run()

    def _run(self):
        close_shift = self._close.shift(1)
        true_range = self._true_range(self._high, self._low, close_shift)
        buying_pressure = self._close - pd.DataFrame(
            {"low": self._low, "close": close_shift}
        ).min(axis=1, skipna=False)
        min_periods_s = 0 if self._fillna else self._window1
        min_periods_m = 0 if self._fillna else self._window2
        min_periods_len = 0 if self._fillna else self._window3
        avg_s = (
            buying_pressure.rolling(self._window1, min_periods=min_periods_s).sum()
            / true_range.rolling(self._window1, min_periods=min_periods_s).sum()
        )
        avg_m = (
            buying_pressure.rolling(self._window2, min_periods=min_periods_m).sum()
            / true_range.rolling(self._window2, min_periods=min_periods_m).sum()
        )
        avg_l = (
            buying_pressure.rolling(self._window3, min_periods=min_periods_len).sum()
            / true_range.rolling(self._window3, min_periods=min_periods_len).sum()
        )
        self._uo = (
            100.0
            * (
                (self._weight1 * avg_s)
                + (self._weight2 * avg_m)
                + (self._weight3 * avg_l)
            )
            / (self._weight1 + self._weight2 + self._weight3)
        )

    def ultimate_oscillator(self) -> pd.Series:
        """Ultimate Oscillator

        Returns:
            pandas.Series: New feature generated.
        """
        ultimate_osc = self._check_fillna(self._uo, value=50)
        return pd.Series(ultimate_osc, name="uo")


class StochasticOscillator(IndicatorMixin):
    """Stochastic Oscillator

    Developed in the late 1950s by George Lane. The stochastic
    oscillator presents the location of the closing price of a
    stock in relation to the high and low range of the price
    of a stock over a period of time, typically a 14-day period.

    https://school.stockcharts.com/doku.php?id=technical_indicators:stochastic_oscillator_fast_slow_and_full

    Args:
        close(pandas.Series): dataset 'Close' column.
        high(pandas.Series): dataset 'High' column.
        low(pandas.Series): dataset 'Low' column.
        window(int): n period.
        smooth_window(int): sma period over stoch_k.
        fillna(bool): if True, fill nan values.
    """

    def __init__(
        self,
        high: pd.Series,
        low: pd.Series,
        close: pd.Series,
        window: int = 14,
        smooth_window: int = 3,
        fillna: bool = False,
    ):
        self._close = close
        self._high = high
        self._low = low
        self._window = window
        self._smooth_window = smooth_window
        self._fillna = fillna
        self._run()

    def _run(self):
        min_periods = 0 if self._fillna else self._window
        smin = self._low.rolling(self._window, min_periods=min_periods).min()
        smax = self._high.rolling(self._window, min_periods=min_periods).max()
        self._stoch_k = 100 * (self._close - smin) / (smax - smin)

    def stoch(self) -> pd.Series:
        """Stochastic Oscillator

        Returns:
            pandas.Series: New feature generated.
        """
        stoch_k = self._check_fillna(self._stoch_k, value=50)
        return pd.Series(stoch_k, name="stoch_k")

    def stoch_signal(self) -> pd.Series:
        """Signal Stochastic Oscillator

        Returns:
            pandas.Series: New feature generated.
        """
        min_periods = 0 if self._fillna else self._smooth_window
        stoch_d = self._stoch_k.rolling(
            self._smooth_window, min_periods=min_periods
        ).mean()
        stoch_d = self._check_fillna(stoch_d, value=50)
        return pd.Series(stoch_d, name="stoch_k_signal")


class KAMAIndicator(IndicatorMixin):
    """Kaufman's Adaptive Moving Average (KAMA)

    Moving average designed to account for market noise or volatility. KAMA
    will closely follow prices when the price swings are relatively small and
    the noise is low. KAMA will adjust when the price swings widen and follow
    prices from a greater distance. This trend-following indicator can be
    used to identify the overall trend, time turning points and filter price
    movements.

    https://www.tradingview.com/ideas/kama/

    Args:
        close(pandas.Series): dataset 'Close' column.
        window(int): n period.
        pow1(int): number of periods for the fastest EMA constant.
        pow2(int): number of periods for the slowest EMA constant.
        fillna(bool): if True, fill nan values.
    """

    def __init__(
        self,
        close: pd.Series,
        window: int = 10,
        pow1: int = 2,
        pow2: int = 30,
        fillna: bool = False,
    ):
        self._close = close
        self._window = window
        self._pow1 = pow1
        self._pow2 = pow2
        self._fillna = fillna
        self._run()

    def _run(self):
        close_values = self._close.values
        vol = pd.Series(abs(self._close - np.roll(self._close, 1)))

        min_periods = 0 if self._fillna else self._window
        er_num = abs(close_values - np.roll(close_values, self._window))
        er_den = vol.rolling(self._window, min_periods=min_periods).sum()
        efficiency_ratio = er_num / er_den

        smoothing_constant = (
            (
                efficiency_ratio * (2.0 / (self._pow1 + 1) - 2.0 / (self._pow2 + 1.0))
                + 2 / (self._pow2 + 1.0)
            )
            ** 2.0
        ).values

        self._kama = np.zeros(smoothing_constant.size)
        len_kama = len(self._kama)
        first_value = True

        for i in range(len_kama):
            if np.isnan(smoothing_constant[i]):
                self._kama[i] = np.nan
            elif first_value:
                self._kama[i] = close_values[i]
                first_value = False
            else:
                self._kama[i] = self._kama[i - 1] + smoothing_constant[i] * (
                    close_values[i] - self._kama[i - 1]
                )

    def kama(self) -> pd.Series:
        """Kaufman's Adaptive Moving Average (KAMA)

        Returns:
            pandas.Series: New feature generated.
        """
        kama_series = pd.Series(self._kama, index=self._close.index)
        kama_series = self._check_fillna(kama_series, value=self._close)
        return pd.Series(kama_series, name="kama")


class ROCIndicator(IndicatorMixin):
    """Rate of Change (ROC)

    The Rate-of-Change (ROC) indicator, which is also referred to as simply
    Momentum, is a pure momentum oscillator that measures the percent change in
    price from one period to the next. The ROC calculation compares the current
    price with the price “n” periods ago. The plot forms an oscillator that
    fluctuates above and below the zero line as the Rate-of-Change moves from
    positive to negative. As a momentum oscillator, ROC signals include
    centerline crossovers, divergences and overbought-oversold readings.
    Divergences fail to foreshadow reversals more often than not, so this
    article will forgo a detailed discussion on them. Even though centerline
    crossovers are prone to whipsaw, especially short-term, these crossovers
    can be used to identify the overall trend. Identifying overbought or
    oversold extremes comes naturally to the Rate-of-Change oscillator.

    https://school.stockcharts.com/doku.php?id=technical_indicators:rate_of_change_roc_and_momentum

    Args:
        close(pandas.Series): dataset 'Close' column.
        window(int): n period.
        fillna(bool): if True, fill nan values.
    """

    def __init__(self, close: pd.Series, window: int = 12, fillna: bool = False):
        self._close = close
        self._window = window
        self._fillna = fillna
        self._run()

    def _run(self):
        self._roc = (
            (self._close - self._close.shift(self._window))
            / self._close.shift(self._window)
        ) * 100

    def roc(self) -> pd.Series:
        """Rate of Change (ROC)

        Returns:
            pandas.Series: New feature generated.
        """
        roc_series = self._check_fillna(self._roc)
        return pd.Series(roc_series, name="roc")


class AwesomeOscillatorIndicator(IndicatorMixin):
    """Awesome Oscillator

    From: https://www.tradingview.com/wiki/Awesome_Oscillator_(AO)

    The Awesome Oscillator is an indicator used to measure market momentum. AO
    calculates the difference of a 34 Period and 5 Period Simple Moving
    Averages. The Simple Moving Averages that are used are not calculated
    using closing price but rather each bar's midpoints. AO is generally used
    to affirm trends or to anticipate possible reversals.

    From: https://www.ifcm.co.uk/ntx-indicators/awesome-oscillator

    Awesome Oscillator is a 34-period simple moving average, plotted through
    the central points of the bars (H+L)/2, and subtracted from the 5-period
    simple moving average, graphed across the central points of the bars
    (H+L)/2.

    MEDIAN PRICE = (HIGH+LOW)/2

    AO = SMA(MEDIAN PRICE, 5)-SMA(MEDIAN PRICE, 34)

    where

    SMA — Simple Moving Average.

    Args:
        high(pandas.Series): dataset 'High' column.
        low(pandas.Series): dataset 'Low' column.
        window1(int): short period.
        window2(int): long period.
        fillna(bool): if True, fill nan values with -50.
    """

    def __init__(
        self,
        high: pd.Series,
        low: pd.Series,
        window1: int = 5,
        window2: int = 34,
        fillna: bool = False,
    ):
        self._high = high
        self._low = low
        self._window1 = window1
        self._window2 = window2
        self._fillna = fillna
        self._run()

    def _run(self):
        median_price = 0.5 * (self._high + self._low)
        min_periods_s = 0 if self._fillna else self._window1
        min_periods_len = 0 if self._fillna else self._window2
        self._ao = (
            median_price.rolling(self._window1, min_periods=min_periods_s).mean()
            - median_price.rolling(self._window2, min_periods=min_periods_len).mean()
        )

    def awesome_oscillator(self) -> pd.Series:
        """Awesome Oscillator

        Returns:
            pandas.Series: New feature generated.
        """
        ao_series = self._check_fillna(self._ao, value=0)
        return pd.Series(ao_series, name="ao")


class WilliamsRIndicator(IndicatorMixin):
    """Williams %R

    Developed by Larry Williams, Williams %R is a momentum indicator that is
    the inverse of the Fast Stochastic Oscillator. Also referred to as %R,
    Williams %R reflects the level of the close relative to the highest high
    for the look-back period. In contrast, the Stochastic Oscillator reflects
    the level of the close relative to the lowest low. %R corrects for the
    inversion by multiplying the raw value by -100. As a result, the Fast
    Stochastic Oscillator and Williams %R produce the exact same lines, only
    the scaling is different. Williams %R oscillates from 0 to -100.

    Readings from 0 to -20 are considered overbought. Readings from -80 to -100
    are considered oversold.

    Unsurprisingly, signals derived from the Stochastic Oscillator are also
    applicable to Williams %R.

    %R = (Highest High - Close)/(Highest High - Lowest Low) * -100

    Lowest Low = lowest low for the look-back period
    Highest High = highest high for the look-back period
    %R is multiplied by -100 correct the inversion and move the decimal.

    https://school.stockcharts.com/doku.php?id=technical_indicators:williams_r

    The Williams %R oscillates from 0 to -100. When the indicator produces
    readings from 0 to -20, this indicates overbought market conditions. When
    readings are -80 to -100, it indicates oversold market conditions.

    Args:
        high(pandas.Series): dataset 'High' column.
        low(pandas.Series): dataset 'Low' column.
        close(pandas.Series): dataset 'Close' column.
        lbp(int): lookback period.
        fillna(bool): if True, fill nan values with -50.
    """

    def __init__(
        self,
        high: pd.Series,
        low: pd.Series,
        close: pd.Series,
        lbp: int = 14,
        fillna: bool = False,
    ):
        self._high = high
        self._low = low
        self._close = close
        self._lbp = lbp
        self._fillna = fillna
        self._run()

    def _run(self):
        min_periods = 0 if self._fillna else self._lbp
        highest_high = self._high.rolling(
            self._lbp, min_periods=min_periods
        ).max()  # highest high over lookback period lbp
        lowest_low = self._low.rolling(
            self._lbp, min_periods=min_periods
        ).min()  # lowest low over lookback period lbp
        self._wr = -100 * (highest_high - self._close) / (highest_high - lowest_low)

    def williams_r(self) -> pd.Series:
        """Williams %R

        Returns:
            pandas.Series: New feature generated.
        """
        wr_series = self._check_fillna(self._wr, value=-50)
        return pd.Series(wr_series, name="wr")


class StochRSIIndicator(IndicatorMixin):
    """Stochastic RSI

    The StochRSI oscillator was developed to take advantage of both momentum
    indicators in order to create a more sensitive indicator that is attuned to
    a specific security's historical performance rather than a generalized analysis
    of price change.

    https://school.stockcharts.com/doku.php?id=technical_indicators:stochrsi
    https://www.investopedia.com/terms/s/stochrsi.asp

    Args:
        close(pandas.Series): dataset 'Close' column.
        window(int): n period
        smooth1(int): moving average of Stochastic RSI
        smooth2(int): moving average of %K
        fillna(bool): if True, fill nan values.
    """

    def __init__(
        self,
        close: pd.Series,
        window: int = 14,
        smooth1: int = 3,
        smooth2: int = 3,
        fillna: bool = False,
    ):
        self._close = close
        self._window = window
        self._smooth1 = smooth1
        self._smooth2 = smooth2
        self._fillna = fillna
        self._run()

    def _run(self):
        self._rsi = RSIIndicator(
            close=self._close, window=self._window, fillna=self._fillna
        ).rsi()
        lowest_low_rsi = self._rsi.rolling(self._window).min()
        self._stochrsi = (self._rsi - lowest_low_rsi) / (
            self._rsi.rolling(self._window).max() - lowest_low_rsi
        )
        self._stochrsi_k = self._stochrsi.rolling(self._smooth1).mean()

    def stochrsi(self):
        """Stochastic RSI

        Returns:
            pandas.Series: New feature generated.
        """
        stochrsi_series = self._check_fillna(self._stochrsi)
        return pd.Series(stochrsi_series, name="stochrsi")

    def stochrsi_k(self):
        """Stochastic RSI %k

        Returns:
            pandas.Series: New feature generated.
        """
        stochrsi_k_series = self._check_fillna(self._stochrsi_k)
        return pd.Series(stochrsi_k_series, name="stochrsi_k")

    def stochrsi_d(self):
        """Stochastic RSI %d

        Returns:
            pandas.Series: New feature generated.
        """
        stochrsi_d_series = self._stochrsi_k.rolling(self._smooth2).mean()
        stochrsi_d_series = self._check_fillna(stochrsi_d_series)
        return pd.Series(stochrsi_d_series, name="stochrsi_d")


class PercentagePriceOscillator(IndicatorMixin):
    """
    The Percentage Price Oscillator (PPO) is a momentum oscillator that measures
    the difference between two moving averages as a percentage of the larger moving average.

    https://school.stockcharts.com/doku.php?id=technical_indicators:price_oscillators_ppo

    Args:
        close(pandas.Series): dataset 'Price' column.
        window_slow(int): n period long-term.
        window_fast(int): n period short-term.
        window_sign(int): n period to signal.
        fillna(bool): if True, fill nan values.
    """

    def __init__(
        self,
        close: pd.Series,
        window_slow: int = 26,
        window_fast: int = 12,
        window_sign: int = 9,
        fillna: bool = False,
    ):
        self._close = close
        self._window_slow = window_slow
        self._window_fast = window_fast
        self._window_sign = window_sign
        self._fillna = fillna
        self._run()

    def _run(self):
        _emafast = _ema(self._close, self._window_fast, self._fillna)
        _emaslow = _ema(self._close, self._window_slow, self._fillna)
        self._ppo = ((_emafast - _emaslow) / _emaslow) * 100
        self._ppo_signal = _ema(self._ppo, self._window_sign, self._fillna)
        self._ppo_hist = self._ppo - self._ppo_signal

    def ppo(self):
        """Percentage Price Oscillator Line

        Returns:
            pandas.Series: New feature generated.
        """
        ppo_series = self._check_fillna(self._ppo, value=0)
        return pd.Series(
            ppo_series, name=f"PPO_{self._window_fast}_{self._window_slow}"
        )

    def ppo_signal(self):
        """Percentage Price Oscillator Signal Line

        Returns:
            pandas.Series: New feature generated.
        """

        ppo_signal_series = self._check_fillna(self._ppo_signal, value=0)
        return pd.Series(
            ppo_signal_series, name=f"PPO_sign_{self._window_fast}_{self._window_slow}"
        )

    def ppo_hist(self):
        """Percentage Price Oscillator Histogram

        Returns:
            pandas.Series: New feature generated.
        """

        ppo_hist_series = self._check_fillna(self._ppo_hist, value=0)
        return pd.Series(
            ppo_hist_series, name=f"PPO_hist_{self._window_fast}_{self._window_slow}"
        )


class PercentageVolumeOscillator(IndicatorMixin):
    """
    The Percentage Volume Oscillator (PVO) is a momentum oscillator for volume.
    The PVO measures the difference between two volume-based moving averages as a
    percentage of the larger moving average.

    https://school.stockcharts.com/doku.php?id=technical_indicators:percentage_volume_oscillator_pvo

    Args:
        volume(pandas.Series): dataset 'Volume' column.
        window_slow(int): n period long-term.
        window_fast(int): n period short-term.
        window_sign(int): n period to signal.
        fillna(bool): if True, fill nan values.
    """

    def __init__(
        self,
        volume: pd.Series,
        window_slow: int = 26,
        window_fast: int = 12,
        window_sign: int = 9,
        fillna: bool = False,
    ):
        self._volume = volume
        self._window_slow = window_slow
        self._window_fast = window_fast
        self._window_sign = window_sign
        self._fillna = fillna
        self._run()

    def _run(self):
        _emafast = _ema(self._volume, self._window_fast, self._fillna)
        _emaslow = _ema(self._volume, self._window_slow, self._fillna)
        self._pvo = ((_emafast - _emaslow) / _emaslow) * 100
        self._pvo_signal = _ema(self._pvo, self._window_sign, self._fillna)
        self._pvo_hist = self._pvo - self._pvo_signal

    def pvo(self) -> pd.Series:
        """PVO Line

        Returns:
            pandas.Series: New feature generated.
        """
        pvo_series = self._check_fillna(self._pvo, value=0)
        return pd.Series(
            pvo_series, name=f"PVO_{self._window_fast}_{self._window_slow}"
        )

    def pvo_signal(self) -> pd.Series:
        """Signal Line

        Returns:
            pandas.Series: New feature generated.
        """

        pvo_signal_series = self._check_fillna(self._pvo_signal, value=0)
        return pd.Series(
            pvo_signal_series, name=f"PVO_sign_{self._window_fast}_{self._window_slow}"
        )

    def pvo_hist(self) -> pd.Series:
        """Histgram

        Returns:
            pandas.Series: New feature generated.
        """

        pvo_hist_series = self._check_fillna(self._pvo_hist, value=0)
        return pd.Series(
            pvo_hist_series, name=f"PVO_hist_{self._window_fast}_{self._window_slow}"
        )


def rsi(close, window=14, fillna=False) -> pd.Series:
    """Relative Strength Index (RSI)

    Compares the magnitude of recent gains and losses over a specified time
    period to measure speed and change of price movements of a security. It is
    primarily used to attempt to identify overbought or oversold conditions in
    the trading of an asset.

    https://www.investopedia.com/terms/r/rsi.asp

    Args:
        close(pandas.Series): dataset 'Close' column.
        window(int): n period.
        fillna(bool): if True, fill nan values.

    Returns:
        pandas.Series: New feature generated.
    """
    return RSIIndicator(close=close, window=window, fillna=fillna).rsi()


def tsi(close, window_slow=25, window_fast=13, fillna=False) -> pd.Series:
    """True strength index (TSI)

    Shows both trend direction and overbought/oversold conditions.

    https://en.wikipedia.org/wiki/True_strength_index

    Args:
        close(pandas.Series): dataset 'Close' column.
        window_slow(int): high period.
        window_fast(int): low period.
        fillna(bool): if True, fill nan values.

    Returns:
        pandas.Series: New feature generated.
    """
    return TSIIndicator(
        close=close, window_slow=window_slow, window_fast=window_fast, fillna=fillna
    ).tsi()


def ultimate_oscillator(
    high,
    low,
    close,
    window1=7,
    window2=14,
    window3=28,
    weight1=4.0,
    weight2=2.0,
    weight3=1.0,
    fillna=False,
) -> pd.Series:
    """Ultimate Oscillator

    Larry Williams' (1976) signal, a momentum oscillator designed to capture
    momentum across three different timeframes.

    http://stockcharts.com/school/doku.php?id=chart_school:technical_indicators:ultimate_oscillator

    BP = Close - Minimum(Low or Prior Close).
    TR = Maximum(High or Prior Close)  -  Minimum(Low or Prior Close)
    Average7 = (7-period BP Sum) / (7-period TR Sum)
    Average14 = (14-period BP Sum) / (14-period TR Sum)
    Average28 = (28-period BP Sum) / (28-period TR Sum)

    UO = 100 x [(4 x Average7)+(2 x Average14)+Average28]/(4+2+1)

    Args:
        high(pandas.Series): dataset 'High' column.
        low(pandas.Series): dataset 'Low' column.
        close(pandas.Series): dataset 'Close' column.
        window1(int): short period.
        window2(int): medium period.
        window3(int): long period.
        weight1(float): weight of short BP average for UO.
        weight2(float): weight of medium BP average for UO.
        weight3(float): weight of long BP average for UO.
        fillna(bool): if True, fill nan values with 50.

    Returns:
        pandas.Series: New feature generated.

    """
    return UltimateOscillator(
        high=high,
        low=low,
        close=close,
        window1=window1,
        window2=window2,
        window3=window3,
        weight1=weight1,
        weight2=weight2,
        weight3=weight3,
        fillna=fillna,
    ).ultimate_oscillator()


def stoch(high, low, close, window=14, smooth_window=3, fillna=False) -> pd.Series:
    """Stochastic Oscillator

    Developed in the late 1950s by George Lane. The stochastic
    oscillator presents the location of the closing price of a
    stock in relation to the high and low range of the price
    of a stock over a period of time, typically a 14-day period.

    https://www.investopedia.com/terms/s/stochasticoscillator.asp

    Args:
        high(pandas.Series): dataset 'High' column.
        low(pandas.Series): dataset 'Low' column.
        close(pandas.Series): dataset 'Close' column.
        window(int): n period.
        smooth_window(int): sma period over stoch_k
        fillna(bool): if True, fill nan values.

    Returns:
        pandas.Series: New feature generated.
    """

    return StochasticOscillator(
        high=high,
        low=low,
        close=close,
        window=window,
        smooth_window=smooth_window,
        fillna=fillna,
    ).stoch()


def stoch_signal(
    high, low, close, window=14, smooth_window=3, fillna=False
) -> pd.Series:
    """Stochastic Oscillator Signal

    Shows SMA of Stochastic Oscillator. Typically a 3 day SMA.

    https://www.investopedia.com/terms/s/stochasticoscillator.asp

    Args:
        high(pandas.Series): dataset 'High' column.
        low(pandas.Series): dataset 'Low' column.
        close(pandas.Series): dataset 'Close' column.
        window(int): n period.
        smooth_window(int): sma period over stoch_k
        fillna(bool): if True, fill nan values.

    Returns:
        pandas.Series: New feature generated.
    """
    return StochasticOscillator(
        high=high,
        low=low,
        close=close,
        window=window,
        smooth_window=smooth_window,
        fillna=fillna,
    ).stoch_signal()


def williams_r(high, low, close, lbp=14, fillna=False) -> pd.Series:
    """Williams %R

    From: http://stockcharts.com/school/doku.php?id=chart_school:technical_indicators:williams_r

    Developed by Larry Williams, Williams %R is a momentum indicator that is
    the inverse of the Fast Stochastic Oscillator. Also referred to as %R,
    Williams %R reflects the level of the close relative to the highest high
    for the look-back period. In contrast, the Stochastic Oscillator reflects
    the level of the close relative to the lowest low. %R corrects for the
    inversion by multiplying the raw value by -100. As a result, the Fast
    Stochastic Oscillator and Williams %R produce the exact same lines, only
    the scaling is different. Williams %R oscillates from 0 to -100.

    Readings from 0 to -20 are considered overbought. Readings from -80 to -100
    are considered oversold.

    Unsurprisingly, signals derived from the Stochastic Oscillator are also
    applicable to Williams %R.

    %R = (Highest High - Close)/(Highest High - Lowest Low) * -100

    Lowest Low = lowest low for the look-back period
    Highest High = highest high for the look-back period
    %R is multiplied by -100 correct the inversion and move the decimal.

    From: https://www.investopedia.com/terms/w/williamsr.asp
    The Williams %R oscillates from 0 to -100. When the indicator produces
    readings from 0 to -20, this indicates overbought market conditions. When
    readings are -80 to -100, it indicates oversold market conditions.

    Args:
        high(pandas.Series): dataset 'High' column.
        low(pandas.Series): dataset 'Low' column.
        close(pandas.Series): dataset 'Close' column.
        lbp(int): lookback period.
        fillna(bool): if True, fill nan values with -50.

    Returns:
        pandas.Series: New feature generated.
    """
    return WilliamsRIndicator(
        high=high, low=low, close=close, lbp=lbp, fillna=fillna
    ).williams_r()


def awesome_oscillator(high, low, window1=5, window2=34, fillna=False) -> pd.Series:
    """Awesome Oscillator

    From: https://www.tradingview.com/wiki/Awesome_Oscillator_(AO)

    The Awesome Oscillator is an indicator used to measure market momentum. AO
    calculates the difference of a 34 Period and 5 Period Simple Moving
    Averages. The Simple Moving Averages that are used are not calculated
    using closing price but rather each bar's midpoints. AO is generally used
    to affirm trends or to anticipate possible reversals.

    From: https://www.ifcm.co.uk/ntx-indicators/awesome-oscillator

    Awesome Oscillator is a 34-period simple moving average, plotted through
    the central points of the bars (H+L)/2, and subtracted from the 5-period
    simple moving average, graphed across the central points of the bars
    (H+L)/2.

    MEDIAN PRICE = (HIGH+LOW)/2

    AO = SMA(MEDIAN PRICE, 5)-SMA(MEDIAN PRICE, 34)

    where

    SMA — Simple Moving Average.

    Args:
        high(pandas.Series): dataset 'High' column.
        low(pandas.Series): dataset 'Low' column.
        window1(int): short period.
        window2(int): long period.
        fillna(bool): if True, fill nan values with -50.

    Returns:
        pandas.Series: New feature generated.
    """
    return AwesomeOscillatorIndicator(
        high=high, low=low, window1=window1, window2=window2, fillna=fillna
    ).awesome_oscillator()


def kama(close, window=10, pow1=2, pow2=30, fillna=False) -> pd.Series:
    """Kaufman's Adaptive Moving Average (KAMA)

    Moving average designed to account for market noise or volatility. KAMA
    will closely follow prices when the price swings are relatively small and
    the noise is low. KAMA will adjust when the price swings widen and follow
    prices from a greater distance. This trend-following indicator can be
    used to identify the overall trend, time turning points and filter price
    movements.

    https://www.tradingview.com/ideas/kama/

    Args:
        close(pandas.Series): dataset 'Close' column.
        window(int): n number of periods for the efficiency ratio.
        pow1(int): number of periods for the fastest EMA constant.
        pow2(int): number of periods for the slowest EMA constant.
        fillna(bool): if True, fill nan values.

    Returns:
        pandas.Series: New feature generated.
    """
    return KAMAIndicator(
        close=close, window=window, pow1=pow1, pow2=pow2, fillna=fillna
    ).kama()


def roc(close: pd.Series, window: int = 12, fillna: bool = False) -> pd.Series:
    """Rate of Change (ROC)

    The Rate-of-Change (ROC) indicator, which is also referred to as simply
    Momentum, is a pure momentum oscillator that measures the percent change in
    price from one period to the next. The ROC calculation compares the current
    price with the price “n” periods ago. The plot forms an oscillator that
    fluctuates above and below the zero line as the Rate-of-Change moves from
    positive to negative. As a momentum oscillator, ROC signals include
    centerline crossovers, divergences and overbought-oversold readings.
    Divergences fail to foreshadow reversals more often than not, so this
    article will forgo a detailed discussion on them. Even though centerline
    crossovers are prone to whipsaw, especially short-term, these crossovers
    can be used to identify the overall trend. Identifying overbought or
    oversold extremes comes naturally to the Rate-of-Change oscillator.

    https://school.stockcharts.com/doku.php?id=technical_indicators:rate_of_change_roc_and_momentum

    Args:
        close(pandas.Series): dataset 'Close' column.
        window(int): n periods.
        fillna(bool): if True, fill nan values.

    Returns:
        pandas.Series: New feature generated.

    """
    return ROCIndicator(close=close, window=window, fillna=fillna).roc()


def stochrsi(
    close: pd.Series,
    window: int = 14,
    smooth1: int = 3,
    smooth2: int = 3,
    fillna: bool = False,
) -> pd.Series:
    """Stochastic RSI

    The StochRSI oscillator was developed to take advantage of both momentum
    indicators in order to create a more sensitive indicator that is attuned to
    a specific security's historical performance rather than a generalized analysis
    of price change.

    https://www.investopedia.com/terms/s/stochrsi.asp

    Args:
        close(pandas.Series): dataset 'Close' column.
        window(int): n period
        smooth1(int): moving average of Stochastic RSI
        smooth2(int): moving average of %K
        fillna(bool): if True, fill nan values.
    Returns:
            pandas.Series: New feature generated.
    """
    return StochRSIIndicator(
        close=close, window=window, smooth1=smooth1, smooth2=smooth2, fillna=fillna
    ).stochrsi()


def stochrsi_k(
    close: pd.Series,
    window: int = 14,
    smooth1: int = 3,
    smooth2: int = 3,
    fillna: bool = False,
) -> pd.Series:
    """Stochastic RSI %k

    The StochRSI oscillator was developed to take advantage of both momentum
    indicators in order to create a more sensitive indicator that is attuned to
    a specific security's historical performance rather than a generalized analysis
    of price change.

    https://www.investopedia.com/terms/s/stochrsi.asp

    Args:
        close(pandas.Series): dataset 'Close' column.
        window(int): n period
        smooth1(int): moving average of Stochastic RSI
        smooth2(int): moving average of %K
        fillna(bool): if True, fill nan values.
    Returns:
            pandas.Series: New feature generated.
    """
    return StochRSIIndicator(
        close=close, window=window, smooth1=smooth1, smooth2=smooth2, fillna=fillna
    ).stochrsi_k()


def stochrsi_d(
    close: pd.Series,
    window: int = 14,
    smooth1: int = 3,
    smooth2: int = 3,
    fillna: bool = False,
) -> pd.Series:
    """Stochastic RSI %d

    The StochRSI oscillator was developed to take advantage of both momentum
    indicators in order to create a more sensitive indicator that is attuned to
    a specific security's historical performance rather than a generalized analysis
    of price change.

    https://www.investopedia.com/terms/s/stochrsi.asp

    Args:
        close(pandas.Series): dataset 'Close' column.
        window(int): n period
        smooth1(int): moving average of Stochastic RSI
        smooth2(int): moving average of %K
        fillna(bool): if True, fill nan values.
    Returns:
            pandas.Series: New feature generated.
    """
    return StochRSIIndicator(
        close=close, window=window, smooth1=smooth1, smooth2=smooth2, fillna=fillna
    ).stochrsi_d()


def ppo(
    close: pd.Series,
    window_slow: int = 26,
    window_fast: int = 12,
    window_sign: int = 9,
    fillna: bool = False,
) -> pd.Series:
    """
    The Percentage Price Oscillator (PPO) is a momentum oscillator that measures
    the difference between two moving averages as a percentage of the larger moving average.

    https://school.stockcharts.com/doku.php?id=technical_indicators:price_oscillators_ppo

    Args:
        close(pandas.Series): dataset 'Price' column.
        window_slow(int): n period long-term.
        window_fast(int): n period short-term.
        window_sign(int): n period to signal.
        fillna(bool): if True, fill nan values.
    Returns:
        pandas.Series: New feature generated.
    """
    return PercentagePriceOscillator(
        close=close,
        window_slow=window_slow,
        window_fast=window_fast,
        window_sign=window_sign,
        fillna=fillna,
    ).ppo()


def ppo_signal(
    close: pd.Series, window_slow=26, window_fast=12, window_sign=9, fillna=False
) -> pd.Series:
    """
    The Percentage Price Oscillator (PPO) is a momentum oscillator that measures
    the difference between two moving averages as a percentage of the larger moving average.

    https://school.stockcharts.com/doku.php?id=technical_indicators:price_oscillators_ppo

    Args:
        close(pandas.Series): dataset 'Price' column.
        window_slow(int): n period long-term.
        window_fast(int): n period short-term.
        window_sign(int): n period to signal.
        fillna(bool): if True, fill nan values.
    Returns:
        pandas.Series: New feature generated.
    """
    return PercentagePriceOscillator(
        close=close,
        window_slow=window_slow,
        window_fast=window_fast,
        window_sign=window_sign,
        fillna=fillna,
    ).ppo_signal()


def ppo_hist(
    close: pd.Series,
    window_slow: int = 26,
    window_fast: int = 12,
    window_sign: int = 9,
    fillna: bool = False,
) -> pd.Series:
    """
    The Percentage Price Oscillator (PPO) is a momentum oscillator that measures
    the difference between two moving averages as a percentage of the larger moving average.

    https://school.stockcharts.com/doku.php?id=technical_indicators:price_oscillators_ppo

    Args:
        close(pandas.Series): dataset 'Price' column.
        window_slow(int): n period long-term.
        window_fast(int): n period short-term.
        window_sign(int): n period to signal.
        fillna(bool): if True, fill nan values.
    Returns:
        pandas.Series: New feature generated.
    """
    return PercentagePriceOscillator(
        close=close,
        window_slow=window_slow,
        window_fast=window_fast,
        window_sign=window_sign,
        fillna=fillna,
    ).ppo_hist()


def pvo(
    volume: pd.Series,
    window_slow: int = 26,
    window_fast: int = 12,
    window_sign: int = 9,
    fillna: bool = False,
) -> pd.Series:
    """
    The Percentage Volume Oscillator (PVO) is a momentum oscillator for volume.
    The PVO measures the difference between two volume-based moving averages as a
    percentage of the larger moving average.

    https://school.stockcharts.com/doku.php?id=technical_indicators:percentage_volume_oscillator_pvo

    Args:
        volume(pandas.Series): dataset 'Volume' column.
        window_slow(int): n period long-term.
        window_fast(int): n period short-term.
        window_sign(int): n period to signal.
        fillna(bool): if True, fill nan values.
    Returns:
        pandas.Series: New feature generated.
    """

    indicator = PercentageVolumeOscillator(
        volume=volume,
        window_slow=window_slow,
        window_fast=window_fast,
        window_sign=window_sign,
        fillna=fillna,
    )
    return indicator.pvo()


def pvo_signal(
    volume: pd.Series,
    window_slow: int = 26,
    window_fast: int = 12,
    window_sign: int = 9,
    fillna: bool = False,
) -> pd.Series:
    """
    The Percentage Volume Oscillator (PVO) is a momentum oscillator for volume.
    The PVO measures the difference between two volume-based moving averages as a
    percentage of the larger moving average.

    https://school.stockcharts.com/doku.php?id=technical_indicators:percentage_volume_oscillator_pvo

    Args:
        volume(pandas.Series): dataset 'Volume' column.
        window_slow(int): n period long-term.
        window_fast(int): n period short-term.
        window_sign(int): n period to signal.
        fillna(bool): if True, fill nan values.
    Returns:
        pandas.Series: New feature generated.
    """

    indicator = PercentageVolumeOscillator(
        volume=volume,
        window_slow=window_slow,
        window_fast=window_fast,
        window_sign=window_sign,
        fillna=fillna,
    )
    return indicator.pvo_signal()


def pvo_hist(
    volume: pd.Series,
    window_slow: int = 26,
    window_fast: int = 12,
    window_sign: int = 9,
    fillna: bool = False,
) -> pd.Series:
    """
    The Percentage Volume Oscillator (PVO) is a momentum oscillator for volume.
    The PVO measures the difference between two volume-based moving averages as a
    percentage of the larger moving average.

    https://school.stockcharts.com/doku.php?id=technical_indicators:percentage_volume_oscillator_pvo

    Args:
        volume(pandas.Series): dataset 'Volume' column.
        window_slow(int): n period long-term.
        window_fast(int): n period short-term.
        window_sign(int): n period to signal.
        fillna(bool): if True, fill nan values.
    Returns:
        pandas.Series: New feature generated.
    """

    indicator = PercentageVolumeOscillator(
        volume=volume,
        window_slow=window_slow,
        window_fast=window_fast,
        window_sign=window_sign,
        fillna=fillna,
    )
    return indicator.pvo_hist()


# Trend Indicators





class AroonIndicator(IndicatorMixin):
    """Aroon Indicator

    Identify when trends are likely to change direction.

    Aroon Up = ((N - Days Since N-day High) / N) x 100
    Aroon Down = ((N - Days Since N-day Low) / N) x 100
    Aroon Indicator = Aroon Up - Aroon Down

    https://www.investopedia.com/terms/a/aroon.asp

    Args:
        close(pandas.Series): dataset 'Close' column.
        window(int): n period.
        fillna(bool): if True, fill nan values.
    """

    def __init__(self, close: pd.Series, window: int = 25, fillna: bool = False):
        self._close = close
        self._window = window
        self._fillna = fillna
        self._run()

    def _run(self):
        min_periods = 0 if self._fillna else self._window
        rolling_close = self._close.rolling(self._window, min_periods=min_periods)
        self._aroon_up = rolling_close.apply(
            lambda x: float(np.argmax(x) + 1) / self._window * 100, raw=True
        )
        self._aroon_down = rolling_close.apply(
            lambda x: float(np.argmin(x) + 1) / self._window * 100, raw=True
        )

    def aroon_up(self) -> pd.Series:
        """Aroon Up Channel

        Returns:
            pandas.Series: New feature generated.
        """
        aroon_up_series = self._check_fillna(self._aroon_up, value=0)
        return pd.Series(aroon_up_series, name=f"aroon_up_{self._window}")

    def aroon_down(self) -> pd.Series:
        """Aroon Down Channel

        Returns:
            pandas.Series: New feature generated.
        """
        aroon_down_series = self._check_fillna(self._aroon_down, value=0)
        return pd.Series(aroon_down_series, name=f"aroon_down_{self._window}")

    def aroon_indicator(self) -> pd.Series:
        """Aroon Indicator

        Returns:
            pandas.Series: New feature generated.
        """
        aroon_diff = self._aroon_up - self._aroon_down
        aroon_diff = self._check_fillna(aroon_diff, value=0)
        return pd.Series(aroon_diff, name=f"aroon_ind_{self._window}")


class MACD(IndicatorMixin):
    """Moving Average Convergence Divergence (MACD)

    Is a trend-following momentum indicator that shows the relationship between
    two moving averages of prices.

    https://school.stockcharts.com/doku.php?id=technical_indicators:moving_average_convergence_divergence_macd

    Args:
        close(pandas.Series): dataset 'Close' column.
        window_fast(int): n period short-term.
        window_slow(int): n period long-term.
        window_sign(int): n period to signal.
        fillna(bool): if True, fill nan values.
    """

    def __init__(
        self,
        close: pd.Series,
        window_slow: int = 26,
        window_fast: int = 12,
        window_sign: int = 9,
        fillna: bool = False,
    ):
        self._close = close
        self._window_slow = window_slow
        self._window_fast = window_fast
        self._window_sign = window_sign
        self._fillna = fillna
        self._run()

    def _run(self):
        self._emafast = _ema(self._close, self._window_fast, self._fillna)
        self._emaslow = _ema(self._close, self._window_slow, self._fillna)
        self._macd = self._emafast - self._emaslow
        self._macd_signal = _ema(self._macd, self._window_sign, self._fillna)
        self._macd_diff = self._macd - self._macd_signal

    def macd(self) -> pd.Series:
        """MACD Line

        Returns:
            pandas.Series: New feature generated.
        """
        macd_series = self._check_fillna(self._macd, value=0)
        return pd.Series(
            macd_series, name=f"MACD_{self._window_fast}_{self._window_slow}"
        )

    def macd_signal(self) -> pd.Series:
        """Signal Line

        Returns:
            pandas.Series: New feature generated.
        """

        macd_signal_series = self._check_fillna(self._macd_signal, value=0)
        return pd.Series(
            macd_signal_series,
            name=f"MACD_sign_{self._window_fast}_{self._window_slow}",
        )

    def macd_diff(self) -> pd.Series:
        """MACD Histogram

        Returns:
            pandas.Series: New feature generated.
        """
        macd_diff_series = self._check_fillna(self._macd_diff, value=0)
        return pd.Series(
            macd_diff_series, name=f"MACD_diff_{self._window_fast}_{self._window_slow}"
        )


class EMAIndicator(IndicatorMixin):
    """EMA - Exponential Moving Average

    Args:
        close(pandas.Series): dataset 'Close' column.
        window(int): n period.
        fillna(bool): if True, fill nan values.
    """

    def __init__(self, close: pd.Series, window: int = 14, fillna: bool = False):
        self._close = close
        self._window = window
        self._fillna = fillna

    def ema_indicator(self) -> pd.Series:
        """Exponential Moving Average (EMA)

        Returns:
            pandas.Series: New feature generated.
        """
        ema_ = _ema(self._close, self._window, self._fillna)
        return pd.Series(ema_, name=f"ema_{self._window}")


class SMAIndicator(IndicatorMixin):
    """SMA - Simple Moving Average

    Args:
        close(pandas.Series): dataset 'Close' column.
        window(int): n period.
        fillna(bool): if True, fill nan values.
    """

    def __init__(self, close: pd.Series, window: int, fillna: bool = False):
        self._close = close
        self._window = window
        self._fillna = fillna

    def sma_indicator(self) -> pd.Series:
        """Simple Moving Average (SMA)

        Returns:
            pandas.Series: New feature generated.
        """
        sma_ = _sma(self._close, self._window, self._fillna)
        return pd.Series(sma_, name=f"sma_{self._window}")


class WMAIndicator(IndicatorMixin):
    """WMA - Weighted Moving Average

    Args:
        close(pandas.Series): dataset 'Close' column.
        window(int): n period.
        fillna(bool): if True, fill nan values.
    """

    def __init__(self, close: pd.Series, window: int = 9, fillna: bool = False):
        self._close = close
        self._window = window
        self._fillna = fillna
        self._run()

    def _run(self):
        _weight = pd.Series(
            [
                i * 2 / (self._window * (self._window + 1))
                for i in range(1, self._window + 1)
            ]
        )

        def weighted_average(weight):
            def _weighted_average(x):
                return (weight * x).sum()

            return _weighted_average

        self._wma = self._close.rolling(self._window).apply(
            weighted_average(_weight), raw=True
        )

    def wma(self) -> pd.Series:
        """Weighted Moving Average (WMA)

        Returns:
            pandas.Series: New feature generated.
        """
        wma = self._check_fillna(self._wma, value=0)
        return pd.Series(wma, name=f"wma_{self._window}")


class TRIXIndicator(IndicatorMixin):
    """Trix (TRIX)

    Shows the percent rate of change of a triple exponentially smoothed moving
    average.

    http://stockcharts.com/school/doku.php?id=chart_school:technical_indicators:trix

    Args:
        close(pandas.Series): dataset 'Close' column.
        window(int): n period.
        fillna(bool): if True, fill nan values.
    """

    def __init__(self, close: pd.Series, window: int = 15, fillna: bool = False):
        self._close = close
        self._window = window
        self._fillna = fillna
        self._run()

    def _run(self):
        ema1 = _ema(self._close, self._window, self._fillna)
        ema2 = _ema(ema1, self._window, self._fillna)
        ema3 = _ema(ema2, self._window, self._fillna)
        self._trix = (ema3 - ema3.shift(1, fill_value=ema3.mean())) / ema3.shift(
            1, fill_value=ema3.mean()
        )
        self._trix *= 100

    def trix(self) -> pd.Series:
        """Trix (TRIX)

        Returns:
            pandas.Series: New feature generated.
        """
        trix_series = self._check_fillna(self._trix, value=0)
        return pd.Series(trix_series, name=f"trix_{self._window}")


class MassIndex(IndicatorMixin):
    """Mass Index (MI)

    It uses the high-low range to identify trend reversals based on range
    expansions. It identifies range bulges that can foreshadow a reversal of
    the current trend.

    http://stockcharts.com/school/doku.php?id=chart_school:technical_indicators:mass_index

    Args:
        high(pandas.Series): dataset 'High' column.
        low(pandas.Series): dataset 'Low' column.
        window_fast(int): fast period value.
        window_slow(int): slow period value.
        fillna(bool): if True, fill nan values.
    """

    def __init__(
        self,
        high: pd.Series,
        low: pd.Series,
        window_fast: int = 9,
        window_slow: int = 25,
        fillna: bool = False,
    ):
        self._high = high
        self._low = low
        self._window_fast = window_fast
        self._window_slow = window_slow
        self._fillna = fillna
        self._run()

    def _run(self):
        min_periods = 0 if self._fillna else self._window_slow
        amplitude = self._high - self._low
        ema1 = _ema(amplitude, self._window_fast, self._fillna)
        ema2 = _ema(ema1, self._window_fast, self._fillna)
        mass = ema1 / ema2
        self._mass = mass.rolling(self._window_slow, min_periods=min_periods).sum()

    def mass_index(self) -> pd.Series:
        """Mass Index (MI)

        Returns:
            pandas.Series: New feature generated.
        """
        mass = self._check_fillna(self._mass, value=0)
        return pd.Series(
            mass, name=f"mass_index_{self._window_fast}_{self._window_slow}"
        )


class IchimokuIndicator(IndicatorMixin):
    """Ichimoku Kinkō Hyō (Ichimoku)

    http://stockcharts.com/school/doku.php?id=chart_school:technical_indicators:ichimoku_cloud

    Args:
        high(pandas.Series): dataset 'High' column.
        low(pandas.Series): dataset 'Low' column.
        window1(int): n1 low period.
        window2(int): n2 medium period.
        window3(int): n3 high period.
        visual(bool): if True, shift n2 values.
        fillna(bool): if True, fill nan values.
    """

    def __init__(
        self,
        high: pd.Series,
        low: pd.Series,
        window1: int = 9,
        window2: int = 26,
        window3: int = 52,
        visual: bool = False,
        fillna: bool = False,
    ):
        self._high = high
        self._low = low
        self._window1 = window1
        self._window2 = window2
        self._window3 = window3
        self._visual = visual
        self._fillna = fillna
        self._run()

    def _run(self):
        min_periods_n1 = 0 if self._fillna else self._window1
        min_periods_n2 = 0 if self._fillna else self._window2
        self._conv = 0.5 * (
            self._high.rolling(self._window1, min_periods=min_periods_n1).max()
            + self._low.rolling(self._window1, min_periods=min_periods_n1).min()
        )
        self._base = 0.5 * (
            self._high.rolling(self._window2, min_periods=min_periods_n2).max()
            + self._low.rolling(self._window2, min_periods=min_periods_n2).min()
        )

    def ichimoku_conversion_line(self) -> pd.Series:
        """Tenkan-sen (Conversion Line)

        Returns:
            pandas.Series: New feature generated.
        """
        conversion = self._check_fillna(self._conv, value=-1)
        return pd.Series(
            conversion, name=f"ichimoku_conv_{self._window1}_{self._window2}"
        )

    def ichimoku_base_line(self) -> pd.Series:
        """Kijun-sen (Base Line)

        Returns:
            pandas.Series: New feature generated.
        """
        base = self._check_fillna(self._base, value=-1)
        return pd.Series(base, name=f"ichimoku_base_{self._window1}_{self._window2}")

    def ichimoku_a(self) -> pd.Series:
        """Senkou Span A (Leading Span A)

        Returns:
            pandas.Series: New feature generated.
        """
        spana = 0.5 * (self._conv + self._base)
        spana = (
            spana.shift(self._window2, fill_value=spana.mean())
            if self._visual
            else spana
        )
        spana = self._check_fillna(spana, value=-1)
        return pd.Series(spana, name=f"ichimoku_a_{self._window1}_{self._window2}")

    def ichimoku_b(self) -> pd.Series:
        """Senkou Span B (Leading Span B)

        Returns:
            pandas.Series: New feature generated.
        """
        spanb = 0.5 * (
            self._high.rolling(self._window3, min_periods=0).max()
            + self._low.rolling(self._window3, min_periods=0).min()
        )
        spanb = (
            spanb.shift(self._window2, fill_value=spanb.mean())
            if self._visual
            else spanb
        )
        spanb = self._check_fillna(spanb, value=-1)
        return pd.Series(spanb, name=f"ichimoku_b_{self._window1}_{self._window2}")


class KSTIndicator(IndicatorMixin):
    """KST Oscillator (KST Signal)

    It is useful to identify major stock market cycle junctures because its
    formula is weighed to be more greatly influenced by the longer and more
    dominant time spans, in order to better reflect the primary swings of stock
    market cycle.

    http://stockcharts.com/school/doku.php?id=chart_school:technical_indicators:know_sure_thing_kst

    Args:
        close(pandas.Series): dataset 'Close' column.
        roc1(int): roc1 period.
        roc2(int): roc2 period.
        roc3(int): roc3 period.
        roc4(int): roc4 period.
        window1(int): n1 smoothed period.
        window2(int): n2 smoothed period.
        window3(int): n3 smoothed period.
        window4(int): n4 smoothed period.
        nsig(int): n period to signal.
        fillna(bool): if True, fill nan values.
    """

    def __init__(
        self,
        close: pd.Series,
        roc1: int = 10,
        roc2: int = 15,
        roc3: int = 20,
        roc4: int = 30,
        window1: int = 10,
        window2: int = 10,
        window3: int = 10,
        window4: int = 15,
        nsig: int = 9,
        fillna: bool = False,
    ):
        self._close = close
        self._r1 = roc1
        self._r2 = roc2
        self._r3 = roc3
        self._r4 = roc4
        self._window1 = window1
        self._window2 = window2
        self._window3 = window3
        self._window4 = window4
        self._nsig = nsig
        self._fillna = fillna
        self._run()

    def _run(self):
        min_periods_n1 = 0 if self._fillna else self._window1
        min_periods_n2 = 0 if self._fillna else self._window2
        min_periods_n3 = 0 if self._fillna else self._window3
        min_periods_n4 = 0 if self._fillna else self._window4
        rocma1 = (
            (
                (
                    self._close
                    - self._close.shift(self._r1, fill_value=self._close.mean())
                )
                / self._close.shift(self._r1, fill_value=self._close.mean())
            )
            .rolling(self._window1, min_periods=min_periods_n1)
            .mean()
        )
        rocma2 = (
            (
                (
                    self._close
                    - self._close.shift(self._r2, fill_value=self._close.mean())
                )
                / self._close.shift(self._r2, fill_value=self._close.mean())
            )
            .rolling(self._window2, min_periods=min_periods_n2)
            .mean()
        )
        rocma3 = (
            (
                (
                    self._close
                    - self._close.shift(self._r3, fill_value=self._close.mean())
                )
                / self._close.shift(self._r3, fill_value=self._close.mean())
            )
            .rolling(self._window3, min_periods=min_periods_n3)
            .mean()
        )
        rocma4 = (
            (
                (
                    self._close
                    - self._close.shift(self._r4, fill_value=self._close.mean())
                )
                / self._close.shift(self._r4, fill_value=self._close.mean())
            )
            .rolling(self._window4, min_periods=min_periods_n4)
            .mean()
        )
        self._kst = 100 * (rocma1 + 2 * rocma2 + 3 * rocma3 + 4 * rocma4)
        self._kst_sig = self._kst.rolling(self._nsig, min_periods=0).mean()

    def kst(self) -> pd.Series:
        """Know Sure Thing (KST)

        Returns:
            pandas.Series: New feature generated.
        """
        kst_series = self._check_fillna(self._kst, value=0)
        return pd.Series(kst_series, name="kst")

    def kst_sig(self) -> pd.Series:
        """Signal Line Know Sure Thing (KST)

        nsig-period SMA of KST

        Returns:
            pandas.Series: New feature generated.
        """
        kst_sig_series = self._check_fillna(self._kst_sig, value=0)
        return pd.Series(kst_sig_series, name="kst_sig")

    def kst_diff(self) -> pd.Series:
        """Diff Know Sure Thing (KST)

        KST - Signal_KST

        Returns:
            pandas.Series: New feature generated.
        """
        kst_diff = self._kst - self._kst_sig
        kst_diff = self._check_fillna(kst_diff, value=0)
        return pd.Series(kst_diff, name="kst_diff")


class DPOIndicator(IndicatorMixin):
    """Detrended Price Oscillator (DPO)

    Is an indicator designed to remove trend from price and make it easier to
    identify cycles.

    http://stockcharts.com/school/doku.php?id=chart_school:technical_indicators:detrended_price_osci

    Args:
        close(pandas.Series): dataset 'Close' column.
        window(int): n period.
        fillna(bool): if True, fill nan values.
    """

    def __init__(self, close: pd.Series, window: int = 20, fillna: bool = False):
        self._close = close
        self._window = window
        self._fillna = fillna
        self._run()

    def _run(self):
        min_periods = 0 if self._fillna else self._window
        self._dpo = (
            self._close.shift(
                int((0.5 * self._window) + 1), fill_value=self._close.mean()
            )
            - self._close.rolling(self._window, min_periods=min_periods).mean()
        )

    def dpo(self) -> pd.Series:
        """Detrended Price Oscillator (DPO)

        Returns:
            pandas.Series: New feature generated.
        """
        dpo_series = self._check_fillna(self._dpo, value=0)
        return pd.Series(dpo_series, name="dpo_" + str(self._window))


class CCIIndicator(IndicatorMixin):
    """Commodity Channel Index (CCI)

    CCI measures the difference between a security's price change and its
    average price change. High positive readings indicate that prices are well
    above their average, which is a show of strength. Low negative readings
    indicate that prices are well below their average, which is a show of
    weakness.

    http://stockcharts.com/school/doku.php?id=chart_school:technical_indicators:commodity_channel_index_cci

    Args:
        high(pandas.Series): dataset 'High' column.
        low(pandas.Series): dataset 'Low' column.
        close(pandas.Series): dataset 'Close' column.
        window(int): n period.
        constant(int): constant.
        fillna(bool): if True, fill nan values.
    """

    def __init__(
        self,
        high: pd.Series,
        low: pd.Series,
        close: pd.Series,
        window: int = 20,
        constant: float = 0.015,
        fillna: bool = False,
    ):
        self._high = high
        self._low = low
        self._close = close
        self._window = window
        self._constant = constant
        self._fillna = fillna
        self._run()

    def _run(self):
        def _mad(x):
            return np.mean(np.abs(x - np.mean(x)))

        min_periods = 0 if self._fillna else self._window
        typical_price = (self._high + self._low + self._close) / 3.0
        self._cci = (
            typical_price
            - typical_price.rolling(self._window, min_periods=min_periods).mean()
        ) / (
            self._constant
            * typical_price.rolling(self._window, min_periods=min_periods).apply(
                _mad, True
            )
        )

    def cci(self) -> pd.Series:
        """Commodity Channel Index (CCI)

        Returns:
            pandas.Series: New feature generated.
        """
        cci_series = self._check_fillna(self._cci, value=0)
        return pd.Series(cci_series, name="cci")


class ADXIndicator(IndicatorMixin):
    """Average Directional Movement Index (ADX)

    The Plus Directional Indicator (+DI) and Minus Directional Indicator (-DI)
    are derived from smoothed averages of these differences, and measure trend
    direction over time. These two indicators are often referred to
    collectively as the Directional Movement Indicator (DMI).

    The Average Directional Index (ADX) is in turn derived from the smoothed
    averages of the difference between +DI and -DI, and measures the strength
    of the trend (regardless of direction) over time.

    Using these three indicators together, chartists can determine both the
    direction and strength of the trend.

    http://stockcharts.com/school/doku.php?id=chart_school:technical_indicators:average_directional_index_adx

    Args:
        high(pandas.Series): dataset 'High' column.
        low(pandas.Series): dataset 'Low' column.
        close(pandas.Series): dataset 'Close' column.
        window(int): n period.
        fillna(bool): if True, fill nan values.
    """

    def __init__(
        self,
        high: pd.Series,
        low: pd.Series,
        close: pd.Series,
        window: int = 14,
        fillna: bool = False,
    ):
        self._high = high
        self._low = low
        self._close = close
        self._window = window
        self._fillna = fillna
        self._run()

    def _run(self):
        if self._window == 0:
            raise ValueError("window may not be 0")

        close_shift = self._close.shift(1)
        pdm = _get_min_max(self._high, close_shift, "max")
        pdn = _get_min_max(self._low, close_shift, "min")
        diff_directional_movement = pdm - pdn

        self._trs_initial = np.zeros(self._window - 1)
        self._trs = np.zeros(len(self._close) - (self._window - 1))
        self._trs[0] = diff_directional_movement.dropna()[0 : self._window].sum()
        diff_directional_movement = diff_directional_movement.reset_index(drop=True)

        for i in range(1, len(self._trs) - 1):
            self._trs[i] = (
                self._trs[i - 1]
                - (self._trs[i - 1] / float(self._window))
                + diff_directional_movement[self._window + i]
            )

        diff_up = self._high - self._high.shift(1)
        diff_down = self._low.shift(1) - self._low
        pos = abs(((diff_up > diff_down) & (diff_up > 0)) * diff_up)
        neg = abs(((diff_down > diff_up) & (diff_down > 0)) * diff_down)

        self._dip = np.zeros(len(self._close) - (self._window - 1))
        self._dip[0] = pos.dropna()[0 : self._window].sum()

        pos = pos.reset_index(drop=True)

        for i in range(1, len(self._dip) - 1):
            self._dip[i] = (
                self._dip[i - 1]
                - (self._dip[i - 1] / float(self._window))
                + pos[self._window + i]
            )

        self._din = np.zeros(len(self._close) - (self._window - 1))
        self._din[0] = neg.dropna()[0 : self._window].sum()

        neg = neg.reset_index(drop=True)

        for i in range(1, len(self._din) - 1):
            self._din[i] = (
                self._din[i - 1]
                - (self._din[i - 1] / float(self._window))
                + neg[self._window + i]
            )

    def adx(self) -> pd.Series:
        """Average Directional Index (ADX)

        Returns:
            pandas.Series: New feature generated.tr
        """
        dip = np.zeros(len(self._trs))
        for i in range(len(self._trs)):
            dip[i] = 100 * (self._dip[i] / self._trs[i])

        din = np.zeros(len(self._trs))
        for i in range(len(self._trs)):
            din[i] = 100 * (self._din[i] / self._trs[i])

        directional_index = 100 * np.abs((dip - din) / (dip + din))

        adx_series = np.zeros(len(self._trs))
        adx_series[self._window] = directional_index[0 : self._window].mean()

        for i in range(self._window + 1, len(adx_series)):
            adx_series[i] = (
                (adx_series[i - 1] * (self._window - 1)) + directional_index[i - 1]
            ) / float(self._window)

        adx_series = np.concatenate((self._trs_initial, adx_series), axis=0)
        adx_series = pd.Series(data=adx_series, index=self._close.index)

        adx_series = self._check_fillna(adx_series, value=20)
        return pd.Series(adx_series, name="adx")

    def adx_pos(self) -> pd.Series:
        """Plus Directional Indicator (+DI)

        Returns:
            pandas.Series: New feature generated.
        """
        dip = np.zeros(len(self._close))
        for i in range(1, len(self._trs) - 1):
            dip[i + self._window] = 100 * (self._dip[i] / self._trs[i])

        adx_pos_series = self._check_fillna(
            pd.Series(dip, index=self._close.index), value=20
        )
        return pd.Series(adx_pos_series, name="adx_pos")

    def adx_neg(self) -> pd.Series:
        """Minus Directional Indicator (-DI)

        Returns:
            pandas.Series: New feature generated.
        """
        din = np.zeros(len(self._close))
        for i in range(1, len(self._trs) - 1):
            din[i + self._window] = 100 * (self._din[i] / self._trs[i])

        adx_neg_series = self._check_fillna(
            pd.Series(din, index=self._close.index), value=20
        )
        return pd.Series(adx_neg_series, name="adx_neg")


class VortexIndicator(IndicatorMixin):
    """Vortex Indicator (VI)

    It consists of two oscillators that capture positive and negative trend
    movement. A bullish signal triggers when the positive trend indicator
    crosses above the negative trend indicator or a key level.

    http://stockcharts.com/school/doku.php?id=chart_school:technical_indicators:vortex_indicator

    Args:
        high(pandas.Series): dataset 'High' column.
        low(pandas.Series): dataset 'Low' column.
        close(pandas.Series): dataset 'Close' column.
        window(int): n period.
        fillna(bool): if True, fill nan values.
    """

    def __init__(
        self,
        high: pd.Series,
        low: pd.Series,
        close: pd.Series,
        window: int = 14,
        fillna: bool = False,
    ):
        self._high = high
        self._low = low
        self._close = close
        self._window = window
        self._fillna = fillna
        self._run()

    def _run(self):
        close_shift = self._close.shift(1, fill_value=self._close.mean())
        true_range = self._true_range(self._high, self._low, close_shift)
        min_periods = 0 if self._fillna else self._window
        trn = true_range.rolling(self._window, min_periods=min_periods).sum()
        vmp = np.abs(self._high - self._low.shift(1))
        vmm = np.abs(self._low - self._high.shift(1))
        self._vip = vmp.rolling(self._window, min_periods=min_periods).sum() / trn
        self._vin = vmm.rolling(self._window, min_periods=min_periods).sum() / trn

    def vortex_indicator_pos(self):
        """+VI

        Returns:
            pandas.Series: New feature generated.
        """
        vip = self._check_fillna(self._vip, value=1)
        return pd.Series(vip, name="vip")

    def vortex_indicator_neg(self):
        """-VI

        Returns:
            pandas.Series: New feature generated.
        """
        vin = self._check_fillna(self._vin, value=1)
        return pd.Series(vin, name="vin")

    def vortex_indicator_diff(self):
        """Diff VI

        Returns:
            pandas.Series: New feature generated.
        """
        vid = self._vip - self._vin
        vid = self._check_fillna(vid, value=0)
        return pd.Series(vid, name="vid")


class PSARIndicator(IndicatorMixin):
    """Parabolic Stop and Reverse (Parabolic SAR)

    The Parabolic Stop and Reverse, more commonly known as the
    Parabolic SAR,is a trend-following indicator developed by
    J. Welles Wilder. The Parabolic SAR is displayed as a single
    parabolic line (or dots) underneath the price bars in an uptrend,
    and above the price bars in a downtrend.

    https://school.stockcharts.com/doku.php?id=technical_indicators:parabolic_sar

    Args:
        high(pandas.Series): dataset 'High' column.
        low(pandas.Series): dataset 'Low' column.
        close(pandas.Series): dataset 'Close' column.
        step(float): the Acceleration Factor used to compute the SAR.
        max_step(float): the maximum value allowed for the Acceleration Factor.
        fillna(bool): if True, fill nan values.
    """

    def __init__(
        self,
        high: pd.Series,
        low: pd.Series,
        close: pd.Series,
        step: float = 0.02,
        max_step: float = 0.20,
        fillna: bool = False,
    ):
        self._high = high
        self._low = low
        self._close = close
        self._step = step
        self._max_step = max_step
        self._fillna = fillna
        self._run()

    def _run(self):  # noqa
        up_trend = True
        acceleration_factor = self._step
        up_trend_high = self._high.iloc[0]
        down_trend_low = self._low.iloc[0]

        self._psar = self._close.copy()
        self._psar_up = pd.Series(index=self._psar.index)
        self._psar_down = pd.Series(index=self._psar.index)

        for i in range(2, len(self._close)):
            reversal = False

            max_high = self._high.iloc[i]
            min_low = self._low.iloc[i]

            if up_trend:
                self._psar.iloc[i] = self._psar.iloc[i - 1] + (
                    acceleration_factor * (up_trend_high - self._psar.iloc[i - 1])
                )

                if min_low < self._psar.iloc[i]:
                    reversal = True
                    self._psar.iloc[i] = up_trend_high
                    down_trend_low = min_low
                    acceleration_factor = self._step
                else:
                    if max_high > up_trend_high:
                        up_trend_high = max_high
                        acceleration_factor = min(
                            acceleration_factor + self._step, self._max_step
                        )

                    low1 = self._low.iloc[i - 1]
                    low2 = self._low.iloc[i - 2]
                    if low2 < self._psar.iloc[i]:
                        self._psar.iloc[i] = low2
                    elif low1 < self._psar.iloc[i]:
                        self._psar.iloc[i] = low1
            else:
                self._psar.iloc[i] = self._psar.iloc[i - 1] - (
                    acceleration_factor * (self._psar.iloc[i - 1] - down_trend_low)
                )

                if max_high > self._psar.iloc[i]:
                    reversal = True
                    self._psar.iloc[i] = down_trend_low
                    up_trend_high = max_high
                    acceleration_factor = self._step
                else:
                    if min_low < down_trend_low:
                        down_trend_low = min_low
                        acceleration_factor = min(
                            acceleration_factor + self._step, self._max_step
                        )

                    high1 = self._high.iloc[i - 1]
                    high2 = self._high.iloc[i - 2]
                    if high2 > self._psar.iloc[i]:
                        self._psar[i] = high2
                    elif high1 > self._psar.iloc[i]:
                        self._psar.iloc[i] = high1

            up_trend = up_trend != reversal  # XOR

            if up_trend:
                self._psar_up.iloc[i] = self._psar.iloc[i]
            else:
                self._psar_down.iloc[i] = self._psar.iloc[i]

    def psar(self) -> pd.Series:
        """PSAR value

        Returns:
            pandas.Series: New feature generated.
        """
        psar_series = self._check_fillna(self._psar, value=-1)
        return pd.Series(psar_series, name="psar")

    def psar_up(self) -> pd.Series:
        """PSAR up trend value

        Returns:
            pandas.Series: New feature generated.
        """
        psar_up_series = self._check_fillna(self._psar_up, value=-1)
        return pd.Series(psar_up_series, name="psarup")

    def psar_down(self) -> pd.Series:
        """PSAR down trend value

        Returns:
            pandas.Series: New feature generated.
        """
        psar_down_series = self._check_fillna(self._psar_down, value=-1)
        return pd.Series(psar_down_series, name="psardown")

    def psar_up_indicator(self) -> pd.Series:
        """PSAR up trend value indicator

        Returns:
            pandas.Series: New feature generated.
        """
        indicator = self._psar_up.where(
            self._psar_up.notnull() & self._psar_up.shift(1).isnull(), 0
        )
        indicator = indicator.where(indicator == 0, 1)
        return pd.Series(indicator, index=self._close.index, name="psariup")

    def psar_down_indicator(self) -> pd.Series:
        """PSAR down trend value indicator

        Returns:
            pandas.Series: New feature generated.
        """
        indicator = self._psar_up.where(
            self._psar_down.notnull() & self._psar_down.shift(1).isnull(), 0
        )
        indicator = indicator.where(indicator == 0, 1)
        return pd.Series(indicator, index=self._close.index, name="psaridown")


class STCIndicator(IndicatorMixin):
    """Schaff Trend Cycle (STC)

    The Schaff Trend Cycle (STC) is a charting indicator that
    is commonly used to identify market trends and provide buy
    and sell signals to traders. Developed in 1999 by noted currency
    trader Doug Schaff, STC is a type of oscillator and is based on
    the assumption that, regardless of time frame, currency trends
    accelerate and decelerate in cyclical patterns.

    https://www.investopedia.com/articles/forex/10/schaff-trend-cycle-indicator.asp

    Args:
        close(pandas.Series): dataset 'Close' column.
        window_fast(int): n period short-term.
        window_slow(int): n period long-term.
        cycle(int): cycle size
        smooth1(int): ema period over stoch_k
        smooth2(int): ema period over stoch_kd
        fillna(bool): if True, fill nan values.
    """

    def __init__(
        self,
        close: pd.Series,
        window_slow: int = 50,
        window_fast: int = 23,
        cycle: int = 10,
        smooth1: int = 3,
        smooth2: int = 3,
        fillna: bool = False,
    ):
        self._close = close
        self._window_slow = window_slow
        self._window_fast = window_fast
        self._cycle = cycle
        self._smooth1 = smooth1
        self._smooth2 = smooth2
        self._fillna = fillna
        self._run()

    def _run(self):

        _emafast = _ema(self._close, self._window_fast, self._fillna)
        _emaslow = _ema(self._close, self._window_slow, self._fillna)
        _macd = _emafast - _emaslow

        _macdmin = _macd.rolling(window=self._cycle).min()
        _macdmax = _macd.rolling(window=self._cycle).max()
        _stoch_k = 100 * (_macd - _macdmin) / (_macdmax - _macdmin)
        _stoch_d = _ema(_stoch_k, self._smooth1, self._fillna)

        _stoch_d_min = _stoch_d.rolling(window=self._cycle).min()
        _stoch_d_max = _stoch_d.rolling(window=self._cycle).max()
        _stoch_kd = 100 * (_stoch_d - _stoch_d_min) / (_stoch_d_max - _stoch_d_min)
        self._stc = _ema(_stoch_kd, self._smooth2, self._fillna)

    def stc(self):
        """Schaff Trend Cycle

        Returns:
            pandas.Series: New feature generated.
        """
        stc_series = self._check_fillna(self._stc)
        return pd.Series(stc_series, name="stc")


def ema_indicator(close, window=12, fillna=False):
    """Exponential Moving Average (EMA)

    Returns:
        pandas.Series: New feature generated.
    """
    return EMAIndicator(close=close, window=window, fillna=fillna).ema_indicator()


def sma_indicator(close, window=12, fillna=False):
    """Simple Moving Average (SMA)

    Returns:
        pandas.Series: New feature generated.
    """
    return SMAIndicator(close=close, window=window, fillna=fillna).sma_indicator()


def wma_indicator(close, window=9, fillna=False):
    """Weighted Moving Average (WMA)

    Returns:
        pandas.Series: New feature generated.
    """
    return WMAIndicator(close=close, window=window, fillna=fillna).wma()


def macd(close, window_slow=26, window_fast=12, fillna=False):
    """Moving Average Convergence Divergence (MACD)

    Is a trend-following momentum indicator that shows the relationship between
    two moving averages of prices.

    https://en.wikipedia.org/wiki/MACD

    Args:
        close(pandas.Series): dataset 'Close' column.
        window_fast(int): n period short-term.
        window_slow(int): n period long-term.
        fillna(bool): if True, fill nan values.

    Returns:
        pandas.Series: New feature generated.
    """
    return MACD(
        close=close,
        window_slow=window_slow,
        window_fast=window_fast,
        window_sign=9,
        fillna=fillna,
    ).macd()


def macd_signal(close, window_slow=26, window_fast=12, window_sign=9, fillna=False):
    """Moving Average Convergence Divergence (MACD Signal)

    Shows EMA of MACD.

    https://en.wikipedia.org/wiki/MACD

    Args:
        close(pandas.Series): dataset 'Close' column.
        window_fast(int): n period short-term.
        window_slow(int): n period long-term.
        window_sign(int): n period to signal.
        fillna(bool): if True, fill nan values.

    Returns:
        pandas.Series: New feature generated.
    """
    return MACD(
        close=close,
        window_slow=window_slow,
        window_fast=window_fast,
        window_sign=window_sign,
        fillna=fillna,
    ).macd_signal()


def macd_diff(close, window_slow=26, window_fast=12, window_sign=9, fillna=False):
    """Moving Average Convergence Divergence (MACD Diff)

    Shows the relationship between MACD and MACD Signal.

    https://en.wikipedia.org/wiki/MACD

    Args:
        close(pandas.Series): dataset 'Close' column.
        window_fast(int): n period short-term.
        window_slow(int): n period long-term.
        window_sign(int): n period to signal.
        fillna(bool): if True, fill nan values.

    Returns:
        pandas.Series: New feature generated.
    """
    return MACD(
        close=close,
        window_slow=window_slow,
        window_fast=window_fast,
        window_sign=window_sign,
        fillna=fillna,
    ).macd_diff()


def adx(high, low, close, window=14, fillna=False):
    """Average Directional Movement Index (ADX)

    The Plus Directional Indicator (+DI) and Minus Directional Indicator (-DI)
    are derived from smoothed averages of these differences, and measure trend
    direction over time. These two indicators are often referred to
    collectively as the Directional Movement Indicator (DMI).

    The Average Directional Index (ADX) is in turn derived from the smoothed
    averages of the difference between +DI and -DI, and measures the strength
    of the trend (regardless of direction) over time.

    Using these three indicators together, chartists can determine both the
    direction and strength of the trend.

    http://stockcharts.com/school/doku.php?id=chart_school:technical_indicators:average_directional_index_adx

    Args:
        high(pandas.Series): dataset 'High' column.
        low(pandas.Series): dataset 'Low' column.
        close(pandas.Series): dataset 'Close' column.
        window(int): n period.
        fillna(bool): if True, fill nan values.

    Returns:
        pandas.Series: New feature generated.
    """
    return ADXIndicator(
        high=high, low=low, close=close, window=window, fillna=fillna
    ).adx()


def adx_pos(high, low, close, window=14, fillna=False):
    """Average Directional Movement Index Positive (ADX)

    The Plus Directional Indicator (+DI) and Minus Directional Indicator (-DI)
    are derived from smoothed averages of these differences, and measure trend
    direction over time. These two indicators are often referred to
    collectively as the Directional Movement Indicator (DMI).

    The Average Directional Index (ADX) is in turn derived from the smoothed
    averages of the difference between +DI and -DI, and measures the strength
    of the trend (regardless of direction) over time.

    Using these three indicators together, chartists can determine both the
    direction and strength of the trend.

    http://stockcharts.com/school/doku.php?id=chart_school:technical_indicators:average_directional_index_adx

    Args:
        high(pandas.Series): dataset 'High' column.
        low(pandas.Series): dataset 'Low' column.
        close(pandas.Series): dataset 'Close' column.
        window(int): n period.
        fillna(bool): if True, fill nan values.

    Returns:
        pandas.Series: New feature generated.
    """
    return ADXIndicator(
        high=high, low=low, close=close, window=window, fillna=fillna
    ).adx_pos()


def adx_neg(high, low, close, window=14, fillna=False):
    """Average Directional Movement Index Negative (ADX)

    The Plus Directional Indicator (+DI) and Minus Directional Indicator (-DI)
    are derived from smoothed averages of these differences, and measure trend
    direction over time. These two indicators are often referred to
    collectively as the Directional Movement Indicator (DMI).

    The Average Directional Index (ADX) is in turn derived from the smoothed
    averages of the difference between +DI and -DI, and measures the strength
    of the trend (regardless of direction) over time.

    Using these three indicators together, chartists can determine both the
    direction and strength of the trend.

    http://stockcharts.com/school/doku.php?id=chart_school:technical_indicators:average_directional_index_adx

    Args:
        high(pandas.Series): dataset 'High' column.
        low(pandas.Series): dataset 'Low' column.
        close(pandas.Series): dataset 'Close' column.
        window(int): n period.
        fillna(bool): if True, fill nan values.

    Returns:
        pandas.Series: New feature generated.
    """
    return ADXIndicator(
        high=high, low=low, close=close, window=window, fillna=fillna
    ).adx_neg()


def vortex_indicator_pos(high, low, close, window=14, fillna=False):
    """Vortex Indicator (VI)

    It consists of two oscillators that capture positive and negative trend
    movement. A bullish signal triggers when the positive trend indicator
    crosses above the negative trend indicator or a key level.

    http://stockcharts.com/school/doku.php?id=chart_school:technical_indicators:vortex_indicator

    Args:
        high(pandas.Series): dataset 'High' column.
        low(pandas.Series): dataset 'Low' column.
        close(pandas.Series): dataset 'Close' column.
        window(int): n period.
        fillna(bool): if True, fill nan values.

    Returns:
        pandas.Series: New feature generated.
    """
    return VortexIndicator(
        high=high, low=low, close=close, window=window, fillna=fillna
    ).vortex_indicator_pos()


def vortex_indicator_neg(high, low, close, window=14, fillna=False):
    """Vortex Indicator (VI)

    It consists of two oscillators that capture positive and negative trend
    movement. A bearish signal triggers when the negative trend indicator
    crosses above the positive trend indicator or a key level.

    http://stockcharts.com/school/doku.php?id=chart_school:technical_indicators:vortex_indicator

    Args:
        high(pandas.Series): dataset 'High' column.
        low(pandas.Series): dataset 'Low' column.
        close(pandas.Series): dataset 'Close' column.
        window(int): n period.
        fillna(bool): if True, fill nan values.

    Returns:
        pandas.Series: New feature generated.
    """
    return VortexIndicator(
        high=high, low=low, close=close, window=window, fillna=fillna
    ).vortex_indicator_neg()


def trix(close, window=15, fillna=False):
    """Trix (TRIX)

    Shows the percent rate of change of a triple exponentially smoothed moving
    average.

    http://stockcharts.com/school/doku.php?id=chart_school:technical_indicators:trix

    Args:
        close(pandas.Series): dataset 'Close' column.
        window(int): n period.
        fillna(bool): if True, fill nan values.

    Returns:
        pandas.Series: New feature generated.
    """
    return TRIXIndicator(close=close, window=window, fillna=fillna).trix()


def mass_index(high, low, window_fast=9, window_slow=25, fillna=False):
    """Mass Index (MI)

    It uses the high-low range to identify trend reversals based on range
    expansions. It identifies range bulges that can foreshadow a reversal of
    the current trend.

    http://stockcharts.com/school/doku.php?id=chart_school:technical_indicators:mass_index

    Args:
        high(pandas.Series): dataset 'High' column.
        low(pandas.Series): dataset 'Low' column.
        window_fast(int): fast window value.
        window_slow(int): slow window value.
        fillna(bool): if True, fill nan values.

    Returns:
        pandas.Series: New feature generated.

    """
    return MassIndex(
        high=high,
        low=low,
        window_fast=window_fast,
        window_slow=window_slow,
        fillna=fillna,
    ).mass_index()


def cci(high, low, close, window=20, constant=0.015, fillna=False):
    """Commodity Channel Index (CCI)

    CCI measures the difference between a security's price change and its
    average price change. High positive readings indicate that prices are well
    above their average, which is a show of strength. Low negative readings
    indicate that prices are well below their average, which is a show of
    weakness.

    http://stockcharts.com/school/doku.php?id=chart_school:technical_indicators:commodity_channel_index_cci

    Args:
        high(pandas.Series): dataset 'High' column.
        low(pandas.Series): dataset 'Low' column.
        close(pandas.Series): dataset 'Close' column.
        window(int): n periods.
        constant(int): constant.
        fillna(bool): if True, fill nan values.

    Returns:
        pandas.Series: New feature generated.

    """
    return CCIIndicator(
        high=high, low=low, close=close, window=window, constant=constant, fillna=fillna
    ).cci()


def dpo(close, window=20, fillna=False):
    """Detrended Price Oscillator (DPO)

    Is an indicator designed to remove trend from price and make it easier to
    identify cycles.

    http://stockcharts.com/school/doku.php?id=chart_school:technical_indicators:detrended_price_osci

    Args:
        close(pandas.Series): dataset 'Close' column.
        window(int): n period.
        fillna(bool): if True, fill nan values.

    Returns:
        pandas.Series: New feature generated.
    """
    return DPOIndicator(close=close, window=window, fillna=fillna).dpo()


def kst(
    close,
    roc1=10,
    roc2=15,
    roc3=20,
    roc4=30,
    window1=10,
    window2=10,
    window3=10,
    window4=15,
    fillna=False,
):
    """KST Oscillator (KST)

    It is useful to identify major stock market cycle junctures because its
    formula is weighed to be more greatly influenced by the longer and more
    dominant time spans, in order to better reflect the primary swings of stock
    market cycle.

    https://en.wikipedia.org/wiki/KST_oscillator

    Args:
        close(pandas.Series): dataset 'Close' column.
        roc1(int): r1 period.
        roc2(int): r2 period.
        roc3(int): r3 period.
        roc4(int): r4 period.
        window1(int): n1 smoothed period.
        window2(int): n2 smoothed period.
        window3(int): n3 smoothed period.
        window4(int): n4 smoothed period.
        fillna(bool): if True, fill nan values.

    Returns:
        pandas.Series: New feature generated.
    """
    return KSTIndicator(
        close=close,
        roc1=roc1,
        roc2=roc2,
        roc3=roc3,
        roc4=roc4,
        window1=window1,
        window2=window2,
        window3=window3,
        window4=window4,
        nsig=9,
        fillna=fillna,
    ).kst()


def stc(
    close, window_slow=50, window_fast=23, cycle=10, smooth1=3, smooth2=3, fillna=False
):
    """Schaff Trend Cycle (STC)

    The Schaff Trend Cycle (STC) is a charting indicator that
    is commonly used to identify market trends and provide buy
    and sell signals to traders. Developed in 1999 by noted currency
    trader Doug Schaff, STC is a type of oscillator and is based on
    the assumption that, regardless of time frame, currency trends
    accelerate and decelerate in cyclical patterns.

    https://www.investopedia.com/articles/forex/10/schaff-trend-cycle-indicator.asp

    Args:
        close(pandas.Series): dataset 'Close' column.
        window_fast(int): n period short-term.
        window_slow(int): n period long-term.
        cycle(int): n period
        smooth1(int): ema period over stoch_k
        smooth2(int): ema period over stoch_kd
        fillna(bool): if True, fill nan values.

    Returns:
        pandas.Series: New feature generated.
    """
    return STCIndicator(
        close=close,
        window_slow=window_slow,
        window_fast=window_fast,
        cycle=cycle,
        smooth1=smooth1,
        smooth2=smooth2,
        fillna=fillna,
    ).stc()


def kst_sig(
    close,
    roc1=10,
    roc2=15,
    roc3=20,
    roc4=30,
    window1=10,
    window2=10,
    window3=10,
    window4=15,
    nsig=9,
    fillna=False,
):
    """KST Oscillator (KST Signal)

    It is useful to identify major stock market cycle junctures because its
    formula is weighed to be more greatly influenced by the longer and more
    dominant time spans, in order to better reflect the primary swings of stock
    market cycle.

    http://stockcharts.com/school/doku.php?id=chart_school:technical_indicators:know_sure_thing_kst

    Args:
        close(pandas.Series): dataset 'Close' column.
        roc1(int): roc1 period.
        roc2(int): roc2 period.
        roc3(int): roc3 period.
        roc4(int): roc4 period.
        window1(int): n1 smoothed period.
        window2(int): n2 smoothed period.
        window3(int): n3 smoothed period.
        window4(int): n4 smoothed period.
        nsig(int): n period to signal.
        fillna(bool): if True, fill nan values.

    Returns:
        pandas.Series: New feature generated.
    """
    return KSTIndicator(
        close=close,
        roc1=roc1,
        roc2=roc2,
        roc3=roc3,
        roc4=roc4,
        window1=window1,
        window2=window2,
        window3=window3,
        window4=window4,
        nsig=nsig,
        fillna=fillna,
    ).kst_sig()


def ichimoku_conversion_line(
    high, low, window1=9, window2=26, visual=False, fillna=False
) -> pd.Series:
    """Tenkan-sen (Conversion Line)

    It identifies the trend and look for potential signals within that trend.

    http://stockcharts.com/school/doku.php?id=chart_school:technical_indicators:ichimoku_cloud

    Args:
        high(pandas.Series): dataset 'High' column.
        low(pandas.Series): dataset 'Low' column.
        window1(int): n1 low period.
        window2(int): n2 medium period.
        visual(bool): if True, shift n2 values.
        fillna(bool): if True, fill nan values.

    Returns:
        pandas.Series: New feature generated.
    """
    return IchimokuIndicator(
        high=high,
        low=low,
        window1=window1,
        window2=window2,
        window3=52,
        visual=visual,
        fillna=fillna,
    ).ichimoku_conversion_line()


def ichimoku_base_line(
    high, low, window1=9, window2=26, visual=False, fillna=False
) -> pd.Series:
    """Kijun-sen (Base Line)

    It identifies the trend and look for potential signals within that trend.

    http://stockcharts.com/school/doku.php?id=chart_school:technical_indicators:ichimoku_cloud

    Args:
        high(pandas.Series): dataset 'High' column.
        low(pandas.Series): dataset 'Low' column.
        window1(int): n1 low period.
        window2(int): n2 medium period.
        visual(bool): if True, shift n2 values.
        fillna(bool): if True, fill nan values.

    Returns:
        pandas.Series: New feature generated.
    """
    return IchimokuIndicator(
        high=high,
        low=low,
        window1=window1,
        window2=window2,
        window3=52,
        visual=visual,
        fillna=fillna,
    ).ichimoku_base_line()


def ichimoku_a(high, low, window1=9, window2=26, visual=False, fillna=False):
    """Ichimoku Kinkō Hyō (Ichimoku)

    It identifies the trend and look for potential signals within that trend.

    http://stockcharts.com/school/doku.php?id=chart_school:technical_indicators:ichimoku_cloud

    Args:
        high(pandas.Series): dataset 'High' column.
        low(pandas.Series): dataset 'Low' column.
        window1(int): n1 low period.
        window2(int): n2 medium period.
        visual(bool): if True, shift n2 values.
        fillna(bool): if True, fill nan values.

    Returns:
        pandas.Series: New feature generated.
    """
    return IchimokuIndicator(
        high=high,
        low=low,
        window1=window1,
        window2=window2,
        window3=52,
        visual=visual,
        fillna=fillna,
    ).ichimoku_a()


def ichimoku_b(high, low, window2=26, window3=52, visual=False, fillna=False):
    """Ichimoku Kinkō Hyō (Ichimoku)

    It identifies the trend and look for potential signals within that trend.

    http://stockcharts.com/school/doku.php?id=chart_school:technical_indicators:ichimoku_cloud

    Args:
        high(pandas.Series): dataset 'High' column.
        low(pandas.Series): dataset 'Low' column.
        window2(int): n2 medium period.
        window3(int): n3 high period.
        visual(bool): if True, shift n2 values.
        fillna(bool): if True, fill nan values.

    Returns:
        pandas.Series: New feature generated.
    """
    return IchimokuIndicator(
        high=high,
        low=low,
        window1=9,
        window2=window2,
        window3=window3,
        visual=visual,
        fillna=fillna,
    ).ichimoku_b()


def aroon_up(close, window=25, fillna=False):
    """Aroon Indicator (AI)

    Identify when trends are likely to change direction (uptrend).

    Aroon Up - ((N - Days Since N-day High) / N) x 100

    https://www.investopedia.com/terms/a/aroon.asp

    Args:
        close(pandas.Series): dataset 'Close' column.
        window(int): n period.
        fillna(bool): if True, fill nan values.

    Returns:
        pandas.Series: New feature generated.

    """
    return AroonIndicator(close=close, window=window, fillna=fillna).aroon_up()


def aroon_down(close, window=25, fillna=False):
    """Aroon Indicator (AI)

    Identify when trends are likely to change direction (downtrend).

    Aroon Down - ((N - Days Since N-day Low) / N) x 100

    https://www.investopedia.com/terms/a/aroon.asp

    Args:
        close(pandas.Series): dataset 'Close' column.
        window(int): n period.
        fillna(bool): if True, fill nan values.

    Returns:
        pandas.Series: New feature generated.
    """
    return AroonIndicator(close=close, window=window, fillna=fillna).aroon_down()


def psar_up(high, low, close, step=0.02, max_step=0.20, fillna=False):
    """Parabolic Stop and Reverse (Parabolic SAR)

    Returns the PSAR series with non-N/A values for upward trends

    https://school.stockcharts.com/doku.php?id=technical_indicators:parabolic_sar

    Args:
        high(pandas.Series): dataset 'High' column.
        low(pandas.Series): dataset 'Low' column.
        close(pandas.Series): dataset 'Close' column.
        step(float): the Acceleration Factor used to compute the SAR.
        max_step(float): the maximum value allowed for the Acceleration Factor.
        fillna(bool): if True, fill nan values.

    Returns:
        pandas.Series: New feature generated.
    """
    indicator = PSARIndicator(
        high=high, low=low, close=close, step=step, max_step=max_step, fillna=fillna
    )
    return indicator.psar_up()


def psar_down(high, low, close, step=0.02, max_step=0.20, fillna=False):
    """Parabolic Stop and Reverse (Parabolic SAR)

    Returns the PSAR series with non-N/A values for downward trends

    https://school.stockcharts.com/doku.php?id=technical_indicators:parabolic_sar

    Args:
        high(pandas.Series): dataset 'High' column.
        low(pandas.Series): dataset 'Low' column.
        close(pandas.Series): dataset 'Close' column.
        step(float): the Acceleration Factor used to compute the SAR.
        max_step(float): the maximum value allowed for the Acceleration Factor.
        fillna(bool): if True, fill nan values.

    Returns:
        pandas.Series: New feature generated.
    """
    indicator = PSARIndicator(
        high=high, low=low, close=close, step=step, max_step=max_step, fillna=fillna
    )
    return indicator.psar_down()


def psar_up_indicator(high, low, close, step=0.02, max_step=0.20, fillna=False):
    """Parabolic Stop and Reverse (Parabolic SAR) Upward Trend Indicator

    Returns 1, if there is a reversal towards an upward trend. Else, returns 0.

    https://school.stockcharts.com/doku.php?id=technical_indicators:parabolic_sar

    Args:
        high(pandas.Series): dataset 'High' column.
        low(pandas.Series): dataset 'Low' column.
        close(pandas.Series): dataset 'Close' column.
        step(float): the Acceleration Factor used to compute the SAR.
        max_step(float): the maximum value allowed for the Acceleration Factor.
        fillna(bool): if True, fill nan values.

    Returns:
        pandas.Series: New feature generated.
    """
    indicator = PSARIndicator(
        high=high, low=low, close=close, step=step, max_step=max_step, fillna=fillna
    )
    return indicator.psar_up_indicator()


def psar_down_indicator(high, low, close, step=0.02, max_step=0.20, fillna=False):
    """Parabolic Stop and Reverse (Parabolic SAR) Downward Trend Indicator

    Returns 1, if there is a reversal towards an downward trend. Else, returns 0.

    https://school.stockcharts.com/doku.php?id=technical_indicators:parabolic_sar

    Args:
        high(pandas.Series): dataset 'High' column.
        low(pandas.Series): dataset 'Low' column.
        close(pandas.Series): dataset 'Close' column.
        step(float): the Acceleration Factor used to compute the SAR.
        max_step(float): the maximum value allowed for the Acceleration Factor.
        fillna(bool): if True, fill nan values.

    Returns:
        pandas.Series: New feature generated.
    """
    indicator = PSARIndicator(
        high=high, low=low, close=close, step=step, max_step=max_step, fillna=fillna
    )
    return indicator.psar_down_indicator()



# VOLATILITY BASED INDICATORS





class AverageTrueRange(IndicatorMixin):
    """Average True Range (ATR)

    The indicator provide an indication of the degree of price volatility.
    Strong moves, in either direction, are often accompanied by large ranges,
    or large True Ranges.

    http://stockcharts.com/school/doku.php?id=chart_school:technical_indicators:average_true_range_atr

    Args:
        high(pandas.Series): dataset 'High' column.
        low(pandas.Series): dataset 'Low' column.
        close(pandas.Series): dataset 'Close' column.
        window(int): n period.
        fillna(bool): if True, fill nan values.
    """

    def __init__(
        self,
        high: pd.Series,
        low: pd.Series,
        close: pd.Series,
        window: int = 14,
        fillna: bool = False,
    ):
        self._high = high
        self._low = low
        self._close = close
        self._window = window
        self._fillna = fillna
        self._run()

    def _run(self):
        close_shift = self._close.shift(1)
        true_range = self._true_range(self._high, self._low, close_shift)
        atr = np.zeros(len(self._close))
        atr[self._window - 1] = true_range[0 : self._window].mean()
        for i in range(self._window, len(atr)):
            atr[i] = (atr[i - 1] * (self._window - 1) + true_range.iloc[i]) / float(
                self._window
            )
        self._atr = pd.Series(data=atr, index=true_range.index)

    def average_true_range(self) -> pd.Series:
        """Average True Range (ATR)

        Returns:
            pandas.Series: New feature generated.
        """
        atr = self._check_fillna(self._atr, value=0)
        return pd.Series(atr, name="atr")


class BollingerBands(IndicatorMixin):
    """Bollinger Bands

    https://school.stockcharts.com/doku.php?id=technical_indicators:bollinger_bands

    Args:
        close(pandas.Series): dataset 'Close' column.
        window(int): n period.
        window_dev(int): n factor standard deviation
        fillna(bool): if True, fill nan values.
    """

    def __init__(
        self,
        close: pd.Series,
        window: int = 20,
        window_dev: int = 2,
        fillna: bool = False,
    ):
        self._close = close
        self._window = window
        self._window_dev = window_dev
        self._fillna = fillna
        self._run()

    def _run(self):
        min_periods = 0 if self._fillna else self._window
        self._mavg = self._close.rolling(self._window, min_periods=min_periods).mean()
        self._mstd = self._close.rolling(self._window, min_periods=min_periods).std(
            ddof=0
        )
        self._hband = self._mavg + self._window_dev * self._mstd
        self._lband = self._mavg - self._window_dev * self._mstd

    def bollinger_mavg(self) -> pd.Series:
        """Bollinger Channel Middle Band

        Returns:
            pandas.Series: New feature generated.
        """
        mavg = self._check_fillna(self._mavg, value=-1)
        return pd.Series(mavg, name="mavg")

    def bollinger_hband(self) -> pd.Series:
        """Bollinger Channel High Band

        Returns:
            pandas.Series: New feature generated.
        """
        hband = self._check_fillna(self._hband, value=-1)
        return pd.Series(hband, name="hband")

    def bollinger_lband(self) -> pd.Series:
        """Bollinger Channel Low Band

        Returns:
            pandas.Series: New feature generated.
        """
        lband = self._check_fillna(self._lband, value=-1)
        return pd.Series(lband, name="lband")

    def bollinger_wband(self) -> pd.Series:
        """Bollinger Channel Band Width

        From: https://school.stockcharts.com/doku.php?id=technical_indicators:bollinger_band_width

        Returns:
            pandas.Series: New feature generated.
        """
        wband = ((self._hband - self._lband) / self._mavg) * 100
        wband = self._check_fillna(wband, value=0)
        return pd.Series(wband, name="bbiwband")

    def bollinger_pband(self) -> pd.Series:
        """Bollinger Channel Percentage Band

        From: https://school.stockcharts.com/doku.php?id=technical_indicators:bollinger_band_perce

        Returns:
            pandas.Series: New feature generated.
        """
        pband = (self._close - self._lband) / (self._hband - self._lband)
        pband = self._check_fillna(pband, value=0)
        return pd.Series(pband, name="bbipband")

    def bollinger_hband_indicator(self) -> pd.Series:
        """Bollinger Channel Indicator Crossing High Band (binary).

        It returns 1, if close is higher than bollinger_hband. Else, it returns 0.

        Returns:
            pandas.Series: New feature generated.
        """
        hband = pd.Series(
            np.where(self._close > self._hband, 1.0, 0.0), index=self._close.index
        )
        hband = self._check_fillna(hband, value=0)
        return pd.Series(hband, index=self._close.index, name="bbihband")

    def bollinger_lband_indicator(self) -> pd.Series:
        """Bollinger Channel Indicator Crossing Low Band (binary).

        It returns 1, if close is lower than bollinger_lband. Else, it returns 0.

        Returns:
            pandas.Series: New feature generated.
        """
        lband = pd.Series(
            np.where(self._close < self._lband, 1.0, 0.0), index=self._close.index
        )
        lband = self._check_fillna(lband, value=0)
        return pd.Series(lband, name="bbilband")


class KeltnerChannel(IndicatorMixin):
    """KeltnerChannel

    Keltner Channels are a trend following indicator used to identify reversals with channel breakouts and
    channel direction. Channels can also be used to identify overbought and oversold levels when the trend
    is flat.

    https://school.stockcharts.com/doku.php?id=technical_indicators:keltner_channels

    Args:
        high(pandas.Series): dataset 'High' column.
        low(pandas.Series): dataset 'Low' column.
        close(pandas.Series): dataset 'Close' column.
        window(int): n period.
        window_atr(int): n atr period. Only valid if original_version param is False.
        fillna(bool): if True, fill nan values.
        original_version(bool): if True, use original version as the centerline (SMA of typical price)
            if False, use EMA of close as the centerline. More info:
            https://school.stockcharts.com/doku.php?id=technical_indicators:keltner_channels
    """

    def __init__(
        self,
        high: pd.Series,
        low: pd.Series,
        close: pd.Series,
        window: int = 20,
        window_atr: int = 10,
        fillna: bool = False,
        original_version: bool = True,
    ):
        self._high = high
        self._low = low
        self._close = close
        self._window = window
        self._window_atr = window_atr
        self._fillna = fillna
        self._original_version = original_version
        self._run()

    def _run(self):
        min_periods = 1 if self._fillna else self._window
        if self._original_version:
            self._tp = (
                ((self._high + self._low + self._close) / 3.0)
                .rolling(self._window, min_periods=min_periods)
                .mean()
            )
            self._tp_high = (
                (((4 * self._high) - (2 * self._low) + self._close) / 3.0)
                .rolling(self._window, min_periods=0)
                .mean()
            )
            self._tp_low = (
                (((-2 * self._high) + (4 * self._low) + self._close) / 3.0)
                .rolling(self._window, min_periods=0)
                .mean()
            )
        else:
            self._tp = self._close.ewm(
                span=self._window, min_periods=min_periods, adjust=False
            ).mean()
            atr = AverageTrueRange(
                close=self._close,
                high=self._high,
                low=self._low,
                window=self._window_atr,
                fillna=self._fillna,
            ).average_true_range()
            self._tp_high = self._tp + (2 * atr)
            self._tp_low = self._tp - (2 * atr)

    def keltner_channel_mband(self) -> pd.Series:
        """Keltner Channel Middle Band

        Returns:
            pandas.Series: New feature generated.
        """
        tp_middle = self._check_fillna(self._tp, value=-1)
        return pd.Series(tp_middle, name="mavg")

    def keltner_channel_hband(self) -> pd.Series:
        """Keltner Channel High Band

        Returns:
            pandas.Series: New feature generated.
        """
        tp_high = self._check_fillna(self._tp_high, value=-1)
        return pd.Series(tp_high, name="kc_hband")

    def keltner_channel_lband(self) -> pd.Series:
        """Keltner Channel Low Band

        Returns:
            pandas.Series: New feature generated.
        """
        tp_low = self._check_fillna(self._tp_low, value=-1)
        return pd.Series(tp_low, name="kc_lband")

    def keltner_channel_wband(self) -> pd.Series:
        """Keltner Channel Band Width

        Returns:
            pandas.Series: New feature generated.
        """
        wband = ((self._tp_high - self._tp_low) / self._tp) * 100
        wband = self._check_fillna(wband, value=0)
        return pd.Series(wband, name="bbiwband")

    def keltner_channel_pband(self) -> pd.Series:
        """Keltner Channel Percentage Band

        Returns:
            pandas.Series: New feature generated.
        """
        pband = (self._close - self._tp_low) / (self._tp_high - self._tp_low)
        pband = self._check_fillna(pband, value=0)
        return pd.Series(pband, name="bbipband")

    def keltner_channel_hband_indicator(self) -> pd.Series:
        """Keltner Channel Indicator Crossing High Band (binary)

        It returns 1, if close is higher than keltner_channel_hband. Else, it returns 0.

        Returns:
            pandas.Series: New feature generated.
        """
        hband = pd.Series(
            np.where(self._close > self._tp_high, 1.0, 0.0), index=self._close.index
        )
        hband = self._check_fillna(hband, value=0)
        return pd.Series(hband, name="dcihband")

    def keltner_channel_lband_indicator(self) -> pd.Series:
        """Keltner Channel Indicator Crossing Low Band (binary)

        It returns 1, if close is lower than keltner_channel_lband. Else, it returns 0.

        Returns:
            pandas.Series: New feature generated.
        """
        lband = pd.Series(
            np.where(self._close < self._tp_low, 1.0, 0.0), index=self._close.index
        )
        lband = self._check_fillna(lband, value=0)
        return pd.Series(lband, name="dcilband")


class DonchianChannel(IndicatorMixin):
    """Donchian Channel

    https://www.investopedia.com/terms/d/donchianchannels.asp

    Args:
        high(pandas.Series): dataset 'High' column.
        low(pandas.Series): dataset 'Low' column.
        close(pandas.Series): dataset 'Close' column.
        window(int): n period.
        fillna(bool): if True, fill nan values.
    """

    def __init__(
        self,
        high: pd.Series,
        low: pd.Series,
        close: pd.Series,
        window: int = 20,
        offset: int = 0,
        fillna: bool = False,
    ):
        self._offset = offset
        self._close = close
        self._high = high
        self._low = low
        self._window = window
        self._fillna = fillna
        self._run()

    def _run(self):
        self._min_periods = 1 if self._fillna else self._window
        self._hband = self._high.rolling(
            self._window, min_periods=self._min_periods
        ).max()
        self._lband = self._low.rolling(
            self._window, min_periods=self._min_periods
        ).min()

    def donchian_channel_hband(self) -> pd.Series:
        """Donchian Channel High Band

        Returns:
            pandas.Series: New feature generated.
        """
        hband = self._check_fillna(self._hband, value=-1)
        if self._offset != 0:
            hband = hband.shift(self._offset)
        return pd.Series(hband, name="dchband")

    def donchian_channel_lband(self) -> pd.Series:
        """Donchian Channel Low Band

        Returns:
            pandas.Series: New feature generated.
        """
        lband = self._check_fillna(self._lband, value=-1)
        if self._offset != 0:
            lband = lband.shift(self._offset)
        return pd.Series(lband, name="dclband")

    def donchian_channel_mband(self) -> pd.Series:
        """Donchian Channel Middle Band

        Returns:
            pandas.Series: New feature generated.
        """
        mband = ((self._hband - self._lband) / 2.0) + self._lband
        mband = self._check_fillna(mband, value=-1)
        if self._offset != 0:
            mband = mband.shift(self._offset)
        return pd.Series(mband, name="dcmband")

    def donchian_channel_wband(self) -> pd.Series:
        """Donchian Channel Band Width

        Returns:
            pandas.Series: New feature generated.
        """
        mavg = self._close.rolling(self._window, min_periods=self._min_periods).mean()
        wband = ((self._hband - self._lband) / mavg) * 100
        wband = self._check_fillna(wband, value=0)
        if self._offset != 0:
            wband = wband.shift(self._offset)
        return pd.Series(wband, name="dcwband")

    def donchian_channel_pband(self) -> pd.Series:
        """Donchian Channel Percentage Band

        Returns:
            pandas.Series: New feature generated.
        """
        pband = (self._close - self._lband) / (self._hband - self._lband)
        pband = self._check_fillna(pband, value=0)
        if self._offset != 0:
            pband = pband.shift(self._offset)
        return pd.Series(pband, name="dcpband")


class UlcerIndex(IndicatorMixin):
    """Ulcer Index

    https://stockcharts.com/school/doku.php?id=chart_school:technical_indicators:ulcer_index

    Args:
        close(pandas.Series): dataset 'Close' column.
        window(int): n period.
        fillna(bool): if True, fill nan values.
    """

    def __init__(self, close: pd.Series, window: int = 14, fillna: bool = False):
        self._close = close
        self._window = window
        self._fillna = fillna
        self._run()

    def _run(self):
        _ui_max = self._close.rolling(self._window, min_periods=1).max()
        _r_i = 100 * (self._close - _ui_max) / _ui_max

        def ui_function():
            def _ui_function(x):
                return np.sqrt((x ** 2 / self._window).sum())

            return _ui_function

        self._ulcer_idx = _r_i.rolling(self._window).apply(ui_function(), raw=True)

    def ulcer_index(self) -> pd.Series:
        """Ulcer Index (UI)

        Returns:
            pandas.Series: New feature generated.
        """
        ulcer_idx = self._check_fillna(self._ulcer_idx)
        return pd.Series(ulcer_idx, name="ui")


def average_true_range(high, low, close, window=14, fillna=False):
    """Average True Range (ATR)

    The indicator provide an indication of the degree of price volatility.
    Strong moves, in either direction, are often accompanied by large ranges,
    or large True Ranges.

    http://stockcharts.com/school/doku.php?id=chart_school:technical_indicators:average_true_range_atr

    Args:
        high(pandas.Series): dataset 'High' column.
        low(pandas.Series): dataset 'Low' column.
        close(pandas.Series): dataset 'Close' column.
        window(int): n period.
        fillna(bool): if True, fill nan values.

    Returns:
        pandas.Series: New feature generated.
    """
    indicator = AverageTrueRange(
        high=high, low=low, close=close, window=window, fillna=fillna
    )
    return indicator.average_true_range()


def bollinger_mavg(close, window=20, fillna=False):
    """Bollinger Bands (BB)

    N-period simple moving average (MA).

    https://en.wikipedia.org/wiki/Bollinger_Bands

    Args:
        close(pandas.Series): dataset 'Close' column.
        window(int): n period.
        fillna(bool): if True, fill nan values.

    Returns:
        pandas.Series: New feature generated.
    """
    indicator = BollingerBands(close=close, window=window, fillna=fillna)
    return indicator.bollinger_mavg()


def bollinger_hband(close, window=20, window_dev=2, fillna=False):
    """Bollinger Bands (BB)

    Upper band at K times an N-period standard deviation above the moving
    average (MA + Kdeviation).

    https://en.wikipedia.org/wiki/Bollinger_Bands

    Args:
        close(pandas.Series): dataset 'Close' column.
        window(int): n period.
        window_dev(int): n factor standard deviation
        fillna(bool): if True, fill nan values.

    Returns:
        pandas.Series: New feature generated.
    """
    indicator = BollingerBands(
        close=close, window=window, window_dev=window_dev, fillna=fillna
    )
    return indicator.bollinger_hband()


def bollinger_lband(close, window=20, window_dev=2, fillna=False):
    """Bollinger Bands (BB)

    Lower band at K times an N-period standard deviation below the moving
    average (MA − Kdeviation).

    https://en.wikipedia.org/wiki/Bollinger_Bands

    Args:
        close(pandas.Series): dataset 'Close' column.
        window(int): n period.
        window_dev(int): n factor standard deviation
        fillna(bool): if True, fill nan values.

    Returns:
        pandas.Series: New feature generated.
    """
    indicator = BollingerBands(
        close=close, window=window, window_dev=window_dev, fillna=fillna
    )
    return indicator.bollinger_lband()


def bollinger_wband(close, window=20, window_dev=2, fillna=False):
    """Bollinger Channel Band Width

    From: https://school.stockcharts.com/doku.php?id=technical_indicators:bollinger_band_width

    Args:
        close(pandas.Series): dataset 'Close' column.
        window(int): n period.
        window_dev(int): n factor standard deviation
        fillna(bool): if True, fill nan values.

    Returns:
        pandas.Series: New feature generated.
    """
    indicator = BollingerBands(
        close=close, window=window, window_dev=window_dev, fillna=fillna
    )
    return indicator.bollinger_wband()


def bollinger_pband(close, window=20, window_dev=2, fillna=False):
    """Bollinger Channel Percentage Band

    From: https://school.stockcharts.com/doku.php?id=technical_indicators:bollinger_band_perce

    Args:
        close(pandas.Series): dataset 'Close' column.
        window(int): n period.
        window_dev(int): n factor standard deviation
        fillna(bool): if True, fill nan values.

    Returns:
        pandas.Series: New feature generated.
    """
    indicator = BollingerBands(
        close=close, window=window, window_dev=window_dev, fillna=fillna
    )
    return indicator.bollinger_pband()


def bollinger_hband_indicator(close, window=20, window_dev=2, fillna=False):
    """Bollinger High Band Indicator

    Returns 1, if close is higher than bollinger high band. Else, return 0.

    https://en.wikipedia.org/wiki/Bollinger_Bands

    Args:
        close(pandas.Series): dataset 'Close' column.
        window(int): n period.
        window_dev(int): n factor standard deviation
        fillna(bool): if True, fill nan values.

    Returns:
        pandas.Series: New feature generated.
    """
    indicator = BollingerBands(
        close=close, window=window, window_dev=window_dev, fillna=fillna
    )
    return indicator.bollinger_hband_indicator()


def bollinger_lband_indicator(close, window=20, window_dev=2, fillna=False):
    """Bollinger Low Band Indicator

    Returns 1, if close is lower than bollinger low band. Else, return 0.

    https://en.wikipedia.org/wiki/Bollinger_Bands

    Args:
        close(pandas.Series): dataset 'Close' column.
        window(int): n period.
        window_dev(int): n factor standard deviation
        fillna(bool): if True, fill nan values.

    Returns:
        pandas.Series: New feature generated.
    """
    indicator = BollingerBands(
        close=close, window=window, window_dev=window_dev, fillna=fillna
    )
    return indicator.bollinger_lband_indicator()


def keltner_channel_mband(
    high, low, close, window=20, window_atr=10, fillna=False, original_version=True
):
    """Keltner channel (KC)

    Showing a simple moving average line (central) of typical price.

    https://school.stockcharts.com/doku.php?id=technical_indicators:keltner_channels

    Args:
        high(pandas.Series): dataset 'High' column.
        low(pandas.Series): dataset 'Low' column.
        close(pandas.Series): dataset 'Close' column.
        window(int): n period.
        window_atr(int): n atr period. Only valid if original_version param is False.
        fillna(bool): if True, fill nan values.
        original_version(bool): if True, use original version as the centerline (SMA of typical price)
            if False, use EMA of close as the centerline. More info:
            https://school.stockcharts.com/doku.php?id=technical_indicators:keltner_channels

    Returns:
        pandas.Series: New feature generated.
    """
    indicator = KeltnerChannel(
        high=high,
        low=low,
        close=close,
        window=window,
        window_atr=window_atr,
        fillna=fillna,
        original_version=original_version,
    )
    return indicator.keltner_channel_mband()


def keltner_channel_hband(
    high, low, close, window=20, window_atr=10, fillna=False, original_version=True
):
    """Keltner channel (KC)

    Showing a simple moving average line (high) of typical price.

    https://school.stockcharts.com/doku.php?id=technical_indicators:keltner_channels

    Args:
        high(pandas.Series): dataset 'High' column.
        low(pandas.Series): dataset 'Low' column.
        close(pandas.Series): dataset 'Close' column.
        window(int): n period.
        window_atr(int): n atr period. Only valid if original_version param is False.
        fillna(bool): if True, fill nan values.
        original_version(bool): if True, use original version as the centerline (SMA of typical price)
            if False, use EMA of close as the centerline. More info:
            https://school.stockcharts.com/doku.php?id=technical_indicators:keltner_channels

    Returns:
        pandas.Series: New feature generated.
    """
    indicator = KeltnerChannel(
        high=high,
        low=low,
        close=close,
        window=window,
        window_atr=window_atr,
        fillna=fillna,
        original_version=original_version,
    )
    return indicator.keltner_channel_hband()


def keltner_channel_lband(
    high, low, close, window=20, window_atr=10, fillna=False, original_version=True
):
    """Keltner channel (KC)

    Showing a simple moving average line (low) of typical price.

    https://school.stockcharts.com/doku.php?id=technical_indicators:keltner_channels

    Args:
        high(pandas.Series): dataset 'High' column.
        low(pandas.Series): dataset 'Low' column.
        close(pandas.Series): dataset 'Close' column.
        window(int): n period.
        window_atr(int): n atr period. Only valid if original_version param is False.
        fillna(bool): if True, fill nan values.
        original_version(bool): if True, use original version as the centerline (SMA of typical price)
            if False, use EMA of close as the centerline. More info:
            https://school.stockcharts.com/doku.php?id=technical_indicators:keltner_channels

    Returns:
        pandas.Series: New feature generated.
    """
    indicator = KeltnerChannel(
        high=high,
        low=low,
        close=close,
        window=window,
        window_atr=window_atr,
        fillna=fillna,
        original_version=original_version,
    )
    return indicator.keltner_channel_lband()


def keltner_channel_wband(
    high, low, close, window=20, window_atr=10, fillna=False, original_version=True
):
    """Keltner Channel Band Width

    https://school.stockcharts.com/doku.php?id=technical_indicators:keltner_channels

    Args:
        high(pandas.Series): dataset 'High' column.
        low(pandas.Series): dataset 'Low' column.
        close(pandas.Series): dataset 'Close' column.
        window(int): n period.
        window_atr(int): n atr period. Only valid if original_version param is False.
        fillna(bool): if True, fill nan values.
        original_version(bool): if True, use original version as the centerline (SMA of typical price)
            if False, use EMA of close as the centerline. More info:
            https://school.stockcharts.com/doku.php?id=technical_indicators:keltner_channels

    Returns:
        pandas.Series: New feature generated.
    """
    indicator = KeltnerChannel(
        high=high,
        low=low,
        close=close,
        window=window,
        window_atr=window_atr,
        fillna=fillna,
        original_version=original_version,
    )
    return indicator.keltner_channel_wband()


def keltner_channel_pband(
    high, low, close, window=20, window_atr=10, fillna=False, original_version=True
):
    """Keltner Channel Percentage Band

    https://school.stockcharts.com/doku.php?id=technical_indicators:keltner_channels

    Args:
        high(pandas.Series): dataset 'High' column.
        low(pandas.Series): dataset 'Low' column.
        close(pandas.Series): dataset 'Close' column.
        window(int): n period.
        window_atr(int): n atr period. Only valid if original_version param is False.
        fillna(bool): if True, fill nan values.
        original_version(bool): if True, use original version as the centerline (SMA of typical price)
            if False, use EMA of close as the centerline. More info:
            https://school.stockcharts.com/doku.php?id=technical_indicators:keltner_channels

    Returns:
        pandas.Series: New feature generated.
    """
    indicator = KeltnerChannel(
        high=high,
        low=low,
        close=close,
        window=window,
        window_atr=window_atr,
        fillna=fillna,
        original_version=original_version,
    )
    return indicator.keltner_channel_pband()


def keltner_channel_hband_indicator(
    high, low, close, window=20, window_atr=10, fillna=False, original_version=True
):
    """Keltner Channel High Band Indicator (KC)

    Returns 1, if close is higher than keltner high band channel. Else,
    return 0.

    https://school.stockcharts.com/doku.php?id=technical_indicators:keltner_channels

    Args:
        high(pandas.Series): dataset 'High' column.
        low(pandas.Series): dataset 'Low' column.
        close(pandas.Series): dataset 'Close' column.
        window(int): n period.
        window_atr(int): n atr period. Only valid if original_version param is False.
        fillna(bool): if True, fill nan values.
        original_version(bool): if True, use original version as the centerline (SMA of typical price)
            if False, use EMA of close as the centerline. More info:
            https://school.stockcharts.com/doku.php?id=technical_indicators:keltner_channels

    Returns:
        pandas.Series: New feature generated.
    """
    indicator = KeltnerChannel(
        high=high,
        low=low,
        close=close,
        window=window,
        window_atr=window_atr,
        fillna=fillna,
        original_version=original_version,
    )
    return indicator.keltner_channel_hband_indicator()


def keltner_channel_lband_indicator(
    high, low, close, window=20, window_atr=10, fillna=False, original_version=True
):
    """Keltner Channel Low Band Indicator (KC)

    Returns 1, if close is lower than keltner low band channel. Else, return 0.

    https://school.stockcharts.com/doku.php?id=technical_indicators:keltner_channels

    Args:
        high(pandas.Series): dataset 'High' column.
        low(pandas.Series): dataset 'Low' column.
        close(pandas.Series): dataset 'Close' column.
        window(int): n period.
        window_atr(int): n atr period. Only valid if original_version param is False.
        fillna(bool): if True, fill nan values.
        original_version(bool): if True, use original version as the centerline (SMA of typical price)
            if False, use EMA of close as the centerline. More info:
            https://school.stockcharts.com/doku.php?id=technical_indicators:keltner_channels

    Returns:
        pandas.Series: New feature generated.
    """
    indicator = KeltnerChannel(
        high=high,
        low=low,
        close=close,
        window=window,
        window_atr=window_atr,
        fillna=fillna,
        original_version=original_version,
    )
    return indicator.keltner_channel_lband_indicator()


def donchian_channel_hband(high, low, close, window=20, offset=0, fillna=False):
    """Donchian Channel High Band (DC)

    The upper band marks the highest price of an issue for n periods.

    https://www.investopedia.com/terms/d/donchianchannels.asp

    Args:
        high(pandas.Series): dataset 'High' column.
        low(pandas.Series): dataset 'Low' column.
        close(pandas.Series): dataset 'Close' column.
        window(int): n period.
        fillna(bool): if True, fill nan values.

    Returns:
        pandas.Series: New feature generated.
    """
    indicator = DonchianChannel(
        high=high, low=low, close=close, window=window, offset=offset, fillna=fillna
    )
    return indicator.donchian_channel_hband()


def donchian_channel_lband(high, low, close, window=20, offset=0, fillna=False):
    """Donchian Channel Low Band (DC)

    The lower band marks the lowest price for n periods.

    https://www.investopedia.com/terms/d/donchianchannels.asp

    Args:
        high(pandas.Series): dataset 'High' column.
        low(pandas.Series): dataset 'Low' column.
        close(pandas.Series): dataset 'Close' column.
        window(int): n period.
        fillna(bool): if True, fill nan values.

    Returns:
        pandas.Series: New feature generated.
    """
    indicator = DonchianChannel(
        high=high, low=low, close=close, window=window, offset=offset, fillna=fillna
    )
    return indicator.donchian_channel_lband()


def donchian_channel_mband(high, low, close, window=10, offset=0, fillna=False):
    """Donchian Channel Middle Band (DC)

    https://www.investopedia.com/terms/d/donchianchannels.asp

    Args:
        high(pandas.Series): dataset 'High' column.
        low(pandas.Series): dataset 'Low' column.
        close(pandas.Series): dataset 'Close' column.
        window(int): n period.
        fillna(bool): if True, fill nan values.

    Returns:
        pandas.Series: New feature generated.
    """
    indicator = DonchianChannel(
        high=high, low=low, close=close, window=window, offset=offset, fillna=fillna
    )
    return indicator.donchian_channel_mband()


def donchian_channel_wband(high, low, close, window=10, offset=0, fillna=False):
    """Donchian Channel Band Width (DC)

    https://www.investopedia.com/terms/d/donchianchannels.asp

    Args:
        high(pandas.Series): dataset 'High' column.
        low(pandas.Series): dataset 'Low' column.
        close(pandas.Series): dataset 'Close' column.
        window(int): n period.
        fillna(bool): if True, fill nan values.

    Returns:
        pandas.Series: New feature generated.
    """
    indicator = DonchianChannel(
        high=high, low=low, close=close, window=window, offset=offset, fillna=fillna
    )
    return indicator.donchian_channel_wband()


def donchian_channel_pband(high, low, close, window=10, offset=0, fillna=False):
    """Donchian Channel Percentage Band (DC)

    https://www.investopedia.com/terms/d/donchianchannels.asp

    Args:
        high(pandas.Series): dataset 'High' column.
        low(pandas.Series): dataset 'Low' column.
        close(pandas.Series): dataset 'Close' column.
        window(int): n period.
        fillna(bool): if True, fill nan values.

    Returns:
        pandas.Series: New feature generated.
    """
    indicator = DonchianChannel(
        high=high, low=low, close=close, window=window, offset=offset, fillna=fillna
    )
    return indicator.donchian_channel_pband()


def ulcer_index(close, window=14, fillna=False):
    """Ulcer Index

    https://stockcharts.com/school/doku.php?id=chart_school:technical_indicators:ulcer_index

    Args:
        close(pandas.Series): dataset 'Close' column.
        window(int): n period.
        fillna(bool): if True, fill nan values.

    Returns:
            pandas.Series: New feature generated.
    """
    indicator = UlcerIndex(close=close, window=window, fillna=fillna)
    return indicator.ulcer_index()


# vOLUME INDICATORS






class AccDistIndexIndicator(IndicatorMixin):
    """Accumulation/Distribution Index (ADI)

    Acting as leading indicator of price movements.

    https://school.stockcharts.com/doku.php?id=technical_indicators:accumulation_distribution_line

    Args:
        high(pandas.Series): dataset 'High' column.
        low(pandas.Series): dataset 'Low' column.
        close(pandas.Series): dataset 'Close' column.
        volume(pandas.Series): dataset 'Volume' column.
        fillna(bool): if True, fill nan values.
    """

    def __init__(
        self,
        high: pd.Series,
        low: pd.Series,
        close: pd.Series,
        volume: pd.Series,
        fillna: bool = False,
    ):
        self._high = high
        self._low = low
        self._close = close
        self._volume = volume
        self._fillna = fillna
        self._run()

    def _run(self):
        clv = ((self._close - self._low) - (self._high - self._close)) / (
            self._high - self._low
        )
        clv = clv.fillna(0.0)  # float division by zero
        adi = clv * self._volume
        self._adi = adi.cumsum()

    def acc_dist_index(self) -> pd.Series:
        """Accumulation/Distribution Index (ADI)

        Returns:
            pandas.Series: New feature generated.
        """
        adi = self._check_fillna(self._adi, value=0)
        return pd.Series(adi, name="adi")


class OnBalanceVolumeIndicator(IndicatorMixin):
    """On-balance volume (OBV)

    It relates price and volume in the stock market. OBV is based on a
    cumulative total volume.

    https://en.wikipedia.org/wiki/On-balance_volume

    Args:
        close(pandas.Series): dataset 'Close' column.
        volume(pandas.Series): dataset 'Volume' column.
        fillna(bool): if True, fill nan values.
    """

    def __init__(self, close: pd.Series, volume: pd.Series, fillna: bool = False):
        self._close = close
        self._volume = volume
        self._fillna = fillna
        self._run()

    def _run(self):
        obv = np.where(self._close < self._close.shift(1), -self._volume, self._volume)
        self._obv = pd.Series(obv, index=self._close.index).cumsum()

    def on_balance_volume(self) -> pd.Series:
        """On-balance volume (OBV)

        Returns:
            pandas.Series: New feature generated.
        """
        obv = self._check_fillna(self._obv, value=0)
        return pd.Series(obv, name="obv")


class ChaikinMoneyFlowIndicator(IndicatorMixin):
    """Chaikin Money Flow (CMF)

    It measures the amount of Money Flow Volume over a specific period.

    http://stockcharts.com/school/doku.php?id=chart_school:technical_indicators:chaikin_money_flow_cmf

    Args:
        high(pandas.Series): dataset 'High' column.
        low(pandas.Series): dataset 'Low' column.
        close(pandas.Series): dataset 'Close' column.
        volume(pandas.Series): dataset 'Volume' column.
        window(int): n period.
        fillna(bool): if True, fill nan values.
    """

    def __init__(
        self,
        high: pd.Series,
        low: pd.Series,
        close: pd.Series,
        volume: pd.Series,
        window: int = 20,
        fillna: bool = False,
    ):
        self._high = high
        self._low = low
        self._close = close
        self._volume = volume
        self._window = window
        self._fillna = fillna
        self._run()

    def _run(self):
        mfv = ((self._close - self._low) - (self._high - self._close)) / (
            self._high - self._low
        )
        mfv = mfv.fillna(0.0)  # float division by zero
        mfv *= self._volume
        min_periods = 0 if self._fillna else self._window
        self._cmf = (
            mfv.rolling(self._window, min_periods=min_periods).sum()
            / self._volume.rolling(self._window, min_periods=min_periods).sum()
        )

    def chaikin_money_flow(self) -> pd.Series:
        """Chaikin Money Flow (CMF)

        Returns:
            pandas.Series: New feature generated.
        """
        cmf = self._check_fillna(self._cmf, value=0)
        return pd.Series(cmf, name="cmf")


class ForceIndexIndicator(IndicatorMixin):
    """Force Index (FI)

    It illustrates how strong the actual buying or selling pressure is. High
    positive values mean there is a strong rising trend, and low values signify
    a strong downward trend.

    http://stockcharts.com/school/doku.php?id=chart_school:technical_indicators:force_index

    Args:
        close(pandas.Series): dataset 'Close' column.
        volume(pandas.Series): dataset 'Volume' column.
        window(int): n period.
        fillna(bool): if True, fill nan values.
    """

    def __init__(
        self,
        close: pd.Series,
        volume: pd.Series,
        window: int = 13,
        fillna: bool = False,
    ):
        self._close = close
        self._volume = volume
        self._window = window
        self._fillna = fillna
        self._run()

    def _run(self):
        fi_series = (self._close - self._close.shift(1)) * self._volume
        self._fi = _ema(fi_series, self._window, fillna=self._fillna)

    def force_index(self) -> pd.Series:
        """Force Index (FI)

        Returns:
            pandas.Series: New feature generated.
        """
        fi_series = self._check_fillna(self._fi, value=0)
        return pd.Series(fi_series, name=f"fi_{self._window}")


class EaseOfMovementIndicator(IndicatorMixin):
    """Ease of movement (EoM, EMV)

    It relate an asset's price change to its volume and is particularly useful
    for assessing the strength of a trend.

    https://en.wikipedia.org/wiki/Ease_of_movement

    Args:
        high(pandas.Series): dataset 'High' column.
        low(pandas.Series): dataset 'Low' column.
        volume(pandas.Series): dataset 'Volume' column.
        window(int): n period.
        fillna(bool): if True, fill nan values.
    """

    def __init__(
        self,
        high: pd.Series,
        low: pd.Series,
        volume: pd.Series,
        window: int = 14,
        fillna: bool = False,
    ):
        self._high = high
        self._low = low
        self._volume = volume
        self._window = window
        self._fillna = fillna
        self._run()

    def _run(self):
        self._emv = (
            (self._high.diff(1) + self._low.diff(1))
            * (self._high - self._low)
            / (2 * self._volume)
        )
        self._emv *= 100000000

    def ease_of_movement(self) -> pd.Series:
        """Ease of movement (EoM, EMV)

        Returns:
            pandas.Series: New feature generated.
        """
        emv = self._check_fillna(self._emv, value=0)
        return pd.Series(emv, name=f"eom_{self._window}")

    def sma_ease_of_movement(self) -> pd.Series:
        """Signal Ease of movement (EoM, EMV)

        Returns:
            pandas.Series: New feature generated.
        """
        min_periods = 0 if self._fillna else self._window
        emv = self._emv.rolling(self._window, min_periods=min_periods).mean()
        emv = self._check_fillna(emv, value=0)
        return pd.Series(emv, name=f"sma_eom_{self._window}")


class VolumePriceTrendIndicator(IndicatorMixin):
    """Volume-price trend (VPT)

    Is based on a running cumulative volume that adds or substracts a multiple
    of the percentage change in share price trend and current volume, depending
    upon the investment's upward or downward movements.

    https://en.wikipedia.org/wiki/Volume%E2%80%93price_trend

    Args:
        close(pandas.Series): dataset 'Close' column.
        volume(pandas.Series): dataset 'Volume' column.
        fillna(bool): if True, fill nan values.
    """

    def __init__(self, close: pd.Series, volume: pd.Series, fillna: bool = False):
        self._close = close
        self._volume = volume
        self._fillna = fillna
        self._run()

    def _run(self):
        vpt = self._volume * (
            (self._close - self._close.shift(1, fill_value=self._close.mean()))
            / self._close.shift(1, fill_value=self._close.mean())
        )
        self._vpt = vpt.shift(1, fill_value=vpt.mean()) + vpt

    def volume_price_trend(self) -> pd.Series:
        """Volume-price trend (VPT)

        Returns:
            pandas.Series: New feature generated.
        """
        vpt = self._check_fillna(self._vpt, value=0)
        return pd.Series(vpt, name="vpt")


class NegativeVolumeIndexIndicator(IndicatorMixin):
    """Negative Volume Index (NVI)

    http://stockcharts.com/school/doku.php?id=chart_school:technical_indicators:negative_volume_inde

    Args:
        close(pandas.Series): dataset 'Close' column.
        volume(pandas.Series): dataset 'Volume' column.
        fillna(bool): if True, fill nan values with 1000.
    """

    def __init__(self, close: pd.Series, volume: pd.Series, fillna: bool = False):
        self._close = close
        self._volume = volume
        self._fillna = fillna
        self._run()

    def _run(self):
        price_change = self._close.pct_change()
        vol_decrease = self._volume.shift(1) > self._volume
        self._nvi = pd.Series(
            data=np.nan, index=self._close.index, dtype="float64", name="nvi"
        )
        self._nvi.iloc[0] = 1000
        for i in range(1, len(self._nvi)):
            if vol_decrease.iloc[i]:
                self._nvi.iloc[i] = self._nvi.iloc[i - 1] * (1.0 + price_change.iloc[i])
            else:
                self._nvi.iloc[i] = self._nvi.iloc[i - 1]

    def negative_volume_index(self) -> pd.Series:
        """Negative Volume Index (NVI)

        Returns:
            pandas.Series: New feature generated.
        """
        # IDEA: There shouldn't be any na; might be better to throw exception
        nvi = self._check_fillna(self._nvi, value=1000)
        return pd.Series(nvi, name="nvi")


class MFIIndicator(IndicatorMixin):
    """Money Flow Index (MFI)

    Uses both price and volume to measure buying and selling pressure. It is
    positive when the typical price rises (buying pressure) and negative when
    the typical price declines (selling pressure). A ratio of positive and
    negative money flow is then plugged into an RSI formula to create an
    oscillator that moves between zero and one hundred.

    http://stockcharts.com/school/doku.php?id=chart_school:technical_indicators:money_flow_index_mfi

    Args:
        high(pandas.Series): dataset 'High' column.
        low(pandas.Series): dataset 'Low' column.
        close(pandas.Series): dataset 'Close' column.
        volume(pandas.Series): dataset 'Volume' column.
        window(int): n period.
        fillna(bool): if True, fill nan values.
    """

    def __init__(
        self,
        high: pd.Series,
        low: pd.Series,
        close: pd.Series,
        volume: pd.Series,
        window: int = 14,
        fillna: bool = False,
    ):
        self._high = high
        self._low = low
        self._close = close
        self._volume = volume
        self._window = window
        self._fillna = fillna
        self._run()

    def _run(self):
        typical_price = (self._high + self._low + self._close) / 3.0
        up_down = np.where(
            typical_price > typical_price.shift(1),
            1,
            np.where(typical_price < typical_price.shift(1), -1, 0),
        )
        mfr = typical_price * self._volume * up_down

        # Positive and negative money flow with n periods
        min_periods = 0 if self._fillna else self._window
        n_positive_mf = mfr.rolling(self._window, min_periods=min_periods).apply(
            lambda x: np.sum(np.where(x >= 0.0, x, 0.0)), raw=True
        )
        n_negative_mf = abs(
            mfr.rolling(self._window, min_periods=min_periods).apply(
                lambda x: np.sum(np.where(x < 0.0, x, 0.0)), raw=True
            )
        )

        # n_positive_mf = np.where(mf.rolling(self._window).sum() >= 0.0, mf, 0.0)
        # n_negative_mf = abs(np.where(mf.rolling(self._window).sum() < 0.0, mf, 0.0))

        # Money flow index
        mfi = n_positive_mf / n_negative_mf
        self._mfi = 100 - (100 / (1 + mfi))

    def money_flow_index(self) -> pd.Series:
        """Money Flow Index (MFI)

        Returns:
            pandas.Series: New feature generated.
        """
        mfi = self._check_fillna(self._mfi, value=50)
        return pd.Series(mfi, name=f"mfi_{self._window}")


class VolumeWeightedAveragePrice(IndicatorMixin):
    """Volume Weighted Average Price (VWAP)

    VWAP equals the dollar value of all trading periods divided
    by the total trading volume for the current day.
    The calculation starts when trading opens and ends when it closes.
    Because it is good for the current trading day only,
    intraday periods and data are used in the calculation.

    https://school.stockcharts.com/doku.php?id=technical_indicators:vwap_intraday

    Args:
        high(pandas.Series): dataset 'High' column.
        low(pandas.Series): dataset 'Low' column.
        close(pandas.Series): dataset 'Close' column.
        volume(pandas.Series): dataset 'Volume' column.
        window(int): n period.
        fillna(bool): if True, fill nan values.

    Returns:
        pandas.Series: New feature generated.
    """

    def __init__(
        self,
        high: pd.Series,
        low: pd.Series,
        close: pd.Series,
        volume: pd.Series,
        window: int = 14,
        fillna: bool = False,
    ):
        self._high = high
        self._low = low
        self._close = close
        self._volume = volume
        self._window = window
        self._fillna = fillna
        self._run()

    def _run(self):
        # 1 typical price
        typical_price = (self._high + self._low + self._close) / 3.0

        # 2 typical price * volume
        typical_price_volume = typical_price * self._volume

        # 3 total price * volume
        min_periods = 0 if self._fillna else self._window
        total_pv = typical_price_volume.rolling(
            self._window, min_periods=min_periods
        ).sum()

        # 4 total volume
        total_volume = self._volume.rolling(self._window, min_periods=min_periods).sum()

        self.vwap = total_pv / total_volume

    def volume_weighted_average_price(self) -> pd.Series:
        """Volume Weighted Average Price (VWAP)

        Returns:
            pandas.Series: New feature generated.
        """
        vwap = self._check_fillna(self.vwap)
        return pd.Series(vwap, name=f"vwap_{self._window}")


def acc_dist_index(high, low, close, volume, fillna=False):
    """Accumulation/Distribution Index (ADI)

    Acting as leading indicator of price movements.

    https://en.wikipedia.org/wiki/Accumulation/distribution_index

    Args:
        high(pandas.Series): dataset 'High' column.
        low(pandas.Series): dataset 'Low' column.
        close(pandas.Series): dataset 'Close' column.
        volume(pandas.Series): dataset 'Volume' column.
        fillna(bool): if True, fill nan values.

    Returns:
        pandas.Series: New feature generated.
    """
    return AccDistIndexIndicator(
        high=high, low=low, close=close, volume=volume, fillna=fillna
    ).acc_dist_index()


def on_balance_volume(close, volume, fillna=False):
    """On-balance volume (OBV)

    It relates price and volume in the stock market. OBV is based on a
    cumulative total volume.

    https://en.wikipedia.org/wiki/On-balance_volume

    Args:
        close(pandas.Series): dataset 'Close' column.
        volume(pandas.Series): dataset 'Volume' column.
        fillna(bool): if True, fill nan values.

    Returns:
        pandas.Series: New feature generated.
    """
    return OnBalanceVolumeIndicator(
        close=close, volume=volume, fillna=fillna
    ).on_balance_volume()


def chaikin_money_flow(high, low, close, volume, window=20, fillna=False):
    """Chaikin Money Flow (CMF)

    It measures the amount of Money Flow Volume over a specific period.

    http://stockcharts.com/school/doku.php?id=chart_school:technical_indicators:chaikin_money_flow_cmf

    Args:
        high(pandas.Series): dataset 'High' column.
        low(pandas.Series): dataset 'Low' column.
        close(pandas.Series): dataset 'Close' column.
        volume(pandas.Series): dataset 'Volume' column.
        window(int): n period.
        fillna(bool): if True, fill nan values.

    Returns:
        pandas.Series: New feature generated.
    """
    return ChaikinMoneyFlowIndicator(
        high=high, low=low, close=close, volume=volume, window=window, fillna=fillna
    ).chaikin_money_flow()


def force_index(close, volume, window=13, fillna=False):
    """Force Index (FI)

    It illustrates how strong the actual buying or selling pressure is. High
    positive values mean there is a strong rising trend, and low values signify
    a strong downward trend.

    http://stockcharts.com/school/doku.php?id=chart_school:technical_indicators:force_index

    Args:
        close(pandas.Series): dataset 'Close' column.
        volume(pandas.Series): dataset 'Volume' column.
        window(int): n period.
        fillna(bool): if True, fill nan values.

    Returns:
        pandas.Series: New feature generated.
    """
    return ForceIndexIndicator(
        close=close, volume=volume, window=window, fillna=fillna
    ).force_index()


def ease_of_movement(high, low, volume, window=14, fillna=False):
    """Ease of movement (EoM, EMV)

    It relate an asset's price change to its volume and is particularly useful
    for assessing the strength of a trend.

    https://en.wikipedia.org/wiki/Ease_of_movement

    Args:
        high(pandas.Series): dataset 'High' column.
        low(pandas.Series): dataset 'Low' column.
        volume(pandas.Series): dataset 'Volume' column.
        window(int): n period.
        fillna(bool): if True, fill nan values.

    Returns:
        pandas.Series: New feature generated.
    """
    return EaseOfMovementIndicator(
        high=high, low=low, volume=volume, window=window, fillna=fillna
    ).ease_of_movement()


def sma_ease_of_movement(high, low, volume, window=14, fillna=False):
    """Ease of movement (EoM, EMV)

    It relate an asset's price change to its volume and is particularly useful
    for assessing the strength of a trend.

    https://en.wikipedia.org/wiki/Ease_of_movement

    Args:
        high(pandas.Series): dataset 'High' column.
        low(pandas.Series): dataset 'Low' column.
        volume(pandas.Series): dataset 'Volume' column.
        window(int): n period.
        fillna(bool): if True, fill nan values.

    Returns:
        pandas.Series: New feature generated.
    """
    return EaseOfMovementIndicator(
        high=high, low=low, volume=volume, window=window, fillna=fillna
    ).sma_ease_of_movement()


def volume_price_trend(close, volume, fillna=False):
    """Volume-price trend (VPT)

    Is based on a running cumulative volume that adds or substracts a multiple
    of the percentage change in share price trend and current volume, depending
    upon the investment's upward or downward movements.

    https://en.wikipedia.org/wiki/Volume%E2%80%93price_trend

    Args:
        close(pandas.Series): dataset 'Close' column.
        volume(pandas.Series): dataset 'Volume' column.
        fillna(bool): if True, fill nan values.

    Returns:
        pandas.Series: New feature generated.
    """
    return VolumePriceTrendIndicator(
        close=close, volume=volume, fillna=fillna
    ).volume_price_trend()


def negative_volume_index(close, volume, fillna=False):
    """Negative Volume Index (NVI)

    http://stockcharts.com/school/doku.php?id=chart_school:technical_indicators:negative_volume_inde

    The Negative Volume Index (NVI) is a cumulative indicator that uses the
    change in volume to decide when the smart money is active. Paul Dysart
    first developed this indicator in the 1930s. [...] Dysart's Negative Volume
    Index works under the assumption that the smart money is active on days
    when volume decreases and the not-so-smart money is active on days when
    volume increases.

    The cumulative NVI line was unchanged when volume increased from one
    period to the other. In other words, nothing was done. Norman Fosback, of
    Stock Market Logic, adjusted the indicator by substituting the percentage
    price change for Net Advances.

    This implementation is the Fosback version.

    If today's volume is less than yesterday's volume then:
        nvi(t) = nvi(t-1) * ( 1 + (close(t) - close(t-1)) / close(t-1) )
    Else
        nvi(t) = nvi(t-1)

    Please note: the "stockcharts.com" example calculation just adds the
    percentange change of price to previous NVI when volumes decline; other
    sources indicate that the same percentage of the previous NVI value should
    be added, which is what is implemented here.

    Args:
        close(pandas.Series): dataset 'Close' column.
        volume(pandas.Series): dataset 'Volume' column.
        fillna(bool): if True, fill nan values with 1000.

    Returns:
        pandas.Series: New feature generated.

    See also:
        https://en.wikipedia.org/wiki/Negative_volume_index
    """
    return NegativeVolumeIndexIndicator(
        close=close, volume=volume, fillna=fillna
    ).negative_volume_index()


def money_flow_index(high, low, close, volume, window=14, fillna=False):
    """Money Flow Index (MFI)

    Uses both price and volume to measure buying and selling pressure. It is
    positive when the typical price rises (buying pressure) and negative when
    the typical price declines (selling pressure). A ratio of positive and
    negative money flow is then plugged into an RSI formula to create an
    oscillator that moves between zero and one hundred.

    http://stockcharts.com/school/doku.php?id=chart_school:technical_indicators:money_flow_index_mfi

    Args:
        high(pandas.Series): dataset 'High' column.
        low(pandas.Series): dataset 'Low' column.
        close(pandas.Series): dataset 'Close' column.
        volume(pandas.Series): dataset 'Volume' column.
        window(int): n period.
        fillna(bool): if True, fill nan values.

    Returns:
        pandas.Series: New feature generated.

    """
    indicator = MFIIndicator(
        high=high, low=low, close=close, volume=volume, window=window, fillna=fillna
    )
    return indicator.money_flow_index()


def volume_weighted_average_price(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    volume: pd.Series,
    window: int = 14,
    fillna: bool = False,
):
    """Volume Weighted Average Price (VWAP)

    VWAP equals the dollar value of all trading periods divided
    by the total trading volume for the current day.
    The calculation starts when trading opens and ends when it closes.
    Because it is good for the current trading day only,
    intraday periods and data are used in the calculation.

    https://school.stockcharts.com/doku.php?id=technical_indicators:vwap_intraday

    Args:
        high(pandas.Series): dataset 'High' column.
        low(pandas.Series): dataset 'Low' column.
        close(pandas.Series): dataset 'Close' column.
        volume(pandas.Series): dataset 'Volume' column.
        window(int): n period.
        fillna(bool): if True, fill nan values.

    Returns:
        pandas.Series: New feature generated.
    """

    indicator = VolumeWeightedAveragePrice(
        high=high, low=low, close=close, volume=volume, window=window, fillna=fillna
    )
    return indicator.volume_weighted_average_price()


# Additional Indicators

class DailyReturnIndicator(IndicatorMixin):
    """Daily Return (DR)

    Args:
        close(pandas.Series): dataset 'Close' column.
        fillna(bool): if True, fill nan values.
    """

    def __init__(self, close: pd.Series, fillna: bool = False):
        self._close = close
        self._fillna = fillna
        self._run()

    def _run(self):
        self._dr = (
            self._close / self._close.shift(1, fill_value=self._close.mean())
        ) - 1
        self._dr *= 100

    def daily_return(self) -> pd.Series:
        """Daily Return (DR)

        Returns:
            pandas.Series: New feature generated.
        """
        dr_series = self._check_fillna(self._dr, value=0)
        return pd.Series(dr_series, name="d_ret")


class DailyLogReturnIndicator(IndicatorMixin):
    """Daily Log Return (DLR)

    https://stackoverflow.com/questions/31287552/logarithmic-returns-in-pandas-dataframe

    Args:
        close(pandas.Series): dataset 'Close' column.
        fillna(bool): if True, fill nan values.
    """

    def __init__(self, close: pd.Series, fillna: bool = False):
        self._close = close
        self._fillna = fillna
        self._run()

    def _run(self):
        self._dr = pd.Series(np.log(self._close)).diff()
        self._dr *= 100

    def daily_log_return(self) -> pd.Series:
        """Daily Log Return (DLR)

        Returns:
            pandas.Series: New feature generated.
        """
        dr_series = self._check_fillna(self._dr, value=0)
        return pd.Series(dr_series, name="d_logret")


class CumulativeReturnIndicator(IndicatorMixin):
    """Cumulative Return (CR)

    Args:
        close(pandas.Series): dataset 'Close' column.
        fillna(bool): if True, fill nan values.
    """

    def __init__(self, close: pd.Series, fillna: bool = False):
        self._close = close
        self._fillna = fillna
        self._run()

    def _run(self):
        self._cr = (self._close / self._close.iloc[0]) - 1
        self._cr *= 100

    def cumulative_return(self) -> pd.Series:
        """Cumulative Return (CR)

        Returns:
            pandas.Series: New feature generated.
        """
        cum_ret = self._check_fillna(self._cr, value=-1)
        return pd.Series(cum_ret, name="cum_ret")


def daily_return(close, fillna=False):
    """Daily Return (DR)

    Args:
        close(pandas.Series): dataset 'Close' column.
        fillna(bool): if True, fill nan values.

    Returns:
        pandas.Series: New feature generated.
    """
    return DailyReturnIndicator(close=close, fillna=fillna).daily_return()


def daily_log_return(close, fillna=False):
    """Daily Log Return (DLR)

    https://stackoverflow.com/questions/31287552/logarithmic-returns-in-pandas-dataframe

    Args:
        close(pandas.Series): dataset 'Close' column.
        fillna(bool): if True, fill nan values.

    Returns:
        pandas.Series: New feature generated.
    """
    return DailyLogReturnIndicator(close=close, fillna=fillna).daily_log_return()


def cumulative_return(close, fillna=False):
    """Cumulative Return (CR)

    Args:
        close(pandas.Series): dataset 'Close' column.
        fillna(bool): if True, fill nan values.

    Returns:
        pandas.Series: New feature generated.
    """
    return CumulativeReturnIndicator(close=close, fillna=fillna).cumulative_return()
