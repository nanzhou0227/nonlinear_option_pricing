import unittest
import numpy as np
from scipy.stats import norm


def blackscholes_price(K, T, S, vol, r=0, q=0, callput='call'):
    """Compute the call/put option price in the Black-Scholes model

    Parameters
    ----------
    K: scalar or array_like
        The strike of the option.
    T: scalar or array_like
        The maturity of the option, expressed in years (e.g. 0.25 for 3-month and 2 for 2 years)
    S: scalar or array_like
        The current price of the underlying asset.
    vol: scalar or array_like
        The implied Black-Scholes volatility.
    r: scalar or array_like
        The annualized risk-free interest rate, continuously compounded.
    q: scalar or array_like
        The annualized continuous dividend yield.
    callput: str
        Must be either 'call' or 'put'.

    Returns
    -------
    price: scalar or array_like
        The price of the option.

    """
    F = S * np.exp((r - q) * T)
    v = np.sqrt(vol ** 2 * T)
    d1 = np.log(F / K) / v + 0.5 * v
    d2 = d1 - v
    try:
        opttype = {'call': 1, 'put': -1}[callput.lower()]
    except:
        raise ValueError('The value of callput must be either "call" or "put".')
    price = opttype * (F * norm.cdf(opttype * d1) - K * norm.cdf(opttype * d2)) * np.exp(-r * T)
    return price


def blackscholes_implied_volatility_scalar(K, T, S, value, r=0, q=0, callput='call', tol=1e-6, maxiter=500):
    """Compute implied vol in Black-Scholes model

    Parameters
    ----------
    K: scalar
        The strike of the option.
    T: scalar
        The maturity of the option.
    S: scalar
        The current price of the underlying asset.
    value: scalar
        The value of the option
    callput: str
        Must be either 'call' or 'put'

    Returns
    -------
    vol: scalar
        The implied vol of the option.
    """
    if (K <= 0) or (T <= 0) or (S <= 0):
        return np.nan
    F = S * np.exp((r - q) * T)
    K = K / F
    value = value * np.exp(r * T) / F
    try:
        opttype = {'call': 1, 'put': -1}[callput.lower()]
    except:
        raise ValueError('The value of callput must be either "call" or "put".')
    # compute the time-value of the option
    value -= max(opttype * (1 - K), 0)
    if value < 0:
        return np.nan
    if value == 0:
        return 0
    j = 1
    p = np.log(K)
    if K >= 1:
        x0 = np.sqrt(2 * p)
        x1 = x0 - (0.5 - K * norm.cdf(-x0) - value) * np.sqrt(2 * np.pi)
        while (abs(x0 - x1) > tol * np.sqrt(T)) and (j < maxiter):
            x0 = x1
            d1 = -p / x1 + 0.5 * x1
            x1 = x1 - (norm.cdf(d1) - K * norm.cdf(d1 - x1) - value) * np.sqrt(2 * np.pi) * np.exp(0.5 * d1 ** 2)
            j += 1
        return x1 / np.sqrt(T)
    else:
        x0 = np.sqrt(-2 * p)
        x1 = x0 - (0.5 * K - norm.cdf(-x0) - value) * np.sqrt(2 * np.pi) / K
        while (abs(x0 - x1) > tol * np.sqrt(T)) and (j < maxiter):
            x0 = x1
            d1 = -p / x1 + 0.5 * x1
            x1 = x1 - (K * norm.cdf(x1 - d1) - norm.cdf(-d1) - value) * np.sqrt(2 * np.pi) * np.exp(0.5 * d1 ** 2)
            j += 1
        return x1 / np.sqrt(T)


class TestStringMethods(unittest.TestCase):

    def test_option_price(self):
        price = blackscholes_price(95, 0.25, 100, 0.2, r=0.05, callput='put')
        self.assertTrue(np.isclose(price, 1.5342604771222823, rtol=1e-6))

    def test_implied_volatility(self):
        volatility = blackscholes_impv(K=90, T=0.5, S=100,r=0.05, q=0.02, value=1.4448488506445187, callput='put')
        self.assertTrue(np.isclose(volatility, 0.2, rtol=1e-6))


# vectorized version
blackscholes_impv = np.vectorize(blackscholes_implied_volatility_scalar, excluded={'callput', 'tol', 'maxiter'})

if __name__ == "__main__":
    unittest.main()