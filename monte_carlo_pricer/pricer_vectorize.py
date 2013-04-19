import numpy as np
import math
from numbapro import vectorize

@vectorize(['f8(f8, f8, f8, f8, f8)'], target='cpu')
def step(last, dt, c0, c1, noise):
    return last * math.exp(c0 * dt + c1 * noise)

def monte_carlo_pricer(paths, dt, interest, volatility):
    c0 = interest - 0.5 * volatility ** 2
    c1 = volatility * np.sqrt(dt)

    for j in xrange(1, paths.shape[1]):
        prices = paths[:, j - 1]
        noises = np.random.normal(0., 1., prices.size)
        step(prices, dt, c0, c1, noises, out=paths[:, j])


if __name__ == '__main__':
    from driver import driver
    driver(monte_carlo_pricer)
