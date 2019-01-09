import numpy as np


def stats(data, w=None):
    """Weighted mean and mean and absolute deviation

    mean = sum(w*data)/sum(w)
    sigma = sum(w*|data-m|)/sum(w)
    """
    if w is not None:
        mu = np.sum(w * data) / np.sum(w)
        sigma = np.sum(w * np.abs(data - mu)) / np.sum(w)
    else:
        mu = np.mean(data)
        sigma = np.mean(np.abs(data - mu))

    return mu, sigma


def signal_likelihood(data, sigma=1.0):
    """Return a likelihood that positive valued data is signal

    w(x) = 1 - exp(-d/sigma)
    """
    w = 1 - np.exp(-data / sigma)
    return w


def noise_likelihood(data, sigma=1.0):
    """Return a likelihood that positive valued data is noise

    w(x) = exp(-d/sigma)
    """
    w = np.exp(-data / sigma)
    return w


def signal_stats(data, niter=3):
    """Estimate the statistics of the signal in an image

    Iterative estimation of signal mean and mad and
    signal likelihood
    """

    # Assume that everything is signal to start with
    mu, sigma = stats(data)
    # iterate (two or three iterations is enough)
    for _iter in range(niter):
        # Get the likelihood
        ws = signal_likelihood(data, sigma)
        # Update
        mu, sigma = stats(data, ws)

    return mu, sigma


def noise_level(data, sigma_s=1.0):
    """Estimate the noise level"""
    # Look at all the non-zero points with noise likelihood greater than 0.5
    w_n = noise_likelihood(data, sigma_s)
    noise = data[np.logical_and(data > 0, w_n > 0.5)]
    mu_n, sigma_n = signal_stats(noise)
    return sigma_n


def global_scale(data):
    """Estimate signal and noise statistics and apply a global scale

    :param data: 2D or 3D magnitude image
    :return: scaled data f
             signal likelihood w_s
             noise level sigma_n
    """

    # Estimate the signal mean and dispersion
    mu_s, sigma_s = signal_stats(data)

    # Scale the signal by the dispersion
    # signal dispersion of f is 1.0
    f = data / sigma_s

    # Get the signal likelihood
    w_s = signal_likelihood(f)

    # Get the noise level and convert to signal likelihood
    # note that for any reasonable SNR, the conversion
    # from noise_mad is essentially moot because
    # 1 - exp(-x) \approx x for x<<1
    noise_mad = noise_level(f)
    sigma_n = signal_likelihood(noise_mad)

    return f, w_s, sigma_n
