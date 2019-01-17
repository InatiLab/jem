import numpy as np

SIGNAL_MAX_ITER = 20


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


def signal_likelihood(data, noise_level):
    """Return a likelihood that data is signal
    in SNR units, sigmoid with width 1, shifted to the right by 1
    ie P(<1)=0, P(2)=0.46, P(3)=0.76, P(4)=0.91, P(5)=0.96
    """
    p = (data > noise_level) * (
        -1.0 + 2.0 / (1.0 + np.exp(-(data - noise_level) / noise_level))
    )

    return p


def noise_likelihood(data, noise_level):
    """Return a likelihood that data is not zero and noise.
    noise_likelihood(x) = (x>0) * (1 - signal_likelihood(x))
    """
    p = (data > 0) * (1.0 - signal_likelihood(data, noise_level))

    return p


def signal_weight(data, sigma=1.0):
    """Return a weight for (saturating exponential response)
    w(x) = 1 - exp(-d/sigma)
    """
    w = 1.0 - np.exp(-data / sigma)
    return w


def signal_stats(data, niter=3):
    """Estimate the statistics of the signal and noise in an image
    Iterative weighted estimation of signal mean and mad
    """

    d = data[data > 0]

    # Assume that everything is signal to start with
    mu_s, sigma_s = stats(d)
    # iterate (two or three iterations is enough)
    for _iter in range(niter):
        # Get the likelihood
        w_s = signal_weight(d, sigma_s)
        # Update
        mu_s, sigma_s = stats(d, w_s)

    w_n = 1.0 - w_s
    mu_n, sigma_n = stats(d, w_n)

    return mu_s, sigma_s, mu_n, sigma_n


def global_scale(data, niter=SIGNAL_MAX_ITER):
    """Estimate signal and noise statistics and apply a global scale

    :param data: 2D or 3D magnitude image
    :return: scaled data (f)
             signal likelihood (w_s)
             noise level relative to signal (sigma_n)
    """

    # Estimate the signal statistics
    mu_s, sigma_s, mu_n, sigma_n = signal_stats(data, niter)

    # Scale the signal by the dispersion
    # signal dispersion of f is 1.0
    f = data / sigma_s

    # Scale the noise leval
    sigma_n = sigma_n / sigma_s

    # Get a signal likelihood
    w_s = signal_likelihood(f, sigma_n)

    return f, w_s, sigma_n
