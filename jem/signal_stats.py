import numpy as np

SIGNAL_MAX_ITER = 5
SIGMA_SCALE = 0.01


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


def signal_weight(data, mu=1.0):
    """Return a weight for the signal

    w(x) = 1 - exp(-x/mu)
    """
    w = 1.0 - np.exp(-data / mu)
    return w


def noise_weight(data, mu=1.0):
    """Return a weight for the noise

    w(x) = (x>0) * exp(-x/mu)
    """
    w = (data > 0) * np.exp(-data / mu)
    return w


def signal_stats(data, niter=SIGNAL_MAX_ITER):
    """Iterative weighted estimation of signal mean and signal lower bound
    """

    d = data[data > 0]

    # Assume that everything is signal to start with
    mu_s = np.mean(d)

    # iterate (two or three iterations is enough)
    for _iter in range(niter):
        # Get the likelihood
        w = signal_weight(d, mu_s)
        # Update
        mu_s = np.sum(w * d) / np.sum(w)

    # Estimate a lower bound on the signal
    w_n = noise_weight(d, mu_s)
    mu_n = np.sum(d * w_n) / np.sum(w_n)

    return mu_s, mu_n


def global_scale(data, sigma_scale=SIGMA_SCALE, niter=SIGNAL_MAX_ITER):
    """Estimate signal and noise statistics and apply a global scale

    :param data: 2D or 3D magnitude image
    :return: scaled data (f)
             signal likelihood (w_s)
             noise level relative to signal (sigma_n)
    """

    # Estimate the signal mean and signal lower bound
    mu_s, mu_n = signal_stats(data, niter)

    # Compute a signal weight based on the signal lower bound
    w = signal_weight(data, mu_n)

    # Scale the signal by the signal mean
    f = data / mu_s

    # Set the reguralization factor to 0.01*scaled signal lower bound
    sigma = sigma_scale * mu_n / mu_s

    return f, w, sigma
