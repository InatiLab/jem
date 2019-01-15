import numpy as np

from .logging import logger

NOISE_MAX_ITER = 20
NOISE_TOL = 1e-2


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


def noise_stats(data, niter=NOISE_MAX_ITER, tol=NOISE_TOL):
    """Estimate the statistics of the noise in an image
    Returns the lower fence, median, and upper fence of the trimmed data.
    The upper fence is a lower bound on a (global) estimate of the noise.
    The algorithm alternates between trimming the data and estimating the quartiles.
    At each step, the data are trimmed by selecting the voxels with intensity greater
    than zero and less than the upper fence from the previous step.
    Iteration stops when the relative change in the UF is less than the specified tolerance
    or when the maximum number of iterations is reached.
    """

    d = data[data > 0].flatten()

    # find the quartiles of the non-zero data
    q1, q2, q3 = np.percentile(d, [25, 50, 75])
    logger.debug("Quartiles of the original data: {}, {}, {}".format(q1, q2, q3))

    # find the quartiles of the non-zero data that is less than a cutoff
    # start with the first quartile and then iterate using the upper fence
    uf = q1

    # repeat
    for it in range(niter):
        q1, q2, q3 = np.percentile(d[d <= uf], [25, 50, 75])
        logger.debug(
            "Iteration {}. Quartiles of the trimmed data: {}, {}, {}".format(
                it, q1, q2, q3
            )
        )
        q13 = q3 - q1
        ufk = q3 + 1.5 * q13
        # check for convergence
        if abs(ufk - uf) / uf < tol or ufk < tol:
            break
        else:
            uf = ufk
    else:
        logger.warning("Warning, number of iterations exceeded")

    # recompute the quartiles
    q1, q2, q3 = np.percentile(d[d <= uf], [25, 50, 75])
    q13 = q3 - q1
    # q1, q2, q3 describes the noise
    # anything above this is a noise outlier above (possibly signal)
    uf = q3 + 1.5 * q13
    # anything below lf is a signal outlier below (not useful)
    lf = q1 - 1.5 * q13
    # but remember that the noise distribution is NOT symmetric, so uf is an underestimate

    return lf, q2, uf


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
    """Return a likelihood that data is noise.
    noise_likelihood(x) = 1 - signal_likelihood(x)
    """
    p = 1.0 - signal_likelihood(data, noise_level)

    return p


def signal_stats(data, niter=NOISE_MAX_ITER, tol=NOISE_TOL):
    """Estimate the statistics of the signal in an image
    """

    # Estimate the noise level
    _, _, uf = noise_stats(data, niter, tol)

    # Compute the signal likelihood
    w = signal_likelihood(data, uf)

    # Computed the weighted mean and mad
    mu, sigma = stats(data, w)

    return uf, mu, sigma, w


def global_scale(data, niter=NOISE_MAX_ITER, tol=NOISE_TOL):
    """Estimate signal and noise statistics and apply a global scale

    :param data: 2D or 3D magnitude image
    :return: scaled data (f)
             signal likelihood (w_s)
             noise level relative to signal (sigma_n)
    """

    # Estimate the signal properties
    uf, mu_s, sigma_s, w_s = signal_stats(data, niter, tol)

    # Scale the signal by the dispersion
    # signal dispersion of f is 1.0
    f = data / sigma_s

    # Scale the noise leval
    sigma_n = uf / sigma_s

    return f, w_s, sigma_n
