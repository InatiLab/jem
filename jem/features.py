import numpy as np

from .filters import (
    _pinv,
    high_pass,
    low_pass,
    band_pass,
    gradient_band_pass,
    gradient_amplitude,
    hessian_band_pass,
    hessian_amplitude,
    hessian_rot,
    weighted_high_pass,
    weighted_low_pass,
    weighted_band_pass,
    weighted_gradient_band_pass,
    weighted_hessian_band_pass,
)
from .signal_stats import global_scale

# Number of spatial scales
NUM_SCALES = 3

# Scales for normalization / gain control
NORMALIZATION_SCALE = 5


def _normalize_bpd_filters(f, w, sigma, normalization_scale=NORMALIZATION_SCALE):
    """In place weighted local normalization of a set of band pass derivative filters
    """
    for scale in range(len(f)):
        amp = np.abs(f[scale]["bandpass"][0])
        gain = local_gain_field(amp, w, sigma, normalization_scale)
        f[scale]["bandpass"] = [x * gain for x in f[scale]["bandpass"]]

        amp = gradient_amplitude(f[scale]["gradient"])
        gain = local_gain_field(amp, w, sigma, normalization_scale)
        f[scale]["gradient"] = [x * gain for x in f[scale]["gradient"]]

        amp = hessian_amplitude(f[scale]["hessian"])
        gain = local_gain_field(amp, w, sigma, normalization_scale)
        f[scale]["hessian"] = [x * gain for x in f[scale]["hessian"]]


def rectify(data, polarity=1):
    """
    Rectify the data, polarity is either 1 or -1
    """
    if polarity not in [1, -1]:
        raise RuntimeError("Rectification polarity is either 1 or -1.")

    output = polarity * data * ((polarity * data) > 0)

    return output


def local_gain_field(f, w=None, sigma=0.001, scale=NORMALIZATION_SCALE):
    """
    Local gain field
    Inverse of the local gain, i.e. multiply by this for local divisive normalization
    """
    if w is not None:
        gain = _pinv(weighted_low_pass(f, w, sigma, scale), sigma)
    else:
        gain = _pinv(low_pass(f, scale), sigma)

    return gain


def local_scale_normalization(d, w, sigma=0.001, scale=NORMALIZATION_SCALE):
    """
    Local normalization by the signal mean.
    """
    f = d * local_gain_field(d, w, sigma, scale)

    return f


def local_contrast_normalization(d, w=None, sigma=0.001, scale=NORMALIZATION_SCALE):
    """
    Local contrast normalization
    Subtract the local mean and scale by the local mean absolute deviation
    """
    if w is not None:
        dev = weighted_high_pass(d, w, sigma, scale)
    else:
        dev = high_pass(d, scale)

    lc = dev * local_gain_field(np.abs(dev), w, sigma, scale)

    return lc


def laplacian_pyramid(d, w=None, sigma=0.001, num_scales=NUM_SCALES):
    """Laplacian pyramid from gaussian filters with signal likelihood weighting

    :param d: 2D or 3D numpy array
    :param w: signal likelihood
    :param sigma: noise level
    :param nscales: int number of scales
    :return: list, one dictionary per level

    The pyramid is organized by scale,
    high pass, level 0, level 1, .., Level num_scales-1, low pass
    Adding the output back together produces the original input data
    """

    # Initialize the pyramid
    lpyr = []

    # For each scale blur at this scale and subtract from the previous scale
    # initialize with the original data
    previous = d
    # Start with the scale 0 and go to num_scales
    # HP=D-G(0), L0=G(0)-G(1), L1=G(1)-G(2), ..., LN=G(N)-G(N+1), LP=G(N+1)
    for scale in range(num_scales + 1):
        if w is not None:
            lp = weighted_low_pass(d, w, sigma, scale=scale)
        else:
            lp = low_pass(d, scale=scale)

        # Compute this level of the pyramid and append
        lpyr.append(previous - lp)
        # Replace the previous
        previous = lp

    # The low pass is left
    lpyr.append(lp)

    return lpyr


def band_pass_derivative_filters(d, w=None, sigma=0.001, num_scales=NUM_SCALES):
    """Multi-scale image filters from the zeroth, first, and
    second order gaussian derivatives with signal likelihood weighting

    :param d: 2D or 3D numpy array
    :param w: signal likelihood
    :param sigma: noise level
    :param num_scales: int number of scales
    :return: list, one dictionary per level

    The multi scale bandpass derivatives are organized by scale,
    zero (band pass), first (gradient), second (hessian)
    """

    # Initialize the features list
    bpdf = [{} for _scale in range(num_scales)]

    for scale in range(num_scales):
        # Bandpass: zeroth order gaussian derivative
        if w is not None:
            bpdf[scale]["bandpass"] = [weighted_band_pass(d, w, sigma, scale)]
        else:
            bpdf[scale]["bandpass"] = [band_pass(d, scale)]

        # Gradient: first order gaussian derivates
        if w is not None:
            bpdf[scale]["gradient"] = weighted_gradient_band_pass(d, w, sigma, scale)
        else:
            bpdf[scale]["gradient"] = gradient_band_pass(d, scale)

        # Hessian: second order gaussian derivatives
        if w is not None:
            bpdf[scale]["hessian"] = weighted_hessian_band_pass(d, w, sigma, scale)
        else:
            bpdf[scale]["hessian"] = hessian_band_pass(d, scale)

    return bpdf


def rotational_invariants(f):
    """Compute rotational invariants of a set of band pass derivative filters
    """
    # At each level
    #    bandpass:
    #       2 rotational invariant: bp, |bp|
    #    gradient:
    #       1 rotational invariant: |g|
    #    hessian:
    #       in 2D, 3 rotational invariants: Tr(H), Det(H), |H|
    #       in 3D, 4 rotational invariants: Tr(H), Det(H), Sec, |H|

    # Initialize the features list
    num_scales = len(f)
    rot = [{} for _scale in range(num_scales)]

    for scale in range(num_scales):
        # G0
        rot[scale]["bandpass"] = [
            f[scale]["bandpass"][0],
            np.abs(f[scale]["bandpass"][0]),
        ]
        # G1
        rot[scale]["gradient"] = [gradient_amplitude(f[scale]["gradient"])]
        # G2
        rot[scale]["hessian"] = list(hessian_rot(f[scale]["hessian"]))

    return rot


def riff(
    data,
    num_scales=NUM_SCALES,
    normalization_scale=NORMALIZATION_SCALE,
    local_normalization=False,
):
    """Rotational invariant front end features
    """

    # Global scaling, signal likelihood, noise level
    f, w, sigma = global_scale(data)

    # Local scale normalization
    f_c = local_scale_normalization(f, w, sigma, normalization_scale)

    # Multi-scale band pass derivative filters
    bpdf = band_pass_derivative_filters(f_c, w, sigma, num_scales, normalization_scale)

    # Local gain control
    if local_normalization:
        _normalize_bpd_filters(bpdf, w, sigma, normalization_scale)

    # Rotational invariants
    rot = rotational_invariants(bpdf)

    return rot
