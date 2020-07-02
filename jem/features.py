import numpy as np

from .filters import _pinv, dog, dog_rotational_invariants, gradient_amplitude, hessian_amplitude, hessian_trace

# Number of spatial scales
NUM_SCALES = 4

# Scales for normalization / gain control
NORMALIZATION_SCALE = 5


def rectify(data, polarity=1):
    """
    Rectify the data, polarity is either 1 or -1
    """
    if polarity not in [1, -1]:
        raise RuntimeError("Rectification polarity is either 1 or -1.")

    output = polarity * data * ((polarity * data) > 0)

    return output


def local_scale_normalization(
    d, normalization_scale=NORMALIZATION_SCALE, w=None, sigma=0.001
):
    """
    Local normalization by the signal mean.
    """
    f = d * _pinv(
        dog(d, order=0, scale=normalization_scale, w=w, sigma=sigma), sigma=sigma
    )

    return f


def local_contrast_normalization(
    d, normalization_scale=NORMALIZATION_SCALE, w=None, sigma=0.001
):
    """
    Local contrast normalization
    Subtract the local mean and scale by the local mean absolute deviation
    """
    dev = d - dog(d, order=0, scale=normalization_scale, w=w, sigma=sigma)
    lc = dev * _pinv(
        dog(np.abs(dev), order=0, scale=normalization_scale, w=w, sigma=sigma),
        sigma=sigma,
    )

    return lc


def gaussian_pyramid(d, order=0, num_scales=NUM_SCALES, w=None, sigma=0.001):
    """Pyramid from gaussian derivative filters with signal likelihood weighting

    :param d: 2D or 3D numpy array
    :param order: Gaussian derivative order, 0, 1, or 2
    :param w: signal likelihood
    :param sigma: noise level
    :param nscales: int number of scales
    :return: list

    The pyramid is organized by scale,
    Level 0, Level 1, .., Level num_scales-1
    """

    pyr = [
        dog(d, order=order, scale=scale, w=w, sigma=sigma)
        for scale in range(num_scales)
    ]

    return pyr


def difference_pyramid(pyr, inplace=False):
    """Difference between levels of a derivative of gaussian pyramid

    :param d: gaussian pyramid
    :param inplace: bool, do the work in place or make a copy, default False
    :return: list of differences

    The output has length one less than the input
    diff_pyr = [pyr[0] - pyr[1], pyr[1] - pyr[2], ..., pyr[N-2] - pyr[N-1]]
    """

    N = len(pyr)
    if type(pyr[0]) == list:
        M = len(pyr[0])
    else:
        M = 1

    if inplace:
        output = pyr
    else:
        if M > 1:
            output = [[None] * M] * N
        else:
            output = [None] * N

    # the first N-1 levels
    for n in range(N - 1):
        if M > 1:
            for m in range(M):
                output[n][m] = pyr[n][m] - pyr[n + 1][m]
        else:
            output[n] = pyr[n] - pyr[n + 1]

    # pop off the last level (the low pass)
    output = output[:-1]

    return output


def laplacian_pyramid(d, num_scales=NUM_SCALES, w=None, sigma=0.001):
    """Laplacian pyramid from gaussian filters with signal likelihood weighting

    :param d: 2D or 3D numpy array
    :param w: signal likelihood
    :param sigma: noise level
    :param nscales: int number of scales
    :return: list of length num_scales + 1

    The pyramid is organized by scale
    high pass, level 0, level 2, .., level num_scales-1, low pass

    Adding the output back together produces the original input data
    """

    # Gaussian pyramid.  Get 1 extra level.
    pyr = gaussian_pyramid(d, order=0, num_scales=num_scales + 1, w=w, sigma=sigma)

    # Save the high pass and the low pass
    high_pass = d - pyr[0]
    low_pass = pyr[-1]

    # The difference pyramid (in place)
    pyr = difference_pyramid(pyr, inplace=True)

    # Insert the high at the front and the low pass at the end
    pyr.insert(0, high_pass)
    pyr.append(low_pass)

    return pyr


def front_end_features(d, w=None, sigma=0.001, num_scales=NUM_SCALES):
    """Multi-scale image filters from the zeroth, first, and
    second order gaussian derivatives with signal likelihood weighting

    :param d: 2D or 3D numpy array
    :param w: signal likelihood
    :param sigma: noise level
    :param num_scales: int number of scales
    :return: dictionary {high_pass, zelist, one dictionary per level

    High pass
    Multi scale bandpass derivatives are organized by scale,
    zero (band pass), first (gradient), second (hessian)
    """

    feats = {
        "high_pass": None,
        "zero": [None] * num_scales,
        "one": [None] * num_scales,
        "two": [None] * num_scales,
        "low_pass": None,
    }

    # High pass, zeroth order, low pass
    # i.e. Laplacian pyramid
    pyr = laplacian_pyramid(d, num_scales=num_scales, w=w, sigma=sigma)
    feats["high_pass"] = pyr[0]
    feats["zero"] = pyr[1:-1]
    feats["low_pass"] = pyr[-1]
    del pyr

    # First order, gradient
    feats["one"] = gaussian_pyramid(
        d, order=1, num_scales=num_scales + 1, w=w, sigma=sigma
    )
    feats["one"] = difference_pyramid(feats["one"], inplace=True)

    # Second order, hessian
    feats["two"] = gaussian_pyramid(
        d, order=2, num_scales=num_scales + 1, w=w, sigma=sigma
    )
    feats["two"] = difference_pyramid(feats["two"], inplace=True)

    return feats


def fef_rotational_invariants(fef, inplace=True):
    """Rotational invariants of the front end features
    """

    num_scales = len(fef["zero"])

    # Rotational invariants
    if inplace:
        rfef = fef
    else:
        rfef = {
            "high_pass": None,
            "zero": [None] * num_scales,
            "one": [None] * num_scales,
            "two": [None] * num_scales,
            "low_pass": None,
        }

    rfef["high_pass"] = dog_rotational_invariants(fef["high_pass"], order=0)
    rfef["low_pass"] = [fef["low_pass"]]
    for n in range(num_scales):
        rfef["zero"][n] = dog_rotational_invariants(fef["zero"][n], order=0)
        rfef["one"][n] = dog_rotational_invariants(fef["one"][n], order=1)
        rfef["two"][n] = dog_rotational_invariants(fef["two"][n], order=2)

    return rfef

def gaussian_pyramid_features(d, w, sigma, num_scales=NUM_SCALES):
    """Rotational invariants of the gaussian pyramid features
    """

    gauss = gaussian_pyramid(d, order=0, num_scales=num_scales, w=w, sigma=sigma)
    grad = gaussian_pyramid(d, order=1, num_scales=num_scales, w=w, sigma=sigma)
    grad = [gradient_amplitude(x) for x in grad]
    hess = gaussian_pyramid(d, order=2, num_scales=num_scales, w=w, sigma=sigma)
    lap = [hessian_trace(x) for x in hess]
    norm = [hessian_amplitude(x) for x in hess]
    features = [d]+gauss+grad+lap+norm

    return features
