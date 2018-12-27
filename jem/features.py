import numpy as np
from .filters import high_pass, low_pass
from .filters import band_pass, gradient_band_pass, hessian_band_pass
from .filters import gradient_rot, hessian_rot


# Number of spatial scales
NUM_SCALES = 3

# Scales for normalization / gain control
NORMALIZATION_SCALE = 5


def _get_dim(data):
    """
    Data dimensionality with error checking
    """

    if data.ndim == 2:
        ndim = 2
    elif data.ndim == 3:
        ndim = 3
    else:
        raise RuntimeError(
            "Unsupported number of dimensions {}. We only supports 2 or 3D arrays.".format(
                data.ndim
            )
        )

    return ndim


def rectify(data, polarity=1):
    """
    Rectify the data, polarity is either 1 or -1
    """
    if polarity not in [1, -1]:
        raise RuntimeError("Rectification polarity is either 1 or -1.")

    output = polarity * data * ((polarity * data) > 0)

    return output


def input_normalization(data, scale=NORMALIZATION_SCALE):
    """
    Normalize in a manner similar to the retina
    """
    z = high_pass(data, scale)
    a = np.abs(z)
    sigma = np.mean(a)
    f = sigma + low_pass(a, scale)
    y = data / f

    return y


def front_end_features(data, nscales=NUM_SCALES, normalization_scale=None):
    """
    Compute multi-scale image features from the zeroth, first, and
    second order gaussian derivatives with divisive normalization

    :param data:  2D or 3D numpy array
    :param nscales: int number of scales
    :normalization_scale: int scale for stage 1 input normalization
    :return: list of 2D or 3D numpy arrays and list of names
    """

    if normalization_scale is None:
        normalization_scale = NORMALIZATION_SCALE
    if normalization_scale < nscales:
        raise RuntimeError(
            "Normalization scale cannot be less than the number of Scales."
        )

    # Initialize the features list and the names list
    t = []
    names = []

    # Stage 1: High pass filter and local normalization
    d = input_normalization(data, scale=normalization_scale)

    # Stage 2: Feature generation
    # The features are organized by level
    # At each level:
    #    bandpass
    #    gradient
    #    hessian

    for lev in range(nscales):
        # Bandpass: zeroth order gaussian derivative
        feat = band_pass(d, scale_one=lev, scale_two=lev + 1)
        t.append(feat)
        names.append(f"Band Pass L{lev}")

        # Gradient: first order gaussian derivates
        feat = gradient_band_pass(d, scale=lev)
        for n in range(len(feat)):
            t.append(feat[n])
            names.append(f"Gradient L{lev}M{n}")

        # Hessian: second order gaussian derivatives
        feat = hessian_band_pass(d, scale=lev)
        for n in range(len(feat)):
            t.append(feat[n])
            names.append(f"Hessian L{lev}M{n}")

    return t, names


def riff(data, nscales=NUM_SCALES, normalization_scale=None):
    """
    Compute multi-scale rotationally invariant image features
    from the zeroth, first, and second order gaussian derivatives
    with divisive normalization

    :param data:  2D or 3D numpy array
    :param nscales: int number of scales
    :normalization_scale: int scale for stage 1 input normalization
    :return: list of 3*nscales+1 2D or 3D numpy arrays and list of names
    """

    if normalization_scale is None:
        normalization_scale = NORMALIZATION_SCALE
    if normalization_scale < nscales:
        raise RuntimeError(
            "Normalization scale cannot be less than the number of Scales."
        )

    # Initialize the textures list and the names list
    t = []
    names = []

    # Stage 1: High pass filter and local normalization
    d = input_normalization(data, scale=normalization_scale)

    # Stage 2: Feature generation
    # The features are organized by level
    # At each level:
    #    bandpass (2 rotational invariant: bp, |bp|)
    #    gradient (1 rotational invariant: |g|)
    #    hessian (4 rotational invariants: Tr(H), Det(H), Sec, |H|)
    #  Keep a running total of the total power from each feature
    total_power = 0.0

    for lev in range(nscales):
        # Bandpass (difference of adjacent gaussians)
        feat = band_pass(d, scale_one=lev, scale_two=lev + 1)
        t.append(feat)
        names.append(f"Band Pass S{lev}")
        t.append(np.abs(feat))
        names.append(f"Band Pass Magnitude S{lev}")
        total_power += np.abs(feat) ** 2

        # Gradient
        g = gradient_band_pass(d, scale=lev)
        feat = gradient_rot(g)
        t.append(feat)
        names.append(f"Gradient Magnitude S{lev}")
        # The power is the square of the gradient amplitude
        total_power += feat ** 2

        # The next set of features are the rotational invariants of the
        # second order gaussian derivatives
        h = hessian_band_pass(d, scale=lev)
        feat = hessian_rot(h)
        t.append(feat[0])
        names.append(f"Laplacian S{lev}")
        if data.ndim == 2:
            t.append(feat[1])
            names.append(f"Hessian Det S{lev}")
            t.append(feat[2])
            names.append(f"Hessian Magnitude S{lev}")
            # The power is the the square of the Frobenius norm
            total_power += feat[2] ** 2
        elif data.ndim == 3:
            t.append(feat[1])
            names.append(f"Hessian R2 S{lev}")
            t.append(feat[2])
            names.append(f"Hessian Det S{lev}")
            t.append(feat[3])
            names.append(f"Hessian Magnitude S{lev}")
            # The power is the the square of the Frobenius norm
            total_power += feat[3] ** 2

    return t, names
