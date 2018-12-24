import math
import numpy as np
from numpy.fft import fftn, ifftn, fftshift, ifftshift


def _scale_coordinates(shape, scale):
    """
    Compute the scaled image coordinates on the box [-pi,pi]^d
    """
    ndim = len(shape)
    # Compute the scaled coordinate system
    if ndim == 2:
        nx, ny = shape
        fx, fy = np.meshgrid(
            np.linspace(-nx / 2, nx / 2, nx),
            np.linspace(-ny / 2, ny / 2, ny),
            indexing="ij",
        )
        sx = nx / (2.0 * math.pi) / 2 ** scale
        sy = ny / (2.0 * math.pi) / 2 ** scale
        x = fx / sx
        y = fy / sy
        return x, y

    elif ndim == 3:
        nx, ny, nz = shape
        fx, fy, fz = np.meshgrid(
            np.linspace(-nx / 2, nx / 2, nx),
            np.linspace(-ny / 2, ny / 2, ny),
            np.linspace(-nz / 2, nz / 2, nz),
            indexing="ij",
        )
        sx = nx / (2.0 * math.pi) / 2 ** scale
        sy = ny / (2.0 * math.pi) / 2 ** scale
        sz = nz / (2.0 * math.pi) / 2 ** scale
        x = fx / sx
        y = fy / sy
        z = fz / sz

        return x, y, z

    else:
        raise RuntimeError(
            "Unsupported number of dimensions {}. We only supports 2 or 3D arrays.".format(
                len(shape)
            )
        )


def _radius(shape, scale):
    """
    Compute the radius in scaled image coordinates
    """
    x = _scale_coordinates(shape, scale)
    r = np.sqrt(np.sum(np.square(x), axis=0))
    return r


def _pinv(x, p=2):
    """
    Pseudoinverse with regularization

    Use the lowest p percent of the signal as an estimate of the noise floor
    """
    d = np.abs(x)
    # find the lowest p% of the non-zero signal
    # to use for regularization
    s = np.percentile(d[d > 0], [p])
    ix = x / (d ** 2 + s ** 2)
    return ix


def radial(data, func, scale=1, truncate=False):
    """
    Rotationally symmetric filter in the fourier domain with truncation
    """

    known_filters = ["gaussian", "dog", "laplacian"]
    if func.lower() not in known_filters:
        raise RuntimeError(
            "Unsupported filter function error {}.  Must be one of {}.".format(
                func, known_filters
            )
        )

    # Get the radius scaled coordinate system
    r = _radius(data.shape, scale)
    rsq = r ** 2

    # Compute filter as a function of radius
    if func.lower() == "gaussian":
        # Gaussian, 0th Hermite, etc.
        g = np.exp(-0.5 * rsq)
    elif func.lower() == "dog":
        # Difference of Gaussians, one level apart
        g = np.exp(-0.5 * rsq) - np.exp(-0.5 * (2.0 * rsq))
    elif func.lower() == "derivative":
        # Derivative of Gaussian, 1st Hermite
        g = -1.0 / (math.pi ** 2) * (1.0 - math.pi ** 2 * rsq) * np.exp(-0.5 * rsq)
    elif func.lower() == "laplacian":
        # Laplacian of Gaussian, 2nd Hermite, Marr, Sombrero, Ricker, etc.
        g = -1.0 / (math.pi ** 2) * (1.0 - math.pi ** 2 * rsq) * np.exp(-0.5 * rsq)
    else:
        raise RuntimeError("Unkown filter function {}.".format(func))

    # Truncate on a sphere of r=pi^2
    if truncate:
        g[r > math.pi] = 0.0

    # Apply the filter
    output = ifftn(ifftshift(g * fftshift(fftn(data))))

    # Ensure that real functions stay real
    if np.isrealobj(data):
        return np.real(output)
    else:
        return output


def gaussian(data, scale=1):
    """
    Rotationally symmetric Gaussian filter in the fourier domain
    """

    # Get the radius scaled coordinate system
    r = _radius(data.shape, scale)

    # Compute filter as a function of radius
    g = np.exp(-0.5 * r ** 2)

    # Apply the filter
    output = ifftn(ifftshift(g * fftshift(fftn(data))))

    # Ensure that real functions stay real
    if np.isrealobj(data):
        return np.real(output)
    else:
        return output


def high_pass(data, scale):
    """
    High pass filter
    data - G(data,s)
    """
    hp = data - gaussian(data, scale=scale)

    return hp


def low_pass(data, scale):
    """
    Low pass filter
    G(data,s)
    """
    lp = gaussian(data, scale=scale)

    return lp


def band_pass(data, scale_one, scale_two):
    """
    Band pass filter
    Difference of two gaussians
    G(data, s1) - G(data, s2)
    """
    bp = gaussian(data, scale=scale_one) - gaussian(data, scale=scale_two)

    return bp


def gradient(data, scale=1):
    """
    Gradient, Gaussian 1st order partial derivative filter in the fourier domain
    """

    # Gausian gradient in each direction
    # i*x*g, i*y*g, i*z*g etc.

    # Get the scaled coordinate system
    if data.ndim == 2:
        x, y = _scale_coordinates(data.shape, scale)
        rsq = x ** 2 + y ** 2
        g = np.exp(-0.5 * rsq)
        temp = 1j * g * fftshift(fftn(data))
        dx = ifftn(ifftshift(x * temp))
        dy = ifftn(ifftshift(y * temp))
        # Ensure that real functions stay real
        if np.isrealobj(data):
            dx = np.real(dx)
            dy = np.real(dy)
            return [dx, dy]

    elif data.ndim == 3:
        x, y, z = _scale_coordinates(data.shape, scale)
        rsq = x ** 2 + y ** 2 + z ** 2
        g = np.exp(-0.5 * rsq)
        temp = 1j * g * fftshift(fftn(data))
        dx = ifftn(ifftshift(x * temp))
        dy = ifftn(ifftshift(y * temp))
        dz = ifftn(ifftshift(z * temp))
        # Ensure that real functions stay real
        if np.isrealobj(data):
            dx = np.real(dx)
            dy = np.real(dy)
            dz = np.real(dy)
            return [dx, dy, dz]
    else:
        raise RuntimeError(
            "Unsupported number of dimensions {}. We only supports 2 or 3D arrays.".format(
                data.ndim
            )
        )


def hessian(data, scale=1):
    """
    Hessian, Gaussian 2nd order partial derivatives filter in the fourier domain
    """

    # Gausian 2nd derivative in each direction
    # (i*x)*(i*y)*g, etc

    # Get the scaled coordinate system
    if data.ndim == 2:
        x, y = _scale_coordinates(data.shape, scale)
        rsq = x ** 2 + y ** 2
        g = np.exp(-0.5 * rsq)
        temp = -1.0 * g * fftshift(fftn(data))
        dxx = ifftn(ifftshift(x * x * temp))
        dxy = ifftn(ifftshift(x * y * temp))
        dyy = ifftn(ifftshift(y * y * temp))
        # Ensure that real functions stay real
        if np.isrealobj(data):
            dxx = np.real(dxx)
            dxy = np.real(dxy)
            dyy = np.real(dxy)
        return [dxx, dxy, dyy]

    elif data.ndim == 3:
        x, y, z = _scale_coordinates(data.shape, scale)
        rsq = x ** 2 + y ** 2 + z ** 2
        g = np.exp(-0.5 * rsq)
        temp = -1.0 * g * fftshift(fftn(data))
        dxx = ifftn(ifftshift(x * x * temp))
        dxy = ifftn(ifftshift(x * y * temp))
        dxz = ifftn(ifftshift(x * z * temp))
        dyy = ifftn(ifftshift(y * y * temp))
        dyz = ifftn(ifftshift(y * z * temp))
        dzz = ifftn(ifftshift(z * z * temp))
        # Ensure that real functions stay real
        if np.isrealobj(data):
            dxx = np.real(dxx)
            dxy = np.real(dxy)
            dxz = np.real(dxz)
            dyy = np.real(dyy)
            dyz = np.real(dyz)
            dzz = np.real(dzz)
        return [dxx, dxy, dxz, dyy, dyz, dzz]

    else:
        raise RuntimeError(
            "Unsupported number of dimensions {}. We only supports 2 or 3D arrays.".format(
                data.ndim
            )
        )


def gradient_band_pass(data, scale=1):
    """
    Gradient, Gaussian 1st order partial derivative filter in the fourier domain
    """

    # Gausian gradient in each direction
    # g = G(s) - G(s+1)
    # i*x*g, i*y*g, i*z*g etc.

    # Get the scaled coordinate system
    if data.ndim == 2:
        x, y = _scale_coordinates(data.shape, scale)
        rsq = x ** 2 + y ** 2
        g = np.exp(-0.5 * rsq) - np.exp(-0.5 * 4 * rsq)
        temp = 1j * g * fftshift(fftn(data))
        dx = ifftn(ifftshift(x * temp))
        dy = ifftn(ifftshift(y * temp))
        # Ensure that real functions stay real
        if np.isrealobj(data):
            dx = np.real(dx)
            dy = np.real(dy)
            return [dx, dy]

    elif data.ndim == 3:
        x, y, z = _scale_coordinates(data.shape, scale)
        rsq = x ** 2 + y ** 2 + z ** 2
        g = np.exp(-0.5 * rsq)
        temp = 1j * g * fftshift(fftn(data))
        dx = ifftn(ifftshift(x * temp))
        dy = ifftn(ifftshift(y * temp))
        dz = ifftn(ifftshift(z * temp))
        # Ensure that real functions stay real
        if np.isrealobj(data):
            dx = np.real(dx)
            dy = np.real(dy)
            dz = np.real(dy)
            return [dx, dy, dz]
    else:
        raise RuntimeError(
            "Unsupported number of dimensions {}. We only supports 2 or 3D arrays.".format(
                data.ndim
            )
        )


def hessian_band_pass(data, scale=1):
    """
    Hessian, Gaussian 2nd order partial derivatives filter in the fourier domain
    """

    # Gausian 2nd derivative in each direction
    # g = G(s) - G(s+1)
    # from one scale to the next r**2 -> 4*r**2
    # (i*x)*(i*y)*g, etc

    # Get the scaled coordinate system
    if data.ndim == 2:
        x, y = _scale_coordinates(data.shape, scale)
        rsq = x ** 2 + y ** 2
        g = np.exp(-0.5 * rsq) - np.exp(-0.5 * 4 * rsq)
        temp = -1.0 * g * fftshift(fftn(data))
        dxx = ifftn(ifftshift(x * x * temp))
        dxy = ifftn(ifftshift(x * y * temp))
        dyy = ifftn(ifftshift(y * y * temp))
        # Ensure that real functions stay real
        if np.isrealobj(data):
            dxx = np.real(dxx)
            dxy = np.real(dxy)
            dyy = np.real(dyy)
        return [dxx, dxy, dyy]

    elif data.ndim == 3:
        x, y, z = _scale_coordinates(data.shape, scale)
        rsq = x ** 2 + y ** 2 + z ** 2
        g = np.exp(-0.5 * rsq)
        temp = -1.0 * g * fftshift(fftn(data))
        dxx = ifftn(ifftshift(x * x * temp))
        dxy = ifftn(ifftshift(x * y * temp))
        dxz = ifftn(ifftshift(x * z * temp))
        dyy = ifftn(ifftshift(y * y * temp))
        dyz = ifftn(ifftshift(y * z * temp))
        dzz = ifftn(ifftshift(z * z * temp))
        # Ensure that real functions stay real
        if np.isrealobj(data):
            dxx = np.real(dxx)
            dxy = np.real(dxy)
            dxz = np.real(dxz)
            dyy = np.real(dyy)
            dyz = np.real(dyz)
            dzz = np.real(dzz)
        return [dxx, dxy, dxz, dyy, dyz, dzz]

    else:
        raise RuntimeError(
            "Unsupported number of dimensions {}. We only supports 2 or 3D arrays.".format(
                data.ndim
            )
        )


def hessian_power(h):
    """
    Power in the hessian filter band
    Frobenius norm squared
    """
    if len(h) == 2:
        p = np.abs(h[0]) ** 2 + 2 * np.abs(h[1]) ** 2 + np.abs(h[2]) ** 2
    elif len(h) == 6:
        p = (
            np.abs(h[0]) ** 2
            + 2 * np.abs(h[1]) ** 2
            + 2 * np.abs(h[2]) ** 2
            + np.abs(h[3]) ** 2
            + 2 * np.abs(h[4]) ** 2
            + np.abs(h[5]) ** 2
        )
    else:
        raise RuntimeError("Unsupported number of dimensions {}.".format(len(h)))
    return p


def gradient_rot(g):
    """
    Rotational invariant of the gradient
    """
    if len(g) == 2:
        # [dx, dy]
        g = np.sqrt(np.abs(g[0]) ** 2 + np.abs(g[1]) ** 2)

    elif len(g) == 3:
        # [dx, dy, dz]
        g = np.sqrt(np.abs(g[0]) ** 2 + np.abs(g[1]) ** 2 + np.abs(g[2]))

    else:
        raise RuntimeError("Unsupported number of dimensions {}.".format(len(g)))

    return g


def hessian_rot(h):
    """
    Rotational invariants of the hessian
    """

    if len(h) == 3:
        # [dxx, dxy, dyy]
        # 1st trace l1 + l2
        trace = h[0] + h[2]
        # 2nd determinant l1*l2
        # det = Dxx*Dyy - Dxy*Dyx
        det = h[0] * h[2] - h[1] * h[1]
        # frobenius norm sqrt(l1**2 + l2**2)
        frobenius = np.sqrt(
            np.abs(h[0]) ** 2 + 2 * np.abs(h[1]) ** 2 + np.abs(h[2]) ** 2
        )

        return (trace, det, frobenius)

    elif len(h) == 6:
        # [dxx, dxy, dxz, dyy, dyz, dzz]
        # 1st trace l1 + l2 + l3
        trace = h[0] + h[3] + h[5]
        # 2nd l1*l2 + l1*l3 + l2*l3
        # sec = Dxx*Dyy - Dxy*Dyx + Dxx*Dzz - Dxz*Dzx + Dyy*Dzz - Dyz*Dzy
        sec = (
            h[0] * h[3]
            - h[1] * h[1]
            + h[0] * h[5]
            - h[2] * h[2]
            + h[3] * h[5]
            - h[4] * h[4]
        )
        # 3rd determinant l1*l2*l3
        # det = Dxx*(Dyy*Dzz - Dyz*Dzy) - Dxy*(Dyx*Dzz - Dyz*Dzx) + Dxz*(Dyx*Dzy - Dyy*Dzx)
        det = (
            h[0] * (h[3] * h[5] - h[4] * h[4])
            - h[1] * (h[1] * h[5] - h[4] * h[2])
            + h[2] * (h[1] * h[4] - h[3] * h[2])
        )
        # frobenius norm sqrt(l1**2 + l2**2 + l3**2)
        frobenius = (
            np.sqrt(
                np.abs(h[0]) ** 2
                + 2 * np.abs(h[1]) ** 2
                + 2 * np.abs(h[2]) ** 2
                + np.abs(h[3]) ** 2
                + 2 * np.abs(h[4]) ** 2
            )
            + np.abs(h[5]) ** 2
        )

        return (trace, sec, det, frobenius)

    else:
        raise RuntimeError("Unsupported number of dimensions {}.".format(len(h)))
