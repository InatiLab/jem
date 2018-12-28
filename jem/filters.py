import math
import numpy as np
from nibabel.eulerangles import euler2mat
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


def _crop_filter(input_filter, scale, radius=None):
    """
    Crop an image domain filter
    """
    ndim = input_filter.ndim

    if radius is None:
        radius = 4 * 2 ** scale

    if ndim == 2:
        nx, ny = input_filter.shape
        output_filter = input_filter[
            int(nx / 2 - radius) : int(nx / 2 + radius + 1),
            int(ny / 2 - radius) : int(ny / 2 + radius + 1),
        ]
    elif ndim == 3:
        nx, ny, nz = input_filter.shape
        output_filter = input_filter[
            int(nx / 2 - radius) : int(nx / 2 + radius + 1),
            int(ny / 2 - radius) : int(ny / 2 + radius + 1),
            int(nz / 2 - radius) : int(nz / 2 + radius + 1),
        ]
    else:
        raise RuntimeError(
            "Unsupported number of dimensions {}. We only supports 2 or 3D arrays.".format(
                ndim
            )
        )

    return output_filter


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


def gaussian_kernel(shape, scale=1):
    """
    Rotationally symmetric Gaussian filter kernel in the image domain
    """

    # Get the radius scaled coordinate system
    r = _radius(shape, scale)

    # Compute filter as a function of radius
    g = np.exp(-0.5 * r ** 2)

    # The filter in the image domain
    im_filter = np.real(fftshift(ifftn(fftshift(g))))

    # Cropped
    im_filter = _crop_filter(im_filter, scale)

    return im_filter


def high_pass(data, scale):
    """
    High pass filter
    data - G(data,s)
    """
    hp = data - gaussian(data, scale=scale)

    return hp


def high_pass_kernel(shape, scale):
    """
    High pass filter kernel in the image domain
    """
    hp = -1.0 * gaussian_kernel(shape, scale=scale)
    if hp.ndim == 2:
        nx, ny = hp.shape
        hp[nx // 2, ny // 2] += 1.0
    else:
        nx, ny, nz = hp.shape
        hp[nx // 2, ny // 2, nz // 2] += 1.0

    return hp


def low_pass(data, scale):
    """
    Low pass filter
    G(data,s)
    """
    lp = gaussian(data, scale=scale)

    return lp


def low_pass_kernel(shape, scale):
    """
    Low pass filter kernel in the image domain
    """
    lp = gaussian_kernel(shape, scale=scale)
    lp = _crop_filter(lp, scale)

    return lp


def band_pass(data, scale_one, scale_two):
    """
    Band pass filter
    Difference of two gaussians
    G(data, s1) - G(data, s2)
    """
    r_1 = _radius(data.shape, scale_one)
    r_2 = _radius(data.shape, scale_two)
    g = np.exp(-0.5 * r_1 ** 2) - np.exp(-0.5 * r_2 ** 2)
    bp = ifftn(ifftshift(g * fftshift(fftn(data))))
    if np.isrealobj(data):
        return np.real(bp)
    else:
        return bp


def band_pass_kernel(shape, scale_one, scale_two):
    """
    Band pass filter kernel in the image domain
    """
    r_1 = _radius(shape, scale_one)
    r_2 = _radius(shape, scale_two)
    g = np.exp(-0.5 * r_1 ** 2) - np.exp(-0.5 * r_2 ** 2)
    bp = np.real(fftshift(ifftn(ifftshift(g))))
    bp = _crop_filter(bp, scale_two)
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


def gradient_kernel(shape, scale=1):
    """
    Gradient, Gaussian 1st order partial derivative filter kernel in the image domain
    """

    # Gausian gradient in each direction
    # i*x*g, i*y*g, i*z*g etc.
    ndim = len(shape)

    if ndim == 2:
        x, y = _scale_coordinates(shape, scale)
        rsq = x ** 2 + y ** 2
        g = np.exp(-0.5 * rsq)
        dx = np.real(fftshift(ifftn(ifftshift(1j * x * g))))
        dy = np.real(fftshift(ifftn(ifftshift(1j * y * g))))
        dx = _crop_filter(dx, scale)
        dy = _crop_filter(dy, scale)
        return [dx, dy]

    elif ndim == 3:
        x, y, z = _scale_coordinates(shape, scale)
        rsq = x ** 2 + y ** 2 + z ** 2
        g = np.exp(-0.5 * rsq)
        dx = np.real(fftshift(ifftn(ifftshift(1j * x * g))))
        dy = np.real(fftshift(ifftn(ifftshift(1j * y * g))))
        dz = np.real(fftshift(ifftn(ifftshift(1j * z * g))))
        dx = _crop_filter(dx, scale)
        dy = _crop_filter(dy, scale)
        dz = _crop_filter(dz, scale)
        return [dx, dy, dz]
    else:
        raise RuntimeError(
            "Unsupported number of dimensions {}. We only supports 2 or 3D arrays.".format(
                ndim
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


def hessian_kernel(shape, scale=1):
    """
    Hessian, Gaussian 2nd order partial derivatives filter kernel in the image domain
    """

    # Gausian 2nd derivative in each direction
    # (i*x)*(i*y)*g, etc
    ndim = len(shape)

    if ndim == 2:
        x, y = _scale_coordinates(shape, scale)
        rsq = x ** 2 + y ** 2
        g = np.exp(-0.5 * rsq)
        dxx = np.real(fftshift(ifftn(ifftshift(-1.0 * x * x * g))))
        dxy = np.real(fftshift(ifftn(ifftshift(-1.0 * x * y * g))))
        dyy = np.real(fftshift(ifftn(ifftshift(-1.0 * y * y * g))))
        dxx = _crop_filter(dxx, scale)
        dxy = _crop_filter(dxy, scale)
        dyy = _crop_filter(dyy, scale)
        return [dxx, dxy, dyy]

    elif ndim == 3:
        x, y, z = _scale_coordinates(shape, scale)
        rsq = x ** 2 + y ** 2 + z ** 2
        g = np.exp(-0.5 * rsq)
        dxx = np.real(fftshift(ifftn(ifftshift(-1.0 * x * x * g))))
        dxy = np.real(fftshift(ifftn(ifftshift(-1.0 * x * y * g))))
        dxz = np.real(fftshift(ifftn(ifftshift(-1.0 * x * z * g))))
        dyy = np.real(fftshift(ifftn(ifftshift(-1.0 * y * y * g))))
        dyz = np.real(fftshift(ifftn(ifftshift(-1.0 * y * z * g))))
        dzz = np.real(fftshift(ifftn(ifftshift(-1.0 * z * z * g))))
        dxx = _crop_filter(dxx, scale + 1)
        dxy = _crop_filter(dxy, scale + 1)
        dxz = _crop_filter(dxz, scale + 1)
        dyy = _crop_filter(dyy, scale + 1)
        dyz = _crop_filter(dyz, scale + 1)
        dzz = _crop_filter(dzz, scale + 1)
        return [dxx, dxy, dxz, dyy, dyz, dzz]

    else:
        raise RuntimeError(
            "Unsupported number of dimensions {}. We only supports 2 or 3D arrays.".format(
                ndim
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
        g = np.exp(-0.5 * rsq) - np.exp(-0.5 * 4 * rsq)
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


def gradient_band_pass_kernel(shape, scale=1):
    """
    Gradient, Gaussian 1st order partial derivative filter kernel in the image domain
    """

    # Gausian gradient in each direction
    # g = G(s) - G(s+1)
    # i*x*g, i*y*g, i*z*g etc.

    ndim = len(shape)

    if ndim == 2:
        x, y = _scale_coordinates(shape, scale)
        rsq = x ** 2 + y ** 2
        g = np.exp(-0.5 * rsq) - np.exp(-0.5 * 4 * rsq)
        dx = np.real(fftshift(ifftn(ifftshift(1j * x * g))))
        dy = np.real(fftshift(ifftn(ifftshift(1j * y * g))))
        dx = _crop_filter(dx, scale + 1)
        dy = _crop_filter(dy, scale + 1)
        return [dx, dy]

    elif ndim == 3:
        x, y, z = _scale_coordinates(shape, scale)
        rsq = x ** 2 + y ** 2 + z ** 2
        g = np.exp(-0.5 * rsq) - np.exp(-0.5 * 4 * rsq)
        dx = np.real(fftshift(ifftn(ifftshift(1j * x * g))))
        dy = np.real(fftshift(ifftn(ifftshift(1j * y * g))))
        dz = np.real(fftshift(ifftn(ifftshift(1j * z * g))))
        dx = _crop_filter(dx, scale + 1)
        dy = _crop_filter(dy, scale + 1)
        dz = _crop_filter(dz, scale + 1)
        return [dx, dy, dz]
    else:
        raise RuntimeError(
            "Unsupported number of dimensions {}. We only supports 2 or 3D arrays.".format(
                ndim
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
        g = np.exp(-0.5 * rsq) - np.exp(-0.5 * 4 * rsq)
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


def hessian_band_pass_kernel(shape, scale=1):
    """
    Hessian, Gaussian 2nd order partial derivatives filter kernel in the image domain
    """

    # Gausian 2nd derivative in each direction
    # g = G(s) - G(s+1)
    # from one scale to the next r**2 -> 4*r**2
    # (i*x)*(i*y)*g, etc

    ndim = len(shape)

    if ndim == 2:
        x, y = _scale_coordinates(shape, scale)
        rsq = x ** 2 + y ** 2
        g = np.exp(-0.5 * rsq) - np.exp(-0.5 * 4 * rsq)
        dxx = np.real(fftshift(ifftn(ifftshift(-1.0 * x * x * g))))
        dxy = np.real(fftshift(ifftn(ifftshift(-1.0 * x * y * g))))
        dyy = np.real(fftshift(ifftn(ifftshift(-1.0 * y * y * g))))
        dxx = _crop_filter(dxx, scale + 1)
        dxy = _crop_filter(dxy, scale + 1)
        dyy = _crop_filter(dyy, scale + 1)
        return [dxx, dxy, dyy]

    elif ndim == 3:
        x, y, z = _scale_coordinates(shape, scale)
        rsq = x ** 2 + y ** 2 + z ** 2
        g = np.exp(-0.5 * rsq) - np.exp(-0.5 * 4 * rsq)
        dxx = np.real(fftshift(ifftn(ifftshift(-1.0 * x * x * g))))
        dxy = np.real(fftshift(ifftn(ifftshift(-1.0 * x * y * g))))
        dxz = np.real(fftshift(ifftn(ifftshift(-1.0 * x * z * g))))
        dyy = np.real(fftshift(ifftn(ifftshift(-1.0 * y * y * g))))
        dyz = np.real(fftshift(ifftn(ifftshift(-1.0 * y * z * g))))
        dzz = np.real(fftshift(ifftn(ifftshift(-1.0 * z * z * g))))
        dxx = _crop_filter(dxx, scale + 1)
        dxy = _crop_filter(dxy, scale + 1)
        dxz = _crop_filter(dxz, scale + 1)
        dyy = _crop_filter(dyy, scale + 1)
        dyz = _crop_filter(dyz, scale + 1)
        dzz = _crop_filter(dzz, scale + 1)
        return [dxx, dxy, dxz, dyy, dyz, dzz]

    else:
        raise RuntimeError(
            "Unsupported number of dimensions {}. We only supports 2 or 3D arrays.".format(
                ndim
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


def rotate_gradient_2d(gx, gy, z=0.0):
    """Rotate a 2d gradient or band pass gradient.

    gp = R * g
    """
    R = euler2mat(z=z)[:2, :2]
    g = np.zeros([2, gx.size], dtype=gx.dtype)
    g[0, :] = gx.ravel()
    g[1, :] = gy.ravel()
    gp = R @ g
    gxp = gp[0].reshape(gx.shape)
    gyp = gp[1].reshape(gy.shape)
    return gxp, gyp


def rotate_gradient_3d(gx, gy, gz, z=0.0, y=0.0, x=0.0):
    """Rotate a 3d gradient or band pass gradient.

    gp = R * g
    """
    R = euler2mat(z=z, y=y, x=x)
    g = np.zeros([3, gx.size], dtype=gx.dtype)
    g[0, :] = gx.ravel()
    g[1, :] = gy.ravel()
    g[2, :] = gz.ravel()
    gp = R @ g
    gxp = gp[0].reshape(gx.shape)
    gyp = gp[1].reshape(gy.shape)
    gzp = gp[2].reshape(gz.shape)
    return gxp, gyp, gzp


def rotate_hessian_2d(dxx, dxy, dyy, z):
    """Rotate a 2d hessian or band pass hessian.

    Hp = R' * H * R
    """
    R = euler2mat(z)[:2, :2]

    H = np.zeros([2, 2, dxx.size], dtype=dxx.dtype)
    H[0, 0, :] = dxx.ravel()
    H[0, 1, :] = dxy.ravel()
    H[1, 0, :] = H[0, 1, :]
    H[1, 1, :] = dyy.ravel()

    #     Hp = np.zeros(H.shape, dtype=H.dtype)
    #     for n in range(H.shape[2]):
    #         Hp[:,:,n] = R.transpose() @ H[:,:,n] @ R
    Hp = np.dot(np.dot(R.transpose(), H).transpose(2, 0, 1), R).transpose(1, 2, 0)

    dxxp = Hp[0, 0, :].reshape(dxx.shape)
    dxyp = Hp[0, 1, :].reshape(dxy.shape)
    dyyp = Hp[1, 1, :].reshape(dyy.shape)

    return dxxp, dxyp, dyyp


def rotate_hessian_3d(dxx, dxy, dxz, dyy, dyz, dzz, z, y=0.0, x=0.0):
    """Rotate a 3d hessian or band pass hessian.

    Hp = R' * H * R
    """
    R = euler2mat(z)

    H = np.zeros([3, 3, dxx.size], dtype=dxx.dtype)
    H[0, 0, :] = dxx.ravel()
    H[0, 1, :] = dxy.ravel()
    H[0, 2, :] = dxz.ravel()
    H[1, 1, :] = dyy.ravel()
    H[1, 2, :] = dyz.ravel()
    H[2, 2, :] = dzz.ravel()
    H[1, 0, :] = H[0, 1, :]
    H[2, 0, :] = H[0, 2, :]
    H[2, 1, :] = H[1, 2, :]

    #     Hp = np.zeros(H.shape, dtype=H.dtype)
    #     for n in range(H.shape[2]):
    #         Hp[:,:,n] = R.transpose() @ H[:,:,n] @ R
    Hp = np.dot(np.dot(R.transpose(), H).transpose(2, 0, 1), R).transpose(1, 2, 0)

    dxxp = Hp[0, 0, :].reshape(dxx.shape)
    dxyp = Hp[0, 1, :].reshape(dxy.shape)
    dxzp = Hp[0, 2, :].reshape(dxz.shape)
    dyyp = Hp[1, 1, :].reshape(dyy.shape)
    dyzp = Hp[1, 2, :].reshape(dyz.shape)
    dzzp = Hp[2, 2, :].reshape(dzz.shape)

    return dxxp, dxyp, dxzp, dyyp, dyzp, dzzp
