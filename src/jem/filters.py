import math
import numpy as np
from nibabel.eulerangles import euler2mat
from numpy.fft import fftn, ifftn, fftshift, ifftshift

# Regularization scaling
# For weighted filtering, the regularization term should be a fraction of the noise level
# we use 0.01 by default
REG_SCALE = 0.01


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


def _pad(d, scale):
    """Pad for gaussian convolutions in the fourier domain

    Only need to pad by a quarter width on either side which
    gives 1/2 a kernel width when wrapped.
    """
    p = 2 * 2 ** scale
    pd = np.pad(d, pad_width=p, mode="constant", constant_values=0)
    return pd


def _crop(d, scale):
    """Inverse of pad"""
    p = 2 * 2 ** scale
    if d.ndim == 2:
        return d[p:-p, p:-p]
    elif d.ndim == 3:
        return d[p:-p, p:-p, p:-p]


def _scale_coordinates(shape, scale):
    """
    Compute the scaled image coordinates on the box [-pi,pi]^d
    """
    ndim = len(shape)
    # Compute the scaled coordinate system
    if ndim == 2:
        nx, ny = shape
        fx, fy = np.meshgrid(
            np.linspace(-nx / 2, nx / 2 - 1, nx),
            np.linspace(-ny / 2, ny / 2 - 1, ny),
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
            np.linspace(-nx / 2, nx / 2 - 1, nx),
            np.linspace(-ny / 2, ny / 2 - 1, ny),
            np.linspace(-nz / 2, nz / 2 - 1, nz),
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


def _pinv(x, sigma):
    """
    Pseudoinverse with regularization
    sigma is the noise level
    pinv(x) = x / (x^2 + sigma^2)

    """
    xinv = x / (x ** 2 + sigma ** 2)
    return xinv


def gaussian(data, scale=1):
    """
    Rotationally symmetric Gaussian filter in the fourier domain

    Parametrized by a scale, resolution reduced by 2^(scale+1).
    Assumes the pixels are all the same size.
    Pad and crop to prevent wrap around junk.
    """
    pd = _pad(data, scale)

    # Get the radius scaled coordinate system
    r = _radius(pd.shape, scale)

    # Compute filter as a function of radius
    g = np.exp(-0.5 * r ** 2)

    # Apply the filter
    temp = ifftn(ifftshift(g * fftshift(fftn(pd))))

    # Crop
    output = _crop(temp, scale)

    # Ensure that real functions stay real
    if np.isrealobj(data):
        return np.real(output)
    else:
        return output


def gradient(data, scale=1):
    """
    Gradient, Gaussian 1st order partial derivative filter in the fourier domain
    """

    # Gausian gradient in each direction
    # i*x*g, i*y*g, i*z*g etc.

    # Pad
    pd = _pad(data, scale)

    # Get the scaled coordinate system
    if data.ndim == 2:
        x, y = _scale_coordinates(pd.shape, scale)
        rsq = x ** 2 + y ** 2
        g = np.exp(-0.5 * rsq)
        temp = 1j * g * fftshift(fftn(pd))
        dx = ifftn(ifftshift(x * temp))
        dy = ifftn(ifftshift(y * temp))
        # Crop
        dx = _crop(dx, scale)
        dy = _crop(dy, scale)
        # Ensure that real functions stay real
        if np.isrealobj(data):
            dx = np.real(dx)
            dy = np.real(dy)
        return [dx, dy]

    elif data.ndim == 3:
        x, y, z = _scale_coordinates(pd.shape, scale)
        rsq = x ** 2 + y ** 2 + z ** 2
        g = np.exp(-0.5 * rsq)
        temp = 1j * g * fftshift(fftn(pd))
        dx = ifftn(ifftshift(x * temp))
        dy = ifftn(ifftshift(y * temp))
        dz = ifftn(ifftshift(z * temp))
        # Crop
        dx = _crop(dx, scale)
        dy = _crop(dy, scale)
        dz = _crop(dz, scale)
        # Ensure that real functions stay real
        if np.isrealobj(data):
            dx = np.real(dx)
            dy = np.real(dy)
            dz = np.real(dz)
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

    # Pad
    pd = _pad(data, scale)

    # Get the scaled coordinate system
    if data.ndim == 2:
        x, y = _scale_coordinates(pd.shape, scale)
        rsq = x ** 2 + y ** 2
        g = np.exp(-0.5 * rsq)
        temp = -1.0 * g * fftshift(fftn(pd))
        dxx = ifftn(ifftshift(x * x * temp))
        dxy = ifftn(ifftshift(x * y * temp))
        dyy = ifftn(ifftshift(y * y * temp))
        # Crop
        dxx = _crop(dxx, scale)
        dxy = _crop(dxy, scale)
        dyy = _crop(dyy, scale)
        # Ensure that real functions stay real
        if np.isrealobj(data):
            dxx = np.real(dxx)
            dxy = np.real(dxy)
            dyy = np.real(dyy)
        return [dxx, dxy, dyy]

    elif data.ndim == 3:
        x, y, z = _scale_coordinates(pd.shape, scale)
        rsq = x ** 2 + y ** 2 + z ** 2
        g = np.exp(-0.5 * rsq)
        temp = -1.0 * g * fftshift(fftn(pd))
        dxx = ifftn(ifftshift(x * x * temp))
        dxy = ifftn(ifftshift(x * y * temp))
        dxz = ifftn(ifftshift(x * z * temp))
        dyy = ifftn(ifftshift(y * y * temp))
        dyz = ifftn(ifftshift(y * z * temp))
        dzz = ifftn(ifftshift(z * z * temp))
        # Crop
        dxx = _crop(dxx, scale)
        dxy = _crop(dxy, scale)
        dxz = _crop(dxz, scale)
        dyy = _crop(dyy, scale)
        dyz = _crop(dyz, scale)
        dzz = _crop(dzz, scale)
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


def high_pass(data, scale=1):
    """
    High pass filter
    data - G(data,s)
    """
    hp = data - gaussian(data, scale=scale)

    return hp


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


def gradient_amplitude(g):
    """
    Amplitude in the gradient filter band
    2/Frobenius norm
    """
    amp = np.sqrt(np.sum(np.abs(g) ** 2, axis=0))
    return amp


def hessian_amplitude(h):
    """
    Amplitude in the hessian filter band
    Frobenius norm
    """
    if len(h) == 3:
        a = np.sqrt(np.abs(h[0]) ** 2 + 2 * np.abs(h[1]) ** 2 + np.abs(h[2]) ** 2)
    elif len(h) == 6:
        a = np.sqrt(
            np.abs(h[0]) ** 2
            + 2 * np.abs(h[1]) ** 2
            + 2 * np.abs(h[2]) ** 2
            + np.abs(h[3]) ** 2
            + 2 * np.abs(h[4]) ** 2
            + np.abs(h[5]) ** 2
        )
    else:
        raise RuntimeError("Unsupported number of dimensions {}.".format(len(h)))
    return a

def hessian_det(h):
    """
    Determinant in the hessian filter band
    """
    if len(h) == 3:
        det = h[0] * h[2] - h[1] * h[1]
    elif len(h) == 6:
        det = (
            h[0] * (h[3] * h[5] - h[4] * h[4])
            - h[1] * (h[1] * h[5] - h[4] * h[2])
            + h[2] * (h[1] * h[4] - h[3] * h[2])
        )
    else:
        raise RuntimeError("Unsupported number of dimensions {}.".format(len(h)))
    return det

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

        return trace, det, frobenius

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
        frobenius = np.sqrt(
            np.abs(h[0]) ** 2
            + 2 * np.abs(h[1]) ** 2
            + 2 * np.abs(h[2]) ** 2
            + np.abs(h[3]) ** 2
            + 2 * np.abs(h[4]) ** 2
            + np.abs(h[5]) ** 2
        )

        return trace, sec, det, frobenius

    else:
        raise RuntimeError("Unsupported number of dimensions {}.".format(len(h)))

def hessian_trace(h):
    """
    Trace in the hessian filter band
    Laplacian
    """
    if len(h) == 3:
        trace = h[0] + h[2]
    elif len(h) == 6:
        trace = h[0] + h[3] + h[5]
    else:
        raise RuntimeError("Unsupported number of dimensions {}.".format(len(h)))
    return trace

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

    Hp = R * H * R'
    """
    R = euler2mat(z)[:2, :2]

    H = np.zeros([2, 2, dxx.size], dtype=dxx.dtype)
    H[0, 0, :] = dxx.ravel()
    H[0, 1, :] = dxy.ravel()
    H[1, 0, :] = H[0, 1, :]
    H[1, 1, :] = dyy.ravel()

    #     Hp = np.zeros(H.shape, dtype=H.dtype)
    #     for n in range(H.shape[2]):
    #         Hp[:,:,n] = R @ H[:,:,n] @ R.transpose()
    Hp = np.dot(np.dot(R, H).transpose(2, 0, 1), R.transpose()).transpose(1, 2, 0)

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


def dog(d, order, scale, w=None, sigma=0.001, band_pass=False):
    """Gaussian derivative operator with regularization
    """

    # Band pass
    #   call gauss twice and subtract
    if band_pass:
        a = dog(d, w, sigma, scale=scale)
        b = dog(d, w, sigma, scale=scale + 1)
        return a - b

    # Low pass
    if order == 0:
        if w is not None:
            a = gaussian(w * d, scale)
            b = _pinv(gaussian(w, scale), sigma)
            return a * b
        else:
            return gaussian(d, scale=scale)

    elif order == 1:
        if w is not None:
            a = gradient(w * d, scale)
            b = _pinv(gaussian(w, scale), sigma)
            a = [x * b for x in a]
            return a
        else:
            return gradient(d, scale=scale)

    elif order == 2:
        if w is not None:
            a = hessian(w * d, scale)
            b = _pinv(gaussian(w, scale), sigma)
            a = [x * b for x in a]
            return a
        else:
            return hessian(d, scale=scale)

    else:
        raise RuntimeError(
            (
                f"Unsupported gaussian derivative order {order}."
                f"We only supports derivatives up to order 2."
            )
        )


def dog_rotational_invariants(f, order=0):
    r"""Rotational invariants of gaussian derivative operator

    returns a list

    Principal invariants and Frobenius norm
    See https://en.wikipedia.org/wiki/Invariants_of_tensors

    Gaussian (Order 0):
        2 rotational invariant: G0, \|G0\|
    Gradient (Order 1):
        1 rotational invariant: \|G1\|
    Hessian (Order 2):
        in 2D, 3 rotational invariants: Tr(G2), Det(G2), \|G2\|
        in 3D, 4 rotational invariants: Tr(G2), Det(G2), I2(G2), \|G2\|
    """

    if order == 0:
        rot = [f, np.abs(f)]
    elif order == 1:
        rot = [gradient_amplitude(f)]
    elif order == 2:
        rot = list(hessian_rot(f))
    else:
        raise RuntimeError(
            f"Unsupported gaussian derivative order {order}."
            f"We only support derivatives of order <=2."
        )

    return rot
