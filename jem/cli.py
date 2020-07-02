# -*- coding: utf-8 -*-

import click
import nibabel
import numpy as np
from .signal_stats import global_scale
from .features import (
    laplacian_pyramid,
    local_scale_normalization,
    local_contrast_normalization,
    front_end_features,
    fef_rotational_invariants,
    NUM_SCALES,
    NORMALIZATION_SCALE,
    gaussian_pyramid_features
)


def _check_image_compatibility(images):
    im_shape = images[0].shape
    im_affine = images[0].affine
    for im in images[1:]:
        assert im.shape == im_shape, "Image shape mismatch: {} and {} differ.".format(
            im.get_filename(), images[0].get_filename()
        )
        assert np.all(
            im.affine == im_affine
        ), "Image affine mismatch: {} and {} differ.".format(
            im.get_filename(), images[0].get_filename()
        )


@click.command()
@click.option(
    "--output",
    type=click.STRING,
    default="out.nii",
    help="Output filename for the contrast normalized image.",
)
@click.option(
    "--scale", type=click.INT, default=NORMALIZATION_SCALE, help="Scale for smoothing."
)
@click.argument("input_image", type=click.STRING)
def contrast_normalization(input_image, scale, output):
    """Automatic local contrast normalization.
    """

    # open the images
    im = nibabel.load(input_image)

    # load the image data and the coil correction data and apply
    data = im.get_data().astype(np.float32)
    f, w, sigma = global_scale(data)
    lc = local_contrast_normalization(f, w, sigma, scale=scale)

    # write out the result as floating point and preserve the header
    out_im = type(im)(lc.astype(np.float32), affine=None, header=im.header)
    out_im.set_data_dtype(np.float32)
    out_im.to_filename(output)

    click.echo("Wrote coil contrast normalized image to {}.".format(output))


@click.command()
@click.option(
    "--output",
    type=click.STRING,
    default="out.nii",
    help="Output filename for the corrected image.",
)
@click.option(
    "--scale", type=click.INT, default=NORMALIZATION_SCALE, help="Scale for smoothing."
)
@click.argument("input_image", type=click.STRING)
def coil_correction(input_image, scale, output):
    """Compute and apply receive coil intensity correction."""

    # open the images
    im = nibabel.load(input_image)

    # load the image data and the coil correction data and apply
    data = im.get_data().astype(np.float32)
    f, w, sigma = global_scale(data)
    data_corr = local_scale_normalization(f, normalization_scale=scale, w=w, sigma=sigma)

    # write out the result as floating point and preserve the header
    out_im = type(im)(data_corr.astype(np.float32), affine=None, header=im.header)
    out_im.set_data_dtype(np.float32)
    out_im.to_filename(output)

    click.echo("Wrote coil intensity corrected image to {}.".format(output))


@click.command(name=laplacian_pyramid)
@click.option(
    "--output",
    type=click.STRING,
    default="out.nii",
    help="Output filename for the laplacian pyramid image.",
)
@click.argument("input_image", type=click.STRING)
@click.option(
    "--num_scales", type=click.INT, default=NUM_SCALES, help="number of spatial scales"
)
@click.option(
    "--normalization_scale",
    type=click.INT,
    default=NORMALIZATION_SCALE,
    help="scale for input gain control",
)
def compute_laplacian_pyramid(input_image, num_scales, normalization_scale, output):
    """Compute rotationally invariant features."""

    click.echo(f"Compute rotationally invariant features for {input_image}.")

    # open the images
    im = nibabel.load(input_image)
    data = im.get_data().astype(np.float32)

    # Global scaling, signal likelihood, noise level
    f, w, sigma = global_scale(data)

    # Local scale normalization
    f_c = local_scale_normalization(
        f, w=w, sigma=sigma, normalization_scale=normalization_scale
    )

    # compute the laplacian pyramid
    lpyr = laplacian_pyramid(f_c, w=w, sigma=sigma, num_scales=num_scales)

    # Smush into a single array and convert to single precision
    # with pyramed level as the fourth dimension
    lpyr = np.stack(lpyr, axis=-1).astype(np.float32)

    # write out the result in the same format and preserve the header
    out_im = type(im)(lpyr, affine=None, header=im.header)
    out_im.set_data_dtype(np.float32)
    out_im.to_filename(output)

    click.echo(f"Wrote Laplacian Pyramid to {output}.")


@click.command()
@click.option(
    "--output",
    type=click.STRING,
    default="out.nii",
    help="Output filename for the features image.",
)
@click.argument("input_image", type=click.STRING)
@click.option(
    "--num_scales", type=click.INT, default=NUM_SCALES, help="number of spatial scales"
)
@click.option(
    "--normalization_scale",
    type=click.INT,
    default=NORMALIZATION_SCALE,
    help="scale for input gain control",
)
@click.option("--lowpass/--no-lowpass", default=False)
def riff(input_image, num_scales, normalization_scale, output, lowpass):
    """Compute rotationally invariant bandpass features."""

    click.echo(f"Compute rotationally invariant features for {input_image}.")

    # open the images
    im = nibabel.load(input_image)
    data = im.get_data().astype(np.float32)

    # Global scaling, signal likelihood, noise level
    f, w, sigma = global_scale(data)

    # Local scale normalization
    f = local_scale_normalization(
        f, w=w, sigma=sigma, normalization_scale=normalization_scale
    )

    # compute the frond end features
    fef = front_end_features(f, w=w, sigma=sigma, num_scales=num_scales)

    # rotational invariants
    fef = fef_rotational_invariants(fef, inplace=True)

    # convert to single precision and smush into a single array
    # with feature number as the fourth dimension
    # level changes fastest on disk, then subband, then order
    # HP,Zero,LP,|HP|,|Zero|,|LP|,|Two|,I1(Two),I2(Two)

    feats = []
    feats.append(fef["high_pass"][0].astype(np.float32))
    for n in range(num_scales):
        feats.append(fef["zero"][n][0].astype(np.float32))
    if lowpass:
        feats.append(fef["low_pass"][0].astype(np.float32))
    feats.append(fef["high_pass"][1].astype(np.float32))
    for n in range(num_scales):
        feats.append(fef["zero"][n][1].astype(np.float32))

    for n in range(num_scales):
        feats.append(fef["one"][n][0].astype(np.float32))

    M = len(fef["two"][0])
    for m in range(M):
        for n in range(num_scales):
            feats.append(fef["two"][n][m].astype(np.float32))

    feats = np.stack(feats, axis=-1)

    # write out the result in the same format and preserve the header
    out_im = type(im)(feats, affine=None, header=im.header)
    out_im.set_data_dtype(np.float32)
    out_im.to_filename(output)

    click.echo(f"Wrote rotationally invariant bandpass features to {output}.")

@click.command()
@click.option(
    "--output",
    type=click.STRING,
    default="out.nii",
    help="Output filename for the features image.",
)
@click.argument("input_image", type=click.STRING)
@click.option(
    "--num_scales", type=click.INT, default=NUM_SCALES, help="number of spatial scales"
)
@click.option(
    "--normalization_scale",
    type=click.INT,
    default=NORMALIZATION_SCALE,
    help="scale for input gain control",
)
def compute_features(input_image, num_scales, normalization_scale, output):
    """Compute rotationally invariant features."""

    click.echo(f"Compute rotationally invariant features for {input_image}.")

    # open the images
    im = nibabel.load(input_image)
    data = im.get_data().astype(np.float32)

    # Global scaling, signal likelihood, noise level
    f, w, sigma = global_scale(data)

    # Local scale normalization
    f = local_scale_normalization(
        f, w=w, sigma=sigma, normalization_scale=normalization_scale
    )

    # compute features
    feats = gaussian_pyramid_features(f, num_scales=num_scales, w=w, sigma=sigma)
    feats = np.stack(feats, axis=-1)

    # write out the result in the same format and preserve the header
    out_im = type(im)(feats, affine=None, header=im.header)
    out_im.set_data_dtype(np.float32)
    out_im.to_filename(output)
