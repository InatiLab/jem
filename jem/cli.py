# -*- coding: utf-8 -*-

import click
import nibabel
import numpy as np
from .signal_stats import global_scale
from .features import (
    local_scale_normalization,
    local_contrast_normalization,
    riff,
    NUM_SCALES,
    NORMALIZATION_SCALE,
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
    lc = local_contrast_normalization(data, w, sigma, scale=scale)

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
    data_corr = local_scale_normalization(data, w, sigma, scale=scale)

    # write out the result as floating point and preserve the header
    out_im = type(im)(data_corr.astype(np.float32), affine=None, header=im.header)
    out_im.set_data_dtype(np.float32)
    out_im.to_filename(output)

    click.echo("Wrote coil intensity corrected image to {}.".format(output))


@click.command(name=riff)
@click.option(
    "--output",
    type=click.STRING,
    default="out.nii",
    help="Output filename for the features image.",
)
@click.argument("input_image", type=click.STRING)
@click.option(
    "--nscales", type=click.INT, default=NUM_SCALES, help="number of spatial scales"
)
@click.option(
    "--normalization_scale",
    type=click.INT,
    default=NORMALIZATION_SCALE,
    help="scale for input gain control",
)
def compute_riff(input_image, nscales, normalization_scale, output):
    """Compute rotationally invariant features."""

    click.echo(f"Compute rotationally invariant features for {input_image}.")

    # open the images
    im = nibabel.load(input_image)
    data = im.get_data().astype(np.float32)

    # compute the rotationally invariant features
    rot = riff(data, num_scales=nscales, normalization_scale=normalization_scale)

    # convert to single precision and smush into a single array
    # with feature number as the fourth dimension
    feats = []
    for scale in rot:
        for feat in scale["bandpass"] + scale["gradient"] + scale["hessian"]:
            feats.append(feat.astype(np.float32))
    feats = np.stack(feats, axis=-1)

    # write out the result in the same format and preserve the header
    out_im = type(im)(feats, affine=None, header=im.header)
    out_im.set_data_dtype(np.float32)
    out_im.to_filename(output)

    click.echo(f"Wrote rotationally invariant features to {output}.")
