# -*- coding: utf-8 -*-

import click
import nibabel
import numpy as np
from .signal_stats import signal_likelihood
from .features import input_normalization, riff, NUM_SCALES, NORMALIZATION_SCALE


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
    "--threshold", type=click.FLOAT, default=0.7, help="Signal likelihood threshold."
)
@click.option(
    "--output",
    type=click.STRING,
    default="signal_mask.nii",
    help="Output filename for the coil signal mask.",
)
@click.argument("input_images", nargs=-1, type=click.STRING)
def signal_mask(input_images, threshold, output):
    """Compute signal mask from one or more images."""

    click.echo("Computing signal mask...")

    # open the images
    images = [nibabel.load(x) for x in input_images]

    # all other images must have matching orientation/dimensions/etc.
    _check_image_compatibility(images)

    # sum the input inputs
    im_shape = images[0].shape
    a = np.zeros(im_shape, dtype="float32")
    for im in images:
        a += im.get_data()

    # compute the signal likelihood and threshold
    mask = signal_likelihood(a) > threshold

    # write out the result in the same format and preserve the header
    out_image = type(images[0])(
        mask.astype(np.float32), affine=None, header=images[0].header
    )

    out_image.to_filename(output)

    click.echo("Wrote signal mask to {}.".format(output))


@click.command()
@click.option(
    "--output",
    type=click.STRING,
    default="out.nii",
    help="Output filename for the corrected image.",
)
@click.option(
    "--scale",
    type=click.INT,
    default=NORMALIZATION_SCALE,
    help="Scale for high pass filtering and smoothing.",
)
@click.argument("input_image", type=click.STRING)
def coil_correction(input_image, scale, output):
    """Compute and apply receive coil intensity correction."""

    # open the images
    im = nibabel.load(input_image)

    # load the image data and the coil correction data and apply
    data = im.get_data().astype("float32")
    data_corr = input_normalization(data, scale=scale)

    # write out the result as floating point and preserve the header
    out_image = type(im)(data_corr, affine=None, header=im.header)
    out_image.to_filename(output)

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

    # compute the textures
    t, names = riff(
        im.get_data().astype("float32"),
        nscales=nscales,
        normalization_scale=normalization_scale,
    )
    nfeats = len(t)
    out = np.zeros([*t[0].shape, nfeats], t[0].dtype)
    for f in range(nfeats):
        out[..., f] = t[f]

    # write out the result in the same format and preserve the header
    out_image = type(im)(out, affine=None, header=im.header)
    out_image.to_filename(output)

    click.echo(f"Wrote rotationally invariant features to {output}.")
