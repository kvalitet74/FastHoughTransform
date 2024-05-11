"""Test suite for hough.py."""
import os
import sys
sys.path.append(os.path.abspath('.') + "/src")
print(sys.path)

import numpy as np
from PIL import Image
import pytest

from hough import HoughTransform


def test_accepts_only_grayscale():
    """It raises error if ndarray has more than 2 dimensions."""
    with pytest.raises(ValueError):
        HoughTransform(np.zeros((2, 2, 2)))  # pass 3d array


def test_accepts_path_to_image():
    """It can handle paths to images."""
    try:
        HoughTransform("samples/clean/cross.jpg")
    except FileNotFoundError:
        pytest.fail("FileNotFoundError")


def test_accepts_pil_image():
    """It can handle PIL.Image.Image objects."""
    image = Image.open("samples/clean/cross.jpg")
    try:
        HoughTransform(image)
    except:
        pytest.fail("Unexpected error occurred!")


def test_raises_on_unknown_type():
    """It raises error if the type is unknown."""
    with pytest.raises(ValueError):
        HoughTransform(1)  # pass int


def test_raises_on_unknown_edge_method():
    """It raises error if provided edge computing method is unknown."""
    ht = HoughTransform(np.zeros((2, 2)))
    with pytest.raises(ValueError):
        ht.compute_edges_(method='unknown')


def test_transform_returns_array():
    """It returns ndarray."""
    ht = HoughTransform("samples/clean/cross.jpg")
    assert isinstance(ht.transform(ht.grayscale_), np.ndarray)


def test_1_pixel_doesnt_lead_to_div_by_zero():
    """It doesn't divide by zero even if image is only 1 pixel."""
    ht = HoughTransform(np.zeros((1, 1)))
    try:
        ht.transform(ht.grayscale_)
    except ZeroDivisionError:
        pytest.fail("ZeroDivisionError in transform")


# THIS TEST FAILS - it produces nan because of division by (max - min)
def test_1_pixel_doesnt_produce_nan():
    """It doesn't produce nan if image is only 1 pixel."""
    ht = HoughTransform(np.zeros((1, 1)))
    assert not np.isnan(ht.transform(ht.grayscale_)).any()


def test_no_lines_on_image_no_crash():
    """It doesn't crash if there are no lines on the image."""
    ht = HoughTransform("samples/clean/cross.jpg")  # it cannot find any lines in this example
    try:
        ht.transform(ht.grayscale_)
    except:
        pytest.fail("Failed because no lines detected")


# THIS TEST FAILS - it finds lines on an image where there is none, obviously
def test_no_lines_prints_0():
    """It returns and prints 0 if there are no lines on the image."""
    image = np.arange(32 * 32).reshape((32, 32)) / (32 * 32)
    ht = HoughTransform(image)
    assert len(ht.find_lines()) == 0


# TESTS BELOW ARE TO HIT 100% COVERAGE
# as far as I'm concerned, statements they hit are never going to be used
def test_draw_lines_returns_pil():
    """It returns PIL.Image.Image object."""
    ht = HoughTransform("samples/clean/cross.jpg")
    assert isinstance(ht.draw_lines(), Image.Image)


def test_no_image_no_crash():
    """It doesn't crash if there is no image provided."""
    ht = HoughTransform("samples/clean/cross.jpg")
    try:
        ht.transform()
    except:
        pytest.fail("Failed because there is no image")


def test_lazy_lines_uses_rotated_images():
    """It stores rotated images and uses them the second time."""
    ht = HoughTransform("samples/clean/cross.jpg")
    ht.find_lines()
    ht.find_lines()  # second time it should use previous rotations and hit corresponding if statement
    assert bool(ht.lazy_transforms)


def test_find_lines_uses_n_max():
    """It uses only {max_lines} lines, if this argument is provided."""
    ht = HoughTransform("samples/clean/line12.jpg")
    n_max = 3
    assert len(ht.find_lines(max_lines=n_max)) == n_max


def test_draw_lines_raises_error_if_called_before_find():
    """It raises error if there is image but no lines."""
    image = np.arange(32 * 32).reshape((32, 32)) / (32 * 32)
    ht = HoughTransform(image)
    with pytest.raises(ValueError):
        ht.draw_lines(image)


def test_draw_lines_accepts_ndarrays():
    """It can handle ndarrays."""
    image = np.arange(32 * 32).reshape((32, 32)) / (32 * 32)
    ht = HoughTransform(image)
    lines = ht.find_lines()
    assert isinstance(ht.draw_lines(image, lines), Image.Image)


def test_rotate_line_90_uses_shape():
    """It uses image shape if it wasn't explicitly provided."""
    ht = HoughTransform("samples/clean/line12.jpg")
    # sample line, taken from hough transform of line12.jpg
    line = [(0, 22.775824131830106, 48, 66.30395979722167), 1.0407061653877145]
    try:
        ht.rotate_line_90_(line)
    except:
        pytest.fail("Failed because shape was not given")


# TEST IS DEPRECATED BECAUSE 'sobel' IS THE DEFAULT METHOD, NO NEED TO CHECK
# def test_uses_provided_method_for_computing_edges():
#     """It uses provided method/"""
#     ht = HoughTransform("samples/clean/line12.jpg")
#     try:
#         ht.compute_edges_(method='sobel')
#     except ValueError:
#         pytest.fail("Failed because sobel is an unknown method")


def test_uses_class_variable_for_computing_edges():
    """It uses class variable if given method is None or False."""
    ht = HoughTransform("samples/clean/line12.jpg", "sobel")
    try:
        ht.compute_edges_(method=None)
    except ValueError:
        pytest.fail("Failed because class variable is not used")
