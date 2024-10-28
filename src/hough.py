"""
An implementation of HoughTransform class to compute "fast Hough transform".

There are methods to compute the transform, strictly find lines and draw them on the image.
"""

import numpy as np
import pathlib
from Pillow import Image as PIL_Image
from Pillow import Image, ImageDraw
import cv2 as cv
from scipy import ndimage


class HoughTransform:
    """Base class of hough transform. Performs empty transform."""

    def __init__(
        self,
        image: np.ndarray | PIL_Image | str | pathlib.PosixPath | pathlib.WindowsPath,
        compute_edges: str | None = "laplacian",
    ) -> None:
        """
        Initializes with numpy image in grayscale or path to image.

        Parameters:
        1) image:
            - numpy image in grayscale with shape=(h, w)
            - or PIL.Image in any format
            - or path to image in PIL accessible format
              (PNG, JPEG, TIFF, etc.).
              It copies the argument.
        2) compute_edges:
            method of edge selection for given image.
            Possible values:
            - 'sobel': Sobel filter
            - 'laplacian': Laplacian filter
            - None or False: considers that the 'image' is already
              an edge map. Does nothing.
        """
        if isinstance(image, np.ndarray):
            if len(image.shape) != 2:
                raise ValueError(
                    "If given image is numpy.ndarray it must be IN GRAYSCALE \
                    and have shape (h, w)."
                )
            self.grayscale_ = np.array(image).copy()
            self.image_ = self.grayscale_
        else:
            # Load image from pathlib path or str
            if (
                isinstance(image, str)
                or isinstance(image, pathlib.PosixPath)
                or isinstance(image, pathlib.WindowsPath)
            ):
                image = Image.open(image)
                # Now image is PIL.Image
            if isinstance(image, PIL_Image):
                # Save as original image & grayscale
                self.image_ = image.copy()
                self.grayscale_ = np.array(self.image_.convert(mode="L"))
            else:
                raise ValueError(
                    f"Unknown type of image: '{type(image)}'. Requires to be \
                    numpy.ndarray, PIL.Image or str"
                )

        # Compute edges
        self.grad_ = self.compute_edges_(method=compute_edges)
        self.shape = self.grayscale_.shape
        # Initialize buffer for lazy computations
        self.lazy_transforms: dict = {}

    ########################
    # --- Main methods --- #
    ########################

    def transform(self, image: np.ndarray | None = None) -> np.ndarray:
        """
        Fast Hough Transform for HORIZONTAL DECREASING line detection.

        Parameters:
            1) image: it performs transform on this image.
            By default (if None) it is gradient oughTransform.grad_
            from initializing.
        Returns:
            1) transform: np.ndarray in [0, 1] of shape=HoughTransform.shape
            transform[x, y] means accumulated brightness in [0, 1] along a line
            with coordinates [(x, 0), (x + y, HoughTransform.shape[1])]
        """
        if image is None:
            image = self.grad_

        imgh, imgw = image.shape
        hough_levels = imgw.bit_length() + 1
        h = imgh
        w = 2 ** (hough_levels - int(imgw == (2**hough_levels)))

        # Initialize 0-th level
        hough = np.zeros((h, w))
        hough[:imgh, :imgw] = image

        for level in range(1, hough_levels):
            bs = 2**level  # block size
            # Computing hough for current level
            new_hough = np.zeros((h, w))
            for x in range(bs):
                # Cyclic shift for vectors on the right
                shift = -((x + 1) // 2)
                lpos = x // 2
                rpos = (bs + x) // 2
                left_values = hough[:, lpos::bs]
                right_values = np.roll(hough[:, rpos::bs], shift, axis=0)
                new_hough[:, x::bs] = left_values + right_values
            hough = new_hough
        result = hough[:imgh, :imgw]
        # MinMax answer
        if result.max() > result.min():
            result = (result - result.min()) / (result.max() - result.min())
        else:
            result.fill(0)
        return result

    def find_lines(
        self,
        horizontal: bool = True,
        vertical: bool = True,
        quantile: float = 0.99,
        min_cluster_size: int | float = 0.001,
        max_lines: int | None = None,
    ) -> list[tuple]:
        """
        Finds lines on image.

        Parameters:
            1) horizontal: Flag to find horizontal lines
            2) vertical: Flag to find vertical lines
            3) quantile: Quantile of the brightest pixels
               in hough transforms to consider them lines.
            4) min_cluster_size: int. If >= 1, it is min number of pixels
               in a cluster to select line in it.
               If in [0, 1) it is considered as
               min percentage of pixels in the image.
            5) max_lines: limit on resulted lines,
               so it returns <= 'max_lines' most obvious lines.
            If None, unlimited.
        Returns: list of tuples (x1, y1, x2, y2) of found lines.
        """

        # To find all lines we need to rotate the image
        # and apply transform (to get lines of any direction)
        def lazy_lines(name: str, rotate_times: int) -> list[tuple]:
            if name in self.lazy_transforms:
                hough = self.lazy_transforms[name]
            else:
                hough = self.transform(np.rot90(self.grad_, rotate_times))
                # Normalize transform. It is important to prevent imbalance
                # in scores
                # with imbalance of width/height while rotating in find_lines
                hough = (hough - hough.min()) / (hough.max() - hough.min())
                # hough is done. we can save it to buffer
                self.lazy_transforms[name] = hough

            shape = (self.shape[rotate_times % 2], self.shape[(rotate_times + 1) % 2])
            hough = (hough - hough.min()) / (hough.max() - hough.min())
            # Find lines and rotate them in reverse
            lines = self.find_horizontal_down_lines_(
                hough, shape[1], quantile, min_cluster_size, max_lines
            )
            lines = [self.rotate_line_90_(line, shape, rotate_times) for line in lines]
            return lines

        # Find all possible lines
        horizontal_down = lazy_lines("horizontal_down", 0) if horizontal else []
        vertical_left = lazy_lines("vertical_left", 1) if vertical else []
        horizontal_up = lazy_lines("horizontal_up", 2) if horizontal else []
        vertical_right = lazy_lines("vertical_right", 3) if vertical else []

        lines = horizontal_down + vertical_left + horizontal_up + vertical_right

        lines.sort(key=lambda line: line[1], reverse=True)  # Sort by score
        # Remove scores
        lines = [line[0] for line in lines]
        # Restrict by max_lines
        if max_lines is not None:
            lines = lines[:max_lines]
        return lines

    def draw_lines(
        self,
        image: np.ndarray | PIL_Image | None = None,
        lines: list[tuple] | None = None,
        color: str = "red",
        thickness: int = 5,
        **find_lines_kwargs: dict,
    ) -> PIL_Image:
        """
        Draws found lines on the image.

        If argument 'image' is not set uses image from initialization.

        Parameters:
            1) image:
            - numpy image in grayscale with shape=(h, w)
            - or PIL.Image in any format
            By default (if None) it is oughTransform.image_
            from initializing.
            !! If not None, you must provide lines for this image
            because it was not initialized !!
            2) lines: list of tuples in format (x1, y1, x2, y2)
            containing two points of line.
            !! If image and lines are None it finds lines automatically
            using **find_lines_kwargs !!
            3) color: Color in PIL.ImageColor format.
            See https://pillow.readthedocs.io/en/
            stable/reference/ImageColor.html#color-names
            4) thickness: Thickness of lines in pixels.
            5) **find_lines_kwargs: if image and lines are None
            it finds lines automatically using **find_lines_kwargs
        Returns:
            1) PIL.Image image with drawn lines

        """
        if image is None and lines is None:
            lines = self.find_lines()
        elif lines is None:
            raise ValueError(
                "You must initialize new HoughTransform to \
                find lines for this image in argument."
            )
        if image is None:
            image = self.image_
        if isinstance(image, np.ndarray):
            image = Image.fromarray(image)
        # Using PIL drawer
        draw = ImageDraw.Draw(image)
        for line in lines:
            draw.line(line, fill=color, width=thickness)
        return image

    ###########################
    # --- Support methods --- #
    ###########################

    def find_horizontal_down_lines_(
        self,
        hough: np.ndarray,
        width: int,
        quantile: float,
        min_cluster_size: int | float,
        max_clusters: int | None = None,
    ) -> list[tuple]:
        """Finds only lines with positive tan(angle)."""
        # Binarize transform
        quantile_value = float(np.quantile(hough.squeeze(), quantile))
        houghmap = hough >= quantile_value
        # Now we have candidates to be lines. Clusterize them:
        clustermap, num_ids = ndimage.label(houghmap, structure=np.ones((3, 3)))
        clusters = [np.where(clustermap == v) for v in range(num_ids)]
        clusters.sort(key=lambda c: hough[c].mean(), reverse=True)
        if max_clusters is not None:
            clusters = clusters[:max_clusters]
        # Remove zero cluster
        clusters = clusters[:-1]
        # Remove noise clusters
        if min_cluster_size < 1:
            min_cluster_size = int(self.shape[0] * self.shape[1] * min_cluster_size)
        clusters = [c for c in clusters if len(c[0]) >= min_cluster_size]

        # Find centroids of clusters: lines that represent clusters.
        def get_centroid(cluster: list[tuple], power: int = 4) -> tuple:
            points_array = np.array(list(zip(*cluster)))
            centroid = np.average(points_array, weights=hough[cluster] ** power, axis=0)
            return centroid

        centroids = list(map(get_centroid, clusters))
        scores = [hough[c].mean() / quantile_value for c in clusters]

        # Finally find lines
        lines = []
        for centroid, score in zip(centroids, scores):
            x, y = centroid
            # Format: ((x1, y1, x2, y2), score)
            lines.append(((0, x, width, x + y), score))
        return lines

    def rotate_line_90_(
        self, line: tuple, shape: tuple | None = None, times: int = 1
    ) -> tuple:
        """Rotates the given line to the given angle by 90 degrees <times> times."""
        if times < 1:
            return line
        if shape is None:
            shape = self.shape
        (x1, y1, x2, y2), score = line
        h, w = shape
        line = ((h - y1, x1, h - y2, x2), score)
        return self.rotate_line_90_(line, (w, h), times - 1)

    def compute_edges_(
        self, method: str | None = "sobel", sobel_size: int = 5
    ) -> np.ndarray:
        """Computes an edge map (gradient map) of the image in grayscale."""
        if method == "laplacian":
            return np.abs(cv.Laplacian(self.grayscale_, cv.CV_64F))
        elif method == "sobel":
            return np.abs(cv.Sobel(self.grayscale_, cv.CV_64F, 1, 1, ksize=sobel_size))
        elif method is None or method is False:
            return self.grayscale_
        raise ValueError(f"Unknown edge computing method: '{method}'.")
