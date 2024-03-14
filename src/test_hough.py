import numpy as np
import pytest
from skimage.transform import hough_line_peaks, hough_line
from skimage.io import imread
from PIL import Image, ImageDraw
import cv2 as cv

import hough

def hough_by_kirill(file):
    ht = hough.HoughTransform(file)
    lines = ht.find_lines()
    image = ht.draw_lines(lines=lines, color='red', thickness=5)
    image.save(file[:-4] + "_k.jpg")
    return lines


def hough_by_opencv(file):
    src = cv.imread(cv.samples.findFile(file), cv.IMREAD_GRAYSCALE)
    if src is None:
        print('Error opening image!')
        return -1

    dst = cv.Canny(src, 50, 200, None, 3)
    lines = cv.HoughLinesP(dst, 1, np.pi / 180, 50, None, 50, 10)

    lines_to_return = []
    if lines is not None:
        for index in lines:
            l = index[0]
            lines_to_return += [[l[0], l[1], l[2], l[3]]]

    return lines_to_return


def draw_cv_lines(file):
    lines = hough_by_opencv(file)
    image = Image.open(file)
    width, height = image.size
    image_ = image.copy()
    draw = ImageDraw.Draw(image_)
    new_lines = []
    for line in lines:
        x1, y1, x2, y2 = line
        k = (y2 - y1) / (x2 - x1) if (x2 - x1) != 0 else 100000
        if k == 0:
            x1 = 0
            x2 = width
        elif abs(k) >= 100000:
            y1 = 0
            y2 = height
        else:
            y_bound1 = y2 - k * x2
            y_bound2 = y2 + k * (width - x2)
            x_bound1 = x2 - y2 / k
            x_bound2 = x2 + (height - y2) / k
            if 0 > y_bound1:
                y1 = 0
                x1 = x_bound1
            else:
                x1 = 0
                y1 = y_bound1
            if height < y_bound2:
                y2 = height
                x2 = x_bound2
            else:
                x2 = width
                y2 = y_bound2
        line = [x1, y1, x2, y2]
        new_lines += [line]
        draw.line(line, fill='red', width=5)
    image_.save(file[:-4] + "_cv.jpg")
    return new_lines

@pytest.mark.parametrize("path", ["go.jpg", "lines_6.jpg", "lines_5.jpg", "lines_4.jpg", "lines_3.jpg", "lines_2.jpg"])
def test_hough_transform(path):
    path = "../samples/clean/" + path
    lines_k = hough_by_kirill(path)
    lines_cv = draw_cv_lines(path)

    equal = 0
    for line in lines_k:

        if line in lines_cv:
            equal += 1

    assert equal >= 3


# if __name__ == "__main__":
#     path = "lines_2.jpg"
#     lines_k = hough_by_kirill(path)
#     lines_cv = draw_cv_lines(path)
#
#     print("lines found by Kirill's implementation")
#     for i_, line_ in enumerate(lines_k):
#         x1_, y1_, x2_, y2_ = line_
#         print(f"{i_}.\tp1=({x1_:0.2f}, {y1_:0.2f}),\tp2=({x2_:0.2f}, {y2_:0.2f})")
#     print("lines found by OpenCV implementation")
#     for i_, line_ in enumerate(lines_cv):
#         x1_, y1_, x2_, y2_ = line_
#         print(f"{i_}.\tp1=({x1_:0.2f}, {y1_:0.2f}),\tp2=({x2_:0.2f}, {y2_:0.2f})")

