import cv2
import numpy
from scipy.sparse import spdiags
from scipy.sparse.linalg import spsolve
import os

IMG_DIR = '../resources/images/'


def wls_filter(image_orig, lambda_=0.4, alpha=1.2, small_eps=1e-4):
    """
    ARGs:
    -----
    image: 0-255, uint8, single channel (e.g. grayscale or single L)
    lambda_:
    alpha:
    RETURN:
    -----
    out: base, 0-1, float
    detail: detail, 0-1, float
    """
    image = image_orig.astype(numpy.float)/255.0
    s = image.shape
    k = numpy.prod(s)

    dy = numpy.diff(image, 1, 0)

    dy = -lambda_ / (numpy.absolute(dy) ** alpha + small_eps)
    dy = numpy.vstack((dy, numpy.zeros((1,s[1]))))
    dy = dy.flatten(1)

    dx = numpy.diff(image, 1, 1)
    dx = -lambda_ / (numpy.absolute(dx) ** alpha + small_eps)
    dx = numpy.hstack((dx, numpy.zeros((s[0],1))))
    dx = dx.flatten(1)

    a = spdiags(numpy.vstack((dx, dy)), [-s[0], -1], k, k)

    d = 1 - (dx + numpy.roll(dx, s[0]) + dy + numpy.roll(dy, 1))
    a = a + a.T + spdiags(d, 0, k, k)
    _out = spsolve(a, image.flatten(1)).reshape(s[::-1])
    out = numpy.rollaxis(_out,1)
    detail = image - out
    return out, detail #float


if __name__ == '__main__':
    IMG_DIR = './Images'
    IMG_NAME = 'Awnings.jpg'
    image_path = os.path.join(IMG_DIR,IMG_NAME)
    image = cv2.imread(image_path)
    cv2.namedWindow("Original")
    cv2.namedWindow("Base")
    cv2.namedWindow("Detail")
    base,detail = wls_filter(image)
    cv2.imshow("Original",image)
    cv2.imshow("Base",base)
    cv2.imshow("Detail", detail)
    cv2.waitKey(0)