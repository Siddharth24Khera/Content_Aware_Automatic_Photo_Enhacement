import numpy as np
import matplotlib.pyplot as plt
import peakutils
import sys


def generate_histogram(gray_img):

    plt.figure()
    plt.subplot(211)
    hist, bins, patches = plt.hist(gray_img.flatten(), 256, [0, 1])
    plt.title("orig_his")

    len_of_hist = hist.shape[0]
    smooth_hist = np.zeros(len_of_hist)
    hist[0] = 0
    smooth_hist[0] = hist[0]
    smooth_hist[len_of_hist - 1] = hist[len_of_hist - 1]
    smooth_hist[1] = (hist[0] + 2*hist[1])/3
    smooth_hist[len_of_hist - 2] = (2*hist[len_of_hist - 2] + hist[len_of_hist - 1])/3

    for i in range(2, len(bins)-3):
        smooth_hist[i] = (hist[i-2] + 2*hist[i-1] + 4*hist[i] + 2*hist[i+1] + hist[i+2])/10
    bin_centers = bins[:-1] + 0.5 * (bins[1:] - bins[:-1])
    plt.subplot(212)
    plt.plot(bin_centers,smooth_hist)
    plt.title("smooth_his")
    return smooth_hist,bin_centers


def detect_modes(bin, hist, thresh_value):
    maximas = []
    minimas = []
    maxima_values = []
    cb = np.array(bin)
    maxima_indexes = peakutils.indexes(cb, thres=0., min_dist=50)
    for num in maxima_indexes:
        maxima_values.append(bin[num])
    pixels = np.sum(bin)
    threshold = thresh_value * (pixels)
    for num in maxima_indexes:
        if num > threshold:
            maximas.append(num)

    if (len(maximas) >= 2):
        maxima_indexes = maximas

    maximas = []
    temp_val = -1
    temp_index = -1
    for num in maxima_indexes:
        if (bin[num] > temp_val):
            temp_val = bin[num]
            temp_index = num
    maximas.append(temp_index)
    temp = temp_index

    temp_val = -1
    temp_index = -1
    for num in maxima_indexes:
        if (bin[num] > temp_val and num != temp):
            temp_val = bin[num]
            temp_index = num
    maximas.append(temp_index)

    if (maximas[0] > maximas[1]):
        temp = maximas[0]
        maximas[0] = maximas[1]
        maximas[1] = temp

    for i in range(0, len(maximas) - 1):
        max1 = maximas[i]
        max2 = maximas[i + 1]
        min = sys.maxint
        min_index = -1
        for j in range(max1, max2):
            if bin[j] < min:
                min = bin[j]
                min_index = j
        minimas.append(min_index)
    print maximas
    print minimas
    print maxima_indexes
    print maxima_values
    print threshold
    print pixels
    for i in range(0, len(maximas) - 1):
        for j in range(0, len(minimas)):
            if ((maximas[i] < minimas[j]) and (maximas[i + 1] > minimas[j]) and bin[minimas[j]] < (
                0.8 * (bin[maximas[i]])) and bin[minimas[j]] < (0.8 * (bin[maximas[i + 1]]))):
                # print hist[maximas[i]]
                # print hist[minimas[j]]
                # print hist[maximas[i+1]]
                # print maximas[i]
                # print minimas[j]
                # print maximas[i+1]
                return hist[maximas[i]], hist[minimas[j]], hist[maximas[i + 1]]
    return -1, -1, -1


def enhance_sidelit_face(face_image,hist_bin,d,m,b):
    mask_A = np.ones(face_image.shape)
    for i in range(face_image.shape[0]):
        for j in range(face_image.shape[1]):
            if face_image[i][j] < m:
                mask_A[i][j] = (b-d)*1.0/(m-d)
    # Apply edge aware constraint propagation to A
    face_image = face_image * mask_A