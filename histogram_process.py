import numpy as np
import matplotlib.pyplot as plt


def generate_histogram(gray_img):

    plt.figure()
    plt.subplot(211)
    hist, bins, patches = plt.hist(gray_img.flatten(), 256, [0, 1])
    plt.title("orig_his")

    len_of_hist = hist.shape[0]
    smooth_hist = np.zeros(len_of_hist)
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

def detect_modes(hist,bin):
    # returns d m b
    pass
