import numpy as np
import peakutils
import sys
from scipy.sparse import spdiags
from scipy.sparse.linalg import spsolve


def generate_histogram(gray_img):

    hist,bins = np.histogram(gray_img.flatten(),256,[0,1])

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
    # print maximas
    # print minimas
    # print maxima_indexes
    # print maxima_values
    # print threshold
    # print pixels
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


def enhance_sidelit_face(face_image,hist_bin_mask):
    d, m, b = detect_modes(hist_bin_mask[0], hist_bin_mask[1], 0.05)
    print (d, m, b)
    if d == -1 and m == -1 and b == -1:
        return face_image
    skinMask = hist_bin_mask[2]
    mask_A = np.ones(face_image.shape)
    for i in range(face_image.shape[0]):
        for j in range(face_image.shape[1]):
            if face_image[i][j] < m and skinMask[i][j] > 0:
                mask_A[i][j] = (b-d)*1.0/(m-d)
    # Apply edge aware constraint propagation to A
    mask_A =  EACP(mask_A,face_image)
    sidelit_corrected = face_image * mask_A
    return sidelit_corrected


def enhance_underexposed(face_image,hist_bin_mask):

    skinMask = hist_bin_mask[2]
    hist = hist_bin_mask[0]
    hist_cdf = hist.cumsum()
    total_skin_pixels = hist_cdf[len(hist_cdf)-1]
    percentile75 = 3 * total_skin_pixels/4
    pValue = 0
    for i in range(len(hist_cdf)):
        if(hist_cdf[i] < percentile75):
            continue
        pValue = hist_bin_mask[1][i]
        break

    pValue = int(pValue*255)
    print pValue
    if pValue >120:
        return face_image
    fValue = (120+pValue)*1.0/(2*pValue)
    fValue = fValue/255
    mask_A = np.full(face_image.shape,fValue)

    mask_A = EACP(mask_A, face_image)
    exposure_corrected = face_image * mask_A
    return exposure_corrected


def EACP(G, I, W=None, lambda_=0.2, alpha=0.3, eps=1e-4):
    """
    Edge-aware constraint propagation
    From "Interactive Local Adjustment of Tonal Values"[LFUS06]
    ARGs:
    -----
    G(A): will be g(x) in 3.2 of LFUS06, desired result.
    I: will be transformed to L (log luminance channel)
    W: float,(0-1) will be flattened to w, specifies a weight for each constrained pixel
    """
    if G.shape != I.shape:
        raise ValueError('A and I are not in the same size')
    if W == None:
        W = np.ones(G.shape)
    # L = np.log(I+eps) # avoid log of 0
    L = I
    g = G.flatten(1)
    w = W.flatten(1)
    s = L.shape

    k = np.prod(s)
    # L_i - L_j along y axis
    dy = np.diff(L, 1, 0)
    dy = -lambda_ / (np.absolute(dy) ** alpha + eps)
    dy = np.vstack((dy, np.zeros(s[1], )))
    dy = dy.flatten(1)
    # L_i - L_j along x axis
    dx = np.diff(L, 1, 1)
    dx = -lambda_ / (np.absolute(dx) ** alpha + eps)
    dx = np.hstack((dx, np.zeros(s[0], )[:, np.newaxis]))
    dx = dx.flatten(1)
    # A case: j \in N_4(i)  (neighbors of diagonal line)
    a = spdiags(np.vstack((dx, dy)), [-s[0], -1], k, k)
    # A case: i=j   (diagonal line)
    d = w - (dx + np.roll(dx, s[0]) + dy + np.roll(dy, 1))
    a = a + a.T + spdiags(d, 0, k, k) # A: put together
    f = spsolve(a, w*g).reshape(s[::-1]) # slove Af  =  b =w*g and restore 2d
    A = np.rollaxis(f,1)
    # A = np.clip( _out*255.0, 0, 255).astype('uint8')
    return A