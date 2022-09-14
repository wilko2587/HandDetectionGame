from PIL import Image
import numpy as np
import matplotlib.pyplot as plt


def read_img(filename):
    img = Image.open(filename)
    bw_img = np.mean(img, axis=2).astype('int')
    return bw_img


def get_histogram(img, normalise=True, cumilative=True):
    hist = []
    for r in range(255):
        hist.append(np.sum(img == r))
    # Now cumilative + normalisation
    if normalise:
        norm = sum(hist)
    else:
        norm = 1

    for r in range(255):
        val = float(hist[r]) / norm
        if cumilative:
            try:
                val += hist[r - 1]
            except(IndexError):
                pass

        hist[r] = val

    return hist


def histogram_equalisation(img, L2=255):
    hist = get_histogram(img, normalise=True, cumilative=True)
    for r in range(255):
        img[img == r] = img[img == r] * hist[r] * float(L2)/np.max(img)
    return np.uint8(img)


def plot_histogram(img, title):
    hist = get_histogram(img, normalise=False, cumilative=False)
    plt.figure()
    plt.bar(range(255), hist)
    plt.suptitle(title)
    return


if __name__ == '__main__':
    # example usage
    _img = read_img("../MP3/moon.bmp")
    _imgt = histogram_equalisation(_img)
    Image.fromarray(_imgt).save('result.bmp') # save

    # view the before/after histograms
    plot_histogram(read_img("../MP3/moon.bmp"), "original image")
    plot_histogram(_imgt, "transformed image")

