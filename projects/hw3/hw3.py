from skimage import io, img_as_float32, transform, filters
from skimage.color import rgb2gray
import matplotlib.pyplot as plt
import numpy as np
from scipy.fft import fft2, fftshift

def main():
    # headshot FFT
    img = io.imread("headshot.jpg")
    float_img = img_as_float32(img[:, :, :3])
    gray_img = rgb2gray(float_img)
    downsampled_img = transform.rescale(gray_img, 0.5)
    img_fur = fftshift(fft2(downsampled_img))
    img_fur_mag = np.abs(img_fur)
    img_fur_pha = np.angle(img_fur)
    img_fur_mag_log = np.log(1+img_fur_mag)

    # Low pass
    low_pass_headshot = filters.gaussian(downsampled_img, sigma=1)
    low_pass_fur = fftshift(fft2(low_pass_headshot))
    low_pass_fur_mag = np.abs(low_pass_fur)
    low_pass_fur_pha = np.angle(low_pass_fur)
    low_pass_fur_mag_log = np.log(1+low_pass_fur_mag)

    # High pass
    cat_img = io.imread("cat.jpg")
    float_cat = img_as_float32(cat_img[200:2400, 850:3600, :3])
    gray_cat = rgb2gray(float_cat)
    downsampled_cat = transform.resize(gray_cat, downsampled_img.shape, anti_aliasing=True)
    low_pass_cat = filters.gaussian(downsampled_cat, sigma=1)
    high_pass_cat = downsampled_cat - low_pass_cat
    high_pass_max = high_pass_cat.max()
    high_pass_min = high_pass_cat.min()
    normalized_cat = (high_pass_cat - high_pass_min) / (high_pass_max - high_pass_min)
    cat_fur = fftshift(fft2(normalized_cat))
    cat_fur_mag = np.abs(cat_fur)
    cat_fur_pha = np.angle(cat_fur)
    cat_fur_mag_log = np.log(1+cat_fur_mag)

    combined_img = high_pass_cat + low_pass_headshot


    plt.imsave("deliverables/downsampled_headshot.png", downsampled_img, cmap="gray")
    plt.imsave("deliverables/headshot_mag_log.png", img_fur_mag_log, cmap="gray")
    plt.imsave("deliverables/low_pass_headshot.png", low_pass_headshot, cmap="gray")
    plt.imsave("deliverables/headshot_low_pass_mag_log.png", low_pass_fur_mag_log, cmap="gray")
    plt.imsave("deliverables/downsampled_cat.png", downsampled_cat, cmap="gray")
    plt.imsave("deliverables/high_pass_cat.png", normalized_cat, cmap="gray")
    plt.imsave("deliverables/cat_fur_mag_log.png", cat_fur_mag_log, cmap="gray")
    plt.imsave("deliverables/combined_img.png", combined_img, cmap="gray")

if __name__ == "__main__":
    main()