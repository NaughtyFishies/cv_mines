from skimage import io, img_as_float32
from skimage.color import rgb2gray
import matplotlib.pyplot as plt
import numpy as np

def main():
    img = io.imread("headshot.jpg")
    float_img = img_as_float32(img[:, :, :3])
    gray_img = rgb2gray(float_img)
    plt.imshow(gray_img, cmap="gray")
    plt.axis("off")
    plt.show()

    print(gray_img[:3, :5])

    print(gray_img[1, 0])

if __name__ == "__main__":
    main()
