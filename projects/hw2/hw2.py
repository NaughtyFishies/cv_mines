from skimage import io, img_as_float32
from skimage.color import rgb2gray
import matplotlib.pyplot as plt
import numpy as np

def main():
    img = io.imread("headshot.jpg")
    float_img = img_as_float32(img[:, :, :3])
    gray_img = rgb2gray(float_img)
    f1 = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]])
    f2_1 = np.array([[0, 0, 0], [0, 2, 0], [0, 0, 0]])
    f2_2 = (1/9) * np.array([[1, 1, 1], [1, 1, 1], [1, 1, 1]])
    f3 = (1/16) * np.array([[1, 2, 1], [2, 4, 2], [1, 2, 1]])
    
    
    downsampled_img = gray_img[119:221, 74:176]
    f1_img = np.empty((100, 100), dtype=np.float32)
    f2_img = np.empty((100, 100), dtype=np.float32)
    f3_img = np.empty((100, 100), dtype=np.float32)
    for y in range(100):
        for x in range(100):
            kernal = np.array([[downsampled_img[y, x], downsampled_img[y, x+1], downsampled_img[y, x+2]], 
                               [downsampled_img[y+1, x], downsampled_img[y+1, x+1], downsampled_img[y+1, x+2]],
                               [downsampled_img[y+2, x], downsampled_img[y+2, x+1], downsampled_img[y+2, x+2]]])
            # f1
            f1_img[y, x] = np.sum(kernal * f1)
            # f2
            f2_img[y, x] = np.sum(kernal * f2_1) - np.sum(kernal * f2_2)
            # f3
            f3_img[y, x] = np.sum(kernal * f3)
            
    plt.imsave("deliverables/downsampled_img.png", downsampled_img, cmap="gray")
    plt.imsave("deliverables/f1_img.png", f1_img, cmap="gray")
    plt.imsave("deliverables/f2_img.png", f2_img, cmap="gray")
    plt.imsave("deliverables/f3_img.png", f3_img, cmap="gray")


if __name__ == "__main__":
    main()
