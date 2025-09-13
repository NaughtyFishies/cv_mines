from skimage import io, img_as_float32, transform
from skimage.color import rgb2gray
import matplotlib.pyplot as plt
import numpy as np

def main():
    img = io.imread("headshot.jpg")
    float_img = img_as_float32(img[:, :, :3])
    gray_img = rgb2gray(float_img)
    kernal1 = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]])
    kernal2 = np.array([[0, 0, 0], [0, 2, 0], [0, 0, 0]]) - ((1/9) * np.array([[1, 1, 1], [1, 1, 1], [1, 1, 1]]))
    kernal3 = (1/16) * np.array([[1, 2, 1], [2, 4, 2], [1, 2, 1]])
    
    
    downsampled_img = transform.rescale(gray_img, 0.5)
    height, width = downsampled_img.shape
    f1_img = np.empty([height - 2, width - 2])
    f2_img = np.empty([height - 2, width - 2])
    f3_img = np.empty([height - 2, width - 2])
    
    for y in range(height - 2):
        for x in range(width - 2):
            img_sample = np.array([[downsampled_img[y, x], downsampled_img[y, x+1], downsampled_img[y, x+2]], 
                               [downsampled_img[y+1, x], downsampled_img[y+1, x+1], downsampled_img[y+1, x+2]],
                               [downsampled_img[y+2, x], downsampled_img[y+2, x+1], downsampled_img[y+2, x+2]]])
            
            # f1
            new_value = np.sum(img_sample * kernal1)
            if new_value < 0:
                f1_img[y, x] = 0
            elif new_value > 1:
                f1_img[y, x] = 1
            else:
                f1_img[y, x] = new_value
            # f2
            f2_img[y, x] = np.sum(img_sample * kernal2)
            # f3
            f3_img[y, x] = np.sum(img_sample * kernal3)
       
    plt.imsave("deliverables/downsampled_img.png", downsampled_img, cmap="gray")
    plt.imsave("deliverables/f1_img.png", f1_img, cmap="gray")
    plt.imsave("deliverables/f2_img.png", f2_img, cmap="gray")
    plt.imsave("deliverables/f3_img.png", f3_img, cmap="gray")


if __name__ == "__main__":
    main()
