import imageio.v2 as imageio 
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import uniform_filter

def histogram_equalization(image):
    
    hist, bins = np.histogram(image.flatten(), bins=256, range=[0, 256])
    cdf = hist.cumsum()  
    cdf_normalized = cdf * 255 / cdf[-1] 

   
    image_equalized = np.interp(image.flatten(), bins[:-1], cdf_normalized).reshape(image.shape)
    return image_equalized

def main():
    
    input_image = imageio.imread('C:\\Users\\ASUS\\Downloads\\doraemon.jpg', mode='F')  # Membaca gambar dalam mode float grayscale

    
    equalized_image = histogram_equalization(input_image)

    
    equalized_image_uint8 = equalized_image.astype(np.uint8)

   
    imageio.imwrite('C:\\Users\\ASUS\\Downloads\\doraemon_equalized.jpg', equalized_image_uint8)

   
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.title('Original Image')
    plt.imshow(input_image, cmap='gray')
    plt.axis('off')

    plt.subplot(1, 2, 2)
    plt.title('Equalized Image')
    plt.imshow(equalized_image, cmap='gray')
    plt.axis('off')
    plt.show()

if __name__ == "__main__":
    main()
