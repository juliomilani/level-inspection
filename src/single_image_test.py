#%% impor modules
import cv2
import os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

def display_image(image):
    Image.fromarray(np.uint8(image)).show()

#%%load image
image_bgr = cv2.imread("images/im06.jpg")
image = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
display_image(image)

#%%BGR2GRAY
image_gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
display_image(image_gray)

# laplacian = cv2.Laplacian(image_gray,cv2.CV_64F)
# display_image(laplacian)

# sobelx = cv2.Sobel(image_gray,cv2.CV_64F,1,0,ksize=5)
# display_image(sobelx)

# sobely = cv2.Sobel(image_gray,cv2.CV_64F,0,1,ksize=5)
# display_image(sobely)

blurred = cv2.GaussianBlur(image_gray,(3,3),cv2.BORDER_DEFAULT)
display_image(blurred)

canny = cv2.Canny(blurred,10,30)
display_image(canny)

# lines = cv2.HoughLinesP(edges, 1, math.pi/2, 2, None, 30, 1);s

