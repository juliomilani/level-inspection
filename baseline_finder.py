#%% impor modules
import cv2
import os
import numpy as np
import matplotlib.pyplot as plt

#%%
def show_images(images,titles,color_map='viridis'):
    """
    Show images using matplotlib figures
    """
    fig=plt.figure(figsize=(10,8))
    for i,image in enumerate(images):
        fig.add_subplot(2,3,i+1)
        plt.title(titles[i])
        plt.imshow(image,cmap=color_map)

#%% import all images in images/ to the vector images
images_name = [path for path in os.listdir("images/") if path.endswith('.jpg')]
images_bgr = [cv2.imread(os.path.join("images",image_name)) for image_name in images_name]
images_rgb = [cv2.cvtColor(im_bgr, cv2.COLOR_BGR2RGB) for im_bgr in images_bgr]
images_gray = [cv2.cvtColor(im_bgr, cv2.COLOR_BGR2GRAY) for im_bgr in images_bgr]
show_images(images_rgb,images_name,'gray')


images_blue = [cv2.split(img_bgr)[0] for img_bgr in images_bgr]
blue_title = ["blue:"+name for name in images_name]
show_images(images_blue,blue_title,'gray')

images_green = [cv2.split(img_bgr)[1] for img_bgr in images_bgr]
images_red = [cv2.split(img_bgr)[2] for img_bgr in images_bgr]

temp = []
for i,_ in enumerate(images_blue):
    img_mean = np.mean(images_blue)
    aux = np.subtract(images_blue[i],images_gray[i]) * (np.subtract(images_blue[i], images_gray[i]) > 0) 
    temp.append(np.subtract(images_blue[i],images_gray[i]) * (np.subtract(images_blue[i], images_gray[i]) > 0))
show_images(temp,blue_title,'gray')

edges = [cv2.Canny(img,25,70) for img in images_gray]
edges_title = ["canny:25,70"+name for name in images_name]
#show_images(edges,edges_title)

plt.show()

# %%
