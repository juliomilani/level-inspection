#%% impor modules
import cv2
import os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from skimage.measure import label, regionprops

#%%
def show_images(images,titles=['x','x','x','x','x','x'],color_map='viridis'):
    """
    Show images using matplotlib figures
    """
    fig=plt.figure(figsize=(10,8))
    for i,image in enumerate(images):
        fig.add_subplot(2,3,i+1)
        plt.title(titles[i])
        plt.imshow(image,cmap=color_map)

#%% import all images in images/ to the vector images

folder_path = "images/with_lighting"
images_name = [path for path in os.listdir(folder_path) if path.endswith('.jpg')]
images_bgr = [cv2.imread(os.path.join(folder_path,image_name)) for image_name in images_name]
images_rgb = [cv2.cvtColor(im_bgr, cv2.COLOR_BGR2RGB) for im_bgr in images_bgr]
images_gray = [cv2.cvtColor(im_bgr, cv2.COLOR_BGR2GRAY) for im_bgr in images_bgr]
show_images(images_rgb,images_name,'gray')


def get_cap_masks(images):
    lower = np.array([0, 60, 100])
    upper = np.array([59, 130, 200])
    images_blurred = [cv2.GaussianBlur(img_rgb,(5,5),0) for img_rgb in images]
    cap_masks = [cv2.inRange(img_rgb, lower, upper) for img_rgb in images_blurred]
    kernel = np.ones((5,5),np.uint8)
    cap_masks = [cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel) for img in cap_masks]

    for i,cap_mask in enumerate(cap_masks):
        props = regionprops(cap_mask)
        (ymin, xmin, ymax, xmax) = props[0].bbox
        cap_mask = cv2.rectangle(cap_mask, (xmin, ymin), (xmax, ymax), (0, 255, 0), thickness=3)
        cap_masks[i] = cap_mask
    show_images(cap_masks,images_name,'gray')
    return cap_masks

def find_cap_bb(cap_masks):
    cap_bbs = []
    images = [img for img in images_rgb]
    for i,cap_mask in enumerate(cap_masks):
        props = regionprops(cap_mask)
        (ymin, xmin, ymax, xmax) = props[0].bbox
        print(props[0].bbox)
        (x, y, w, h) = (xmin, ymin, xmax - xmin, ymax - ymin)
        print("x:%s, y:%s, w:%s, h:%s i:%s" % (x, y, w, h, i))
        cv2.rectangle(images[i], (x, y), (x + w, y + h), (0, 255, 0), thickness=3)
        cv2.line(images[i], (0, y+h), (images[i].shape[1], y+h), (0, 255, 0), thickness=3)
        el = cap_mask.copy()[y:y + h, x:x + w]
        pil_im = Image.fromarray(images_rgb[0])
    show_images(images,images_name)

find_cap_bb(get_cap_masks(images_rgb))


plt.show()
