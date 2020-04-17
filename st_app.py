import cv2 as cv
import streamlit as st
import os
import numpy as np
import pandas as pd
from PIL import Image
from skimage import io,color,filters, measure
import matplotlib.pyplot as plt
import altair as alt

class Params:
    def __init__(self):
        self.baseline = 2
        self.img_width = 300
        if st.sidebar.checkbox('Filtros:'):
            self.kernel = st.sidebar.slider('Kernal Gauss',min_value = 0,max_value=100,value=5)
            if(self.kernel%2==0):
                self.kernel+=1
            self.dia_bil = st.sidebar.slider('Diameter Bilateral',min_value = 0,max_value=255,value=8)
            self.sigcol_bil = st.sidebar.slider('SigmaColor Bilateral',min_value = 0,max_value=255,value=50)
            self.sigspa_bil = st.sidebar.slider('SigmaSpace Bilateral',min_value = 0,max_value=255,value=50)
            self.t1can = st.sidebar.slider('MinThresh Canny',min_value = 0,max_value=255,value=10)
            self.t2can = st.sidebar.slider('MaxThresh Canny',min_value = 0,max_value=255,value=30)
        else:
            self.kernel = 5
            self.dia_bil = 8
            self.sigcol_bil = 50
            self.sigspa_bil = 50 
            self.t1can = 10
            self.t2can = 30
        if st.sidebar.checkbox('Ydif:'):
            self.ydiff = st.sidebar.slider('Ydif_inicial',min_value = 0,max_value=255,value=70)
        else:
            self.ydiff = 70
        if st.sidebar.checkbox('Roi:'):
            self.roix = st.sidebar.slider('Roi_X_scale',min_value = 0.0,max_value=5.0,value=1.5)
            self.roiymin = st.sidebar.slider('Roi_y_min',min_value = 0.0,max_value=3.0,value=1.0)
            self.roiymax = st.sidebar.slider('Roi_y_max',min_value = 0.0,max_value=3.0,value=1.0)
        else:
            self.roix = 1.5
            self.roiymin = 1.0
            self.roiymax = 1.0
params = Params()

def main():
    st.title('Fill level inspection playground:')
    "TCC do Julio"
    # Uploaded image part:
    uploaded_file = st.file_uploader("Choose a image file", type=["jpg",'png','jpeg','gif'])
    if uploaded_file is not None:
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        upl_img_0 = cv.imdecode(file_bytes, 1)
        upl_img_0 = rescale(upl_img_0)
        upl_img_1 = foo(upl_img_0)
        st.image([upl_img_0,upl_img_1], width=params.img_width)

    # Images in folder part:
    imgs_path = get_all_imgs_paths()
    for i,img_path in enumerate(imgs_path):
        st.write("Img",i,os.path.abspath(imgs_path[i]),":")
        img_0 = cv.imread(img_path,0)
        img_0 = rescale(img_0)
        img_1 = foo(img_0)
        st.image([img_0,img_1], width=params.img_width)


    
def get_all_imgs_paths():
    folder_0 = "images/"
    folders_1 = os.listdir(folder_0)
    imgs_path = []
    for folder_1 in folders_1:
        folder_2 = os.path.join(folder_0,folder_1)
        imgs_filenames = os.listdir(folder_2)
        for img_filename in imgs_filenames:
            img_path = os.path.join(folder_2,img_filename)
            imgs_path.append(img_path)
    st.write("Number of images: ", len(imgs_path))
    return imgs_path

def rescale(img_in):
    new_width = 1280
    scale_ratio = new_width / img_in.shape[1]
    new_height = int(img_in.shape[0] * scale_ratio)
    dim = (new_width, new_height)
    img_out = cv.resize(img_in.copy(), dim, interpolation = cv.INTER_AREA)
    return img_out

# @st.cache()
def foo(img_in):
    img_out = cv.GaussianBlur(img_in,(params.kernel,params.kernel),cv.BORDER_DEFAULT)
    img_out = cv.bilateralFilter(img_out, params.dia_bil, params.sigcol_bil, params.sigspa_bil)
    img_out = cv.Canny(img_out,params.t1can,params.t2can)
    # Find ymin: the Y coordinate of the toppest point
    ymin = np.nonzero(img_out)[0].min()
    for i,line in enumerate(img_out):
        if(np.count_nonzero(line)>5):
            ymin = i
            break
    
    im_height, im_width = img_out.shape
    y_line_2 = ymin+params.ydiff
    if y_line_2 < im_height and y_line_2 > 0:
        line_2 = img_out[y_line_2]
    else:
        line_2 = img_out[0]
    
    mask1 = (np.arange(im_width)>100)*1
    mask2 = (np.arange(im_width)<im_width-100)*1
    line_2 = line_2*mask1*mask2 # Crop edges
    if len(np.nonzero(line_2)[0]) != 0:
        xmin = np.nonzero(line_2)[0].min()
        xmax = np.nonzero(line_2)[0].max()
        cap_width = xmax - xmin
        st.write("Cap Width:",cap_width)
    else:
        xmin, xmax = 0,0
        cap_width = 0 
        params.ydiff = 0  

    # <DEV>

    
    roi_xmin = int(xmin - (params.roix-1)*cap_width/2)
    roi_xmax = int(xmax + (params.roix-1)*cap_width/2)
    roi_ymin = int(ymin + (207*cap_width/218)*params.roiymin)
    roi_ymax = int(ymin + (470*cap_width/218)*params.roiymax)

    img_roi = np.zeros(img_out.shape,dtype=int)
    img_roi[roi_ymin:roi_ymax,roi_xmin:roi_xmax] = img_out.copy()[roi_ymin:roi_ymax,roi_xmin:roi_xmax]
    st.image(img_roi[roi_ymin:roi_ymax,roi_xmin:roi_xmax], width=params.img_width)


    points = np.transpose(np.nonzero(img_roi))
    points[:,[0, 1]] = points[:,[1, 0]] #inversting x and y
    # df = pd.DataFrame(points,columns=['x','y'])
    # c = alt.Chart(df).mark_circle().encode(x=alt.X('x', scale=alt.Scale(domain=[0,1280])), y=alt.Y('y', scale=alt.Scale(domain=[1024,0])), tooltip=['x', 'y'])
    # st.write(c)
    y_tmp = points[:,1]
    # hist = plt.hist(y_tmp,bins=20)

    hist,edges = np.histogram(y_tmp,bins=20)
    edges = edges[:-1]+np.diff(edges)/2
    # hist = np.hstack

    'Info:',edges[:-1].shape
    'Info:',hist.shape
    plt.bar(edges,hist,width=5)
    st.pyplot()
    # st.bar_chart(info)

    # a=np.argmax(hist,axis=0)
    # "argmax",a
    # # hist[a]
    # # "max",a
    # st.pyplot()
    # st.write(hist)

    # num_linhas = params.num_cols
    # x_tests = np.linspace(xmin-100,xmax+100,num=num_linhas)
    # x_tests = [int(x_test) for x_test in x_tests]
    # points = np.transpose(np.nonzero(img_out))
    # points[:,[0, 1]] = points[:,[1, 0]] #inversting x and y
    
    # mask = np.isin(points[:,0],x_tests).nonzero()
    # points = points[mask]
    # df = pd.DataFrame(points,columns=['x','y'])
    # c = alt.Chart(df).mark_circle().encode(x=alt.X('x', scale=alt.Scale(domain=[0,1280])), y=alt.Y('y', scale=alt.Scale(domain=[1024,0])), tooltip=['x', 'y'])
    # st.write(c)

    # y_tmp = points[:,1]
    # hist = plt.hist(y_tmp,bins=30)
    # st.pyplot()
    # st.write(hist)
    
    # h = int(cap_width*400/218)
    # cv.line(img_out, (0, y_line_2+h), (im_width, y_line_2+h), 255, thickness=1)

    # y_tmp = points[mask_tmp]
    # y_tmp = y_tmp[:,1]
    # y_tmp = points[:,1]
  
    # hist = plt.hist(y_tmp,bins=30)
    # st.pyplot()
    # st.write(hist)

    # for x in x_tests:
        # cv.line(img_out, (x, 0), (x, im_height), 255, thickness=1)
    # <\DEV>

    cv.rectangle(img_out, (roi_xmin,roi_ymin),(roi_xmax,roi_ymax), 255, 1)
    cv.line(img_out, (0, ymin), (im_width, ymin), 255, thickness=2)
    cv.line(img_out, (0, y_line_2), (im_width, y_line_2), 255, thickness=2)
    cv.line(img_out, (xmin, 0), (xmin, y_line_2), 255, thickness=2)
    cv.line(img_out, (xmax, 0), (xmax, y_line_2), 255, thickness=2)
    cv.putText(img_out,str(cap_width),(xmin,ymin-15),cv.FONT_HERSHEY_SIMPLEX, 2, 255,2)

    


    return img_out








if __name__ == '__main__':
    main()
