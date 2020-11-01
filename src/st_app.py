# Programa que encontra a quantidade de água presente em uma garrafa
# Inspeção de nível
# 
# 
# Projeto de Diplomacao
# Julio Milani de Lucena
# Orientador: Altamiro Susin
# UFRGS 2020/1
# 
# 
# TODO:
# -Adicionar try..except
# -Resolver problema das escritas
# 
import cv2 as cv
import numpy as np
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
        self.img_width = 300
        self.border_crop = 100
        self.y_tol = 20      
        self.cap_width = 30 #mm
        
        self.set = st.sidebar.selectbox('Choose image set',('Train set', 'Test set'))
        self.reference = st.sidebar.slider('Reference [mm]',min_value = 40.0,max_value=60.0,value=48.5)
        self.tolerance_plus = st.sidebar.slider('Tolerance + [mm]',min_value = 0.0, max_value=3.0,value=1.0)    
        self.tolerance_minus = st.sidebar.slider('Tolerance - [mm]',min_value = 0.0, max_value=3.0,value=1.0)
        self.draw_cursor = st.sidebar.checkbox('Cursor:')
        if self.draw_cursor:
            self.cursor_y = st.sidebar.slider('Cursor Y',min_value = 0,max_value=1500,value=5)
            self.cursor_x = st.sidebar.slider('Cursor X',min_value = 0,max_value=1500,value=5)


        if st.sidebar.checkbox('Filtros:'):
            self.kernel = st.sidebar.slider('Kernal Gauss',min_value = 0,max_value=100,value=3)
            if(self.kernel%2==0):
                self.kernel+=1
            self.dia_bil = st.sidebar.slider('Diameter Bilateral',min_value = 0,max_value=255,value=15)
            self.sigcol_bil = st.sidebar.slider('SigmaColor Bilateral',min_value = 0,max_value=255,value=50)
            self.sigspa_bil = st.sidebar.slider('SigmaSpace Bilateral',min_value = 0,max_value=255,value=50)
            self.t1can = st.sidebar.slider('MinThresh Canny',min_value = 0,max_value=255,value=5)
            self.t2can = st.sidebar.slider('MaxThresh Canny',min_value = 0,max_value=255,value=15)
        else:
            self.kernel = 3
            self.dia_bil = 15
            self.sigcol_bil = 50
            self.sigspa_bil = 50 
            self.t1can = 5
            self.t2can = 15

        
        self.ydiff = st.sidebar.slider('Ydif_tampa',min_value = 0,max_value=255,value=70)
        

        self.show_roi = st.sidebar.checkbox('Show_roi')

        if st.sidebar.checkbox('Roi config'):
            self.roix = st.sidebar.slider('Roi_X_scale',min_value = 0.0,max_value=5.0,value=1.5)
            self.roiymin = st.sidebar.slider('Roi_y_min',min_value = 0.0,max_value=3.0,value=1.0)
            self.roiymax = st.sidebar.slider('Roi_y_max',min_value = 0.0,max_value=3.0,value=1.0)
        else:
            self.roix = 1.5
            self.roiymin = 1.0
            self.roiymax = 1.0

        self.draw_histogram = st.sidebar.checkbox('Histogram:')
        if self.draw_histogram:
            self.hist_thresh = st.sidebar.slider('Hist Thresh',min_value = 0.0,max_value=8.0,value=1.0)
            self.hist_bins = st.sidebar.slider('Hist Bins',min_value = 0,max_value=100,value=20)
        else:
            self.hist_thresh = 1.0
            self.hist_bins = 20

# GLOBAL VARS
params = Params()
confusion_matrix = np.zeros((3,3)) # [nok-, ok, nok+]

def main():
    global params
    global confusion_matrix
    st.title('Fill level inspection playground:')
    "TCC do Julio"
    # Uploaded image part:
    uploaded_file = st.file_uploader("Choose a image file", type=["jpg",'png','jpeg','gif'])
    if uploaded_file is not None:
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        upl_img_0 = cv.imdecode(file_bytes,1)
        upl_img_0 = rescale(upl_img_0)
        upl_img_1,y_level,status = foo(upl_img_0)
        st.write("Status:",status)
        st.image([upl_img_0,upl_img_1], width=params.img_width)

    # Images in folder part:
    imgs_path = get_all_imgs_paths()
    for i,img_path in enumerate(imgs_path):
        st.write("Img",i,os.path.abspath(imgs_path[i]),":")
        base_name = os.path.basename(img_path)
        img_0 = cv.imread(img_path,0)
        img_0 = rescale(img_0)
        img_1,y_level,status = foo(img_0)
        check_gt(img_path,status)
        cv.line(img_0, (0, y_level), (img_0.shape[1], y_level), 0, thickness=2)
        img_0,img_1 = draw_cursor(img_0,color=0),draw_cursor(img_1)
        st.image([img_0,img_1], width=params.img_width)
        st.write('--------------------------------')
    df = pd.DataFrame(
        confusion_matrix,
        columns = ['PRED_NOK-','PRED_OK','PRED_NOK+'],
        index = ['GT_NOK-','GT_OK','GT_NOK+']
    )
    st.dataframe(df)
    accuracy = (confusion_matrix[0,0]+confusion_matrix[1,1]+confusion_matrix[2,2])/np.sum(confusion_matrix)
    precision = (confusion_matrix[1,1])/np.sum(confusion_matrix[:,1])
    recall = (confusion_matrix[1,1])/np.sum(confusion_matrix[1,:])
    f1 = 2*(recall * precision) / (recall + precision)
    st.write("Accuracy:",accuracy*100,"%")
    st.write("Precision:",precision*100,"%")
    st.write("Recall:",recall*100,"%")
    st.write("F1:",f1)
    # true_negatives = confusion_matrix[]
    # Accuracy = TP+TN/TP+FP+FN+TN
    # accuracy = 



def check_gt(img_path,status):
    global confusion_matrix
    gt = img_path.split("/")[-2]    
    st.write("Estimated:",status,"Ground Truth:",gt)
    idx_gt = 0
    idx_pred = 0
    if gt == "nok-":
        idx_gt = 0
    if gt == "ok":
        idx_gt = 1
    if gt == "nok+":
        idx_gt = 2
    if status == "nok-":
        idx_pred = 0
    if status == "ok":
        idx_pred = 1
    if status == "nok+":
        idx_pred = 2
    confusion_matrix[idx_gt,idx_pred] += 1

def draw_cursor(img_in,color=255):
    img_out = img_in.copy()
    if params.draw_cursor:
        cv.line(img_out, (0, params.cursor_y), (img_out.shape[1], params.cursor_y), color, thickness=1)
        cv.line(img_out, (params.cursor_x, 0), (params.cursor_x, img_out.shape[0]), color, thickness=1)
    return img_out

def get_all_imgs_paths():
    global params
    if params.set=="Train set":
        folder_0 = "train_images/"
    else:
        folder_0 = "test_images/"
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
    # Finds xmin, xmax: X's coordenates of the cap
    y_line_2 = ymin+params.ydiff
    if y_line_2 < im_height and y_line_2 > 0:
        line_2 = img_out[y_line_2]
    else:
        line_2 = img_out[0]
    mask1 = (np.arange(im_width)>params.border_crop)*1
    mask2 = (np.arange(im_width)<im_width-params.border_crop)*1
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

    roi_xmin = int(xmin - (params.roix-1)*cap_width/2)
    roi_xmax = int(xmax + (params.roix-1)*cap_width/2)
    roi_ymin = int(ymin + (270*cap_width/218)*params.roiymin)
    roi_ymax = int(ymin + (450*cap_width/218)*params.roiymax)

    img_roi = np.zeros(img_out.shape,dtype=int)
    img_roi[roi_ymin:roi_ymax,roi_xmin:roi_xmax] = img_out.copy()[roi_ymin:roi_ymax,roi_xmin:roi_xmax]
    
    if params.show_roi:
        st.image(img_roi[roi_ymin:roi_ymax,roi_xmin:roi_xmax], width=params.img_width)

    points = np.transpose(np.nonzero(img_roi))
    points[:,[0, 1]] = points[:,[1, 0]] #inversting x and y    
    hist,edges = np.histogram(points[:,1],bins=params.hist_bins)
    edges = edges[:-1]+np.diff(edges)/2 #edges.size--; Faz a media entre os extremos

    id_max = np.argmax(hist) 
    hist_max = hist[id_max] 
    "Histmax", hist_max

    hist_thresh_abs = int(params.hist_thresh*100*(cap_width/184)*(20/params.hist_bins))
    if hist_max > hist_thresh_abs:
        y_level = int(edges[id_max])
    else:
        y_level = 0
        st.write("Empty bottle")

    "Y level = ",y_level

    # porcentage = (y_level-ymin-224*cap_width/250)/(314*cap_width/250)
    # porcentage = 100*(1 - porcentage)
    porcentage = (y_level-ymin) / cap_width
    distance = porcentage * params.cap_width
    error = params.reference - distance
    st.write("Level:",distance,"mm")
    st.write("Error:",error,"mm")
    if error > params.tolerance_plus:
        status = "nok+"
    elif error < - params.tolerance_minus:
        status = "nok-"
    else:
        status = "ok"

    if params.draw_histogram:
        st.write("Histogram Threshold:",hist_thresh_abs)
        plt.bar(edges,hist,width=5)
        st.pyplot()
    
    cv.line(img_out, (0, y_level), (im_width, y_level), 255, thickness=2)
    cv.rectangle(img_out, (roi_xmin,roi_ymin),(roi_xmax,roi_ymax), 255, 1)
    cv.line(img_out, (0, ymin), (im_width, ymin), 255, thickness=2)
    cv.line(img_out, (0, y_line_2), (im_width, y_line_2), 255, thickness=2)
    cv.line(img_out, (xmin, 0), (xmin, y_line_2), 255, thickness=2)
    cv.line(img_out, (xmax, 0), (xmax, y_line_2), 255, thickness=2)
    cv.putText(img_out,str(cap_width),(xmin,ymin-15),cv.FONT_HERSHEY_SIMPLEX, 2, 255,2)

    return img_out,y_level,status



if __name__ == '__main__':
    main()
