#%% impor modules
import cv2 as cv
import numpy as np
import time

class Params:
    def __init__(self):
        self.img_width = 300
        self.border_crop = 100
        self.y_tol = 20      
        self.cap_width = 30 #mm
        self.reference = 48.5
        self.tolerance_plus = 3.6    
        self.tolerance_minus = 2.52
        self.kernel = 3
        self.dia_bil = 3
        self.sigcol_bil = 30
        self.sigspa_bil = 30
        self.t1can = 5
        self.t2can = 15
        self.ydiff = 70
        self.roix = 1.5
        self.roiymin = 1.0
        self.roiymax = 1.0
        self.hist_thresh = 1.0
        self.hist_bins = 20





params = Params()

def rescale(img_in):
    new_width = 1280
    scale_ratio = new_width / img_in.shape[1]
    new_height = int(img_in.shape[0] * scale_ratio)
    dim = (new_width, new_height)
    img_out = cv.resize(img_in.copy(), dim, interpolation = cv.INTER_AREA)
    return img_out

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
    points = np.transpose(np.nonzero(img_roi))
    points[:,[0, 1]] = points[:,[1, 0]] #inversting x and y    
    hist,edges = np.histogram(points[:,1],bins=params.hist_bins)
    edges = edges[:-1]+np.diff(edges)/2 #edges.size--; Faz a media entre os extremos
    id_max = np.argmax(hist) 
    hist_max = hist[id_max] 
    hist_thresh_abs = int(params.hist_thresh*100*(cap_width/184)*(20/params.hist_bins))
    if hist_max > hist_thresh_abs:
        y_level = int(edges[id_max])
    else:
        y_level = 0
    porcentage = (y_level-ymin) / cap_width
    distance = porcentage * params.cap_width
    error = params.reference - distance
    return 1

img_path = "15ml_images/ok/photo4976746277559707944.jpg"
time_start = time.time()
rounds = 100
for i in range(rounds):
    img_0 = cv.imread(img_path,0)
    img_0 = rescale(img_0)
    foo(img_0)
print("Time elapsed:", (time.time()-time_start)/rounds)