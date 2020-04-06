'''
Alter Vision::Visao Inteligente
Author: @juliomilani
Date: 06/04/2020

Script: pylon_main.py
Description: Get frame from Basler Ace through pypylon and displays detected level
'''
from pypylon import pylon
import cv2
from level_finder import find_level


# conecting to the first available camera
camera = pylon.InstantCamera(pylon.TlFactory.GetInstance().CreateFirstDevice())

# Grabing Continusely (video) with minimal delay
camera.StartGrabbing(pylon.GrabStrategy_LatestImageOnly) 
converter = pylon.ImageFormatConverter()

# converting to opencv bgr format
converter.OutputPixelFormat = pylon.PixelType_BGR8packed
converter.OutputBitAlignment = pylon.OutputBitAlignment_MsbAligned

while camera.IsGrabbing():
    grabResult = camera.RetrieveResult(5000, pylon.TimeoutHandling_ThrowException)

    if grabResult.GrabSucceeded():
        # Access the image data
        image = converter.Convert(grabResult)
        img = image.GetArray()
        y_level = find_level(img)
        cv2.line(img, (0, y_level), (img.shape[1], y_level), (255, 0, 0), thickness=3)
        cv2.namedWindow('title', cv2.WINDOW_NORMAL)
        cv2.imshow('title', img)
        k = cv2.waitKey(1)
        if k == 27:
            break
    grabResult.Release()
    
# Releasing the resource    
camera.StopGrabbing()

cv2.destroyAllWindows()
