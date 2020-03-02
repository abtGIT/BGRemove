# This is the problem for First technical round for the role of Computer Vision Engineer at Vectorly
# More details at https://www.linkedin.com/jobs/view/1629909785/
#
# Write a function which will segment and extract the text overlay "Bart & Homer's EXCELLENT Adventure" 
# Input image is at https://vectorly.io/demo/simpsons_frame0.png
# Output : Image with only the overlay visible and everything else white
# 
# Note that you don't need to extract the text, the output is an image with only 
# the overlay visible and everything else (background) white
#
# You can use the snipped below (in python) to get started if you like 
# Python is not required but is preferred. You are free to use any libraries or any language


#####################
import cv2
import numpy as np

def getTextOverlay(input_image):
    output = np.zeros(input_image.shape, dtype=np.uint8)
    output[input_image>np.std(input_image)]=255 # same as removeColor
    output = cv2.cvtColor(output,cv2.COLOR_BGR2GRAY)
    output = keepROI(output)
    cv2.imshow('output',output)
    cv2.waitKey(0)
    return output

def keepROI(image):
    thresh = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))
    erode = cv2.erode(thresh, kernel, iterations=2)
    erode = cv2.bitwise_not(erode)
    return erode
    
def removeColor(input_image):
    out = input_image / np.linalg.norm(input_image)
    out = np.sum(out, axis=2)
    out=out*765
    std_val = np.std(out)
    out[out>=std_val]=255
    out[out<std_val]=0
    out= keepROI(out.astype(np.uint8))
    return out
    
if __name__ == '__main__':
    image = cv2.imread('simpsons_frame0.png')
    output = getTextOverlay(image)
    cv2.imwrite('simpons_text.png', output)
#####################

