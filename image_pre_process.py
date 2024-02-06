from PIL import Image
import cv2
import numpy as np
import math
from scipy.ndimage import interpolation as inter
import scipy.ndimage as ndimage
from deskew import determine_skew

def image_resize(img):
    height_x = img.shape[0]
    width_y = img.shape[1]
    factor = max(1, float(1024.0 / height_x))
    width = int(factor * width_y)
    height = int(factor * height_x)
    return img

def super_resolution(img):
    height_x = img.shape[0]
    width_y = img.shape[1]
    if(height_x * width_y<2480*1280):
        sr = cv2.dnn_superres.DnnSuperResImpl_create()
        path = "ESPCN_x3.pb"
        sr.readModel(path)
        sr.setModel("espcn",3)
        img = sr.upsample(img)
    return img

def binarization(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ret, thresh = cv2.threshold(gray, 130, 255, cv2.THRESH_BINARY)
    return thresh


def sharpen(img):
    sharpen_op = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]], dtype=np.float32)
    sharpen_image = cv.filter2D(img, cv.CV_32F, sharpen_op)
    sharpen_image = cv.convertScaleAbs(sharpen_image)
    return sharpen_image
    
def projection_correction(image, delta=1, limit=45):
    def determine_score(img, angle):
        data = inter.rotate(img, angle, reshape=False, order=0)
        histogram = np.sum(data, axis=1, dtype=float)
        score = np.sum((histogram[1:] - histogram[:-1]) ** 2, dtype=float)
        return histogram, score
    
    scores = []
    angles = np.arange(-limit, limit + delta, delta)
    for angle in angles:
        histogram, score = determine_score(image, angle)
        scores.append(score)

    best_angle = angles[scores.index(max(scores))]
    
    #affine transformation to correct the skew
    (h, w) = image.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, best_angle, 1.0)
    corrected = cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_CUBIC,borderMode=cv2.BORDER_REPLICATE)

    return corrected


def Hough_correction(img):
    
    # gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(img,50,150,apertureSize = 3)
    lines = cv2.HoughLines(edges,1,np.pi/180,200)
    if lines is None:
        print ('no lines')

    for rho,theta in lines[0]:
        
        a = np.cos(theta)
        b = np.sin(theta)
        x0 = a*rho
        y0 = b*rho
        x1 = int(x0 + 1000*(-b))
        y1 = int(y0 + 1000*(a))
        x2 = int(x0 - 1000*(-b))
        y2 = int(y0 - 1000*(a))
        cv.line(img, (x1, y1), (x2, y2), (0, 0, 255), 1)
        if x1 == x2 or y1 == y2:
           continue
    cv.imwrite('files/line.jpg', img)     
    t = float(y2-y1)/(x2-x1)
    rotate_angle = math.degrees(math.atan(t))
    if rotate_angle > 45:
        rotate_angle = -90 + rotate_angle
    elif rotate_angle < -45:
        rotate_angle = 90 + rotate_angle

    #affine transformation to correct the skew
    (h, w) = img.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, rotate_angle, 1.0)
    corrected = cv2.warpAffine(img, M, (w, h), flags=cv2.INTER_CUBIC,borderMode=cv2.BORDER_REPLICATE)

    return corrected

def dskew_correction(image):
    #preprocessed
    img = image.copy()
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    
    #get angle using hough
    rotate_angle = determine_skew(img)
    
    #affine transformation to correct the skew
    (h, w) = img.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, rotate_angle, 1.0)
    corrected = cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_CUBIC,borderMode=cv2.BORDER_REPLICATE)
    return corrected

def findContours_img(img):
    exist_countours = False
    
    #preprocessed
    cnt_img = img.copy()
    gray = cv2.cvtColor(cnt_img, cv2.COLOR_BGR2GRAY)
    Blur = cv2.GaussianBlur(gray, (5, 5), 0)
    edged = cv2.Canny(Blur, 75, 200)
    
    #find countours
    contours, hierarchy = cv2.findContours(edged, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    
    #get the second largest contour
    contours = sorted(contours, key=cv2.contourArea, reverse=True)[:5]  
    
    for c in contours:
        #get the Polygon fitting curve
        epsilon=0.02*cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, epsilon, True)
        if len(approx) == 4:
            screenCnt = approx
            exist_countours = True
            break


    
    #less than 4 points detected, use hough transform instead
    if exist_countours==False:
        return exist_countours,1, dskew_correction(img) 
    
    #more than 4 points detected
    exist_countours = True
    box = approx.reshape(4, 2)

    #draw the contours
    draw_img = cv2.drawContours(img.copy(), [box], -1, (0, 255, 0), 3)
    
    #sort countours
    sum = box.sum(axis=1)
    diff = np.diff(box,axis=1)
    
    #in the order of lower right, upper right, upper left, lower left
    temp0 = box[np.argmax(sum)].copy()
    temp1 = box[np.argmax(diff)].copy()
    temp2 = box[np.argmin(sum)].copy()
    temp3 = box[np.argmin(diff)].copy()

    box[0]=temp0
    box[1]=temp1
    box[2]=temp2
    box[3]=temp3

    # print("box[0]:", box[0])
    # print("box[1]:", box[1])
    # print("box[2]:", box[2])
    # print("box[3]:", box[3])
    return exist_countours,box,draw_img   

def Perspective_transform(box,original_img):
    # Get width and height of the ouput image
    width = math.ceil(np.sqrt((box[3][1] - box[2][1])**2 + (box[3][0] - box[2][0])**2))
    height= math.ceil(np.sqrt((box[3][1] - box[0][1])**2 + (box[3][0] - box[0][0])**2))

    # Four vertices of the original image
    pts1 = np.float32([box[0], box[1], box[2], box[3]])
    
    # Transformation matrix
    pts2 = np.float32([[int(width+1),int(height+1)], [0, int(height+1)], [0, 0], [int(width+1), 0]])

    # Generate the perspective transformation matrix
    M = cv2.getPerspectiveTransform(pts1, pts2)
    
    #Perform perspective transformations
    result_img = cv2.warpPerspective(original_img, M, (int(width+3),int(height+1)))

    return result_img

def pre_process(img):
    img = super_resolution(img)
    exist,box,img = findContours_img(img)
    if (exist==True):        
       img = Perspective_transform(box, img)
    img = binarization(img)
    return img