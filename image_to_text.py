import pytesseract
from PIL import Image
import image_pre_process
import cv2
from PaddleOCR.tools.infer import predict_e2e

def tesseract(path):
    img = cv2.imread(path)
    img = image_pre_process.pre_process(img)
    cv2.imwrite('files/result.jpg', img)
    img = Image.open('files/result.jpg')
    string = pytesseract.image_to_string(img,lang='eng')
    return string

def PGNet(path):
    img = cv2.imread(path)
    text, points, strs, src_im = predict_e2e.PGNet(img)
    height = img.shape[0]
    width = img.shape[1]
    array = []
    ##Get average coordinates of each words
    for element in points:
        x_sum = 0
        y_sum = 0
        for i in range(len(element)):
            x_sum += element[i][0]
            y_sum += element[i][1]
        x_mean = (x_sum/len(element))/width
        y_mean = (y_sum/len(element))/height
        array.append([x_mean,y_mean])
    
    arr = [array[i] + [strs[i]] for i in range(len(array))]
    
    ##First sort base on Y coordinates
    arr.sort(key=lambda x: x[1])
    
    #Check adjacent elements, if Y coordinate difference is less than 2% of the height, sort by X coordinate
    i = 0
    while i < len(arr) - 1:
        if abs(arr[i][1] - arr[i+1][1]) < 0.02:
            j = i
            while j < len(arr) - 1 and abs(arr[j][1] - arr[j+1][1]) < 0.02:
                j += 1
            sub_arr = arr[i:j+1]
            sub_arr.sort(key=lambda x: x[0])
            arr[i:j+1] = sub_arr
            i = j + 1
        else:
            i += 1
    result = []
    for i in range(len(arr)):
        result.append(arr[i][2])
    result = ' '.join(result)
    cv2.imwrite('files/PGNet_result.jpg', src_im)
    return result
    
    



