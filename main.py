import PySimpleGUI as sg
import image_to_text as ocr
import text_to_speech as tts
import pygame
import cv2
import text_process
from time import sleep
import image_pre_process
import numpy as np
from textblob import TextBlob
import os


def main():
    global window
    global image_path
    global ocr_finish #detect if the ocr procedure has finish
    global take_image #detect if user has take an image
    global disable_correction
    global spell_correction
    global context_correction
    
    
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    spell_correction = False
    context_correction = False
    ocr_finish = False
    take_image = False
    pygame.mixer.init()
    sg.theme('BrownBlue')
    layout = [[]]
    window = sg.Window('Image to Speech System',getlayout(),size=(750,790),element_justification='c')

    while True:
        event, values = window.read() 

        spell_correction = values['spellCorrection']
        context_correction = values['contextCorrection']
        if event is None: #close the window
           break
       
       
        if event in event_callbacks: #events handler
            event_callbacks[event]()
            
        if pygame.mixer.music.get_busy():
            volume(values['volume']) #listen to the changes of volume
            # voiceProgress(values['voiceProgress'])
            # window['voiceProgress'].Update(pygame.mixer.music.get_pos())
        
            
        # if pygame.mixer.music.get_busy():
        #     window['voiceProgress'].Update(pygame.mixer.music.get_pos())
        #     window.refresh()
        #     sleep(1)
        
    window.close()

    
def getlayout(): #User Interface
    return [
        [sg.Text('Welcome to the Image to Speech System',font=('Arial',30))],
        [sg.HSeparator()],
        [sg.Text('')],
        [
            sg.Button('Select Image',font=('Arial',30),auto_size_button=True,key="get_image"),
            sg.Text('    '),
            sg.Button('Open Camera',font=('Arial',30),auto_size_button=True,key='camera')
        ],
        [sg.pin(sg.Text('Tips: Please upload or take an image',visible=True,font=('Arial',15),key='tips',text_color='yellow'))],
        [sg.pin(sg.Text('Tips: Image input successful, converting to text now...',visible=False,font=('Arial',15),key="upLoading",text_color='#8fd400'))],
        [sg.pin(sg.Text('Tips: Please select an image file, try again!',visible=False,font=('Arial',15),key="fileTypeError",text_color='#cc0000'))],
        [sg.HSeparator()],
        [
            sg.Text('Auto Text Correction',font=('Arial',15)),
            sg.Radio('Disable', group_id=1,default= True,font=('Arial',15),key='disableCorrection'),
            sg.Radio('Spell Correction', group_id=1,default = False,font=('Arial',15),key='spellCorrection'),
            sg.Radio('Contextual Correction', group_id=1,default = False,font=('Arial',15),key='contextCorrection'),
            sg.Button('Help',key='help')
        ],
        [sg.Text('Text Results',font=('Arial',30))],
        [sg.Multiline('',size=(120,16),background_color='white',text_color='black',key='textResult',font=('Arial',20))],
        [sg.Text('Tips: Click the button below to read the text',visible=True,font=('Arial',15),text_color='yellow')],
        [
            sg.pin(sg.Button('',image_filename='icons/play.png',image_subsample=4,font=('Arial',30),auto_size_button=True,key='play_speech',visible=True)),
            sg.pin(sg.Button('',image_filename='icons/pause.png',image_subsample=4,font=('Arial',30),auto_size_button=True,key='pause_speech',visible=False)),
            sg.pin(sg.Button('',image_filename='icons/play.png',image_subsample=4,font=('Arial',30),auto_size_button=True,key='restart_speech',visible=False)),
            sg.Text('  '),
            sg.pin(sg.Button('',image_filename='icons/stop.png',image_subsample=4,font=('Arial',30),auto_size_button=True,key='stop_speech',visible=True))
        ],
        # [sg.Slider(orientation='h',size=(60,5),key='voiceProgress',range=(0,10),default_value=0,enable_events=True)],
        [sg.Slider(orientation='h',size=(35,20),key='volume',range=(0,10),default_value=5,enable_events=True)],
        [sg.Text('Volume',font=('Arial',20))]
    ]

def getImage(): #user upload an image
    path=sg.popup_get_file("Select Image File",file_types=(('Image Files','*.jpg;*.jpeg;*.png;*.bmp'),),font=('Arial',20))

    
    if path==None: #if user close the window without select a file
        window['tips'].Update(visible=True)
        window['upLoading'].Update(visible=False)
        window['fileTypeError'].Update(visible=False)
    elif path.endswith(('.jpg','.png','.bmp','.jpeg')): #if it's a image file, begin ocr and tts
        image_path=path
        result = ocr.tesseract(image_path)
        # print(len(result))
        if len(result)==0: #try PGNet for recognition
            result = ocr.PGNet(image_path)
            print('PGNet has been used')
        elif len(result)<70:
            if text_process.detect_accuracy(result)<0.85:
                result = ocr.PGNet(image_path)
                print('PGNet has been used')
        
        result = text_process.connect_sentence(result) #text process
        
        if spell_correction == True: #auto text spelling correction
            result = text_process.single_spell_check(result)
        elif context_correction ==True:
            result = text_process.contextual_spell_check(result)
        
        with open('files/result.txt','w') as f:
            f.write(result)
            
        if len(result)==0: #if still no text in the image
            window['upLoading'].Update(visible=False)
            window['tips'].Update(visible=True)
            window['fileTypeError'].Update(visible=False)
            window['textResult'].Update("No text found, please select another image!")
        else:
            window['fileTypeError'].Update(visible=False)
            window['tips'].Update(visible=False)
            window['pause_speech'].Update(visible=False)
            window['restart_speech'].Update(visible=False)
            window['play_speech'].Update(visible=True)
            window['upLoading'].Update(visible=True)
            window['textResult'].Update(result)
            tts.gtts(result)
            global ocr_finish
            ocr_finish = True
    else: #if other file type
        window['fileTypeError'].Update(visible=True)
        window['upLoading'].Update(visible=False)
        window['tips'].Update(visible=False)
        
def takeImage(): #user take an image using the camera
    cap=cv2.VideoCapture(0)
    if not cap.isOpened():
        return None
    while True: #loop
        ok, img = cap.read()
        if not ok: break
        
        img2 = img.copy()
        img, angle, success= detect_paper(img)
        cv2.putText(img, 'Tips: Use \"Space\" to take a photo, \"esc\" to exit the camera', (180,40), cv2.FONT_ITALIC, 1, (0, 0, 255),2)
        if success:
            cv2.putText(img, "Angle:{:.0f}".format(90 - angle),(100,80), cv2.FONT_ITALIC,0.65, (225, 190, 0), 2)     
            if angle>75 or angle<25:
                cv2.putText(img, 'Good paper position',(100,100), cv2.FONT_ITALIC,0.65, (225, 190, 0), 2) 
            else:
                cv2.putText(img, 'Paper placement is skewed',(100,100), cv2.FONT_ITALIC,0.65, (225, 190, 0), 2) 
        else:
            cv2.putText(img, 'No paper detected',(100,80), cv2.FONT_ITALIC,0.65, (225, 190, 0), 2) 
                             
        cv2.imshow("Camera", img)
        key = cv2.waitKey(10)
        
        if key == 27: #if user click the esc button
            break
        if key == 32: # if user click the space button
            global take_image
            take_image = True
            cv2.imwrite('files/camera.jpg', img2)
            
    cap.release()
    cv2.destroyAllWindows()
    
    if take_image == True: #if a image has been taken, begin ocr and tts
        image_path='files/camera.jpg'
        result = ocr.tesseract(image_path)
        if len(result)==0: #try PGNet for recognition
            result = ocr.PGNet(image_path)
        elif len(result)<50:
            if text_process.detect_accuracy(result)<0.8:
                result = ocr.PGNet(image_path)
                print('PGNet has been used')
            
        result = text_process.connect_sentence(result) #text process
        
        if spell_correction == True: #auto text spelling correction
            result = text_process.single_spell_check(result)
        elif context_correction ==True:
            result = text_process.contextual_spell_check(result)
        
        if len(result) ==0: #if there is still no text in the image
            window['upLoading'].Update(visible=False)
            window['tips'].Update(visible=True)
            window['fileTypeError'].Update(visible=False)
            window['textResult'].Update("No text found, please try to take another image!")
        else:
            window['fileTypeError'].Update(visible=False)
            window['tips'].Update(visible=False)
            window['pause_speech'].Update(visible=False)
            window['restart_speech'].Update(visible=False)
            window['play_speech'].Update(visible=True)
            window['upLoading'].Update(visible=True)
            window['textResult'].Update(result)
            tts.gtts(result)
            global ocr_finish
            ocr_finish = True
    
    
def playSpeech():
    if ocr_finish ==False: #if ocr has not finished
        sg.popup('   \n No text yet! \n   ',font=('Arial',20),title='')
    else:
        window['play_speech'].Update(visible=False)
        window['pause_speech'].Update(visible=True)
        file_path='files/result.mp3'
        # voice_file = eyed3.load(file_path)
        # voice_length = int(voice_file.info.time_secs)
        # print(voice_length)
        # window['voiceProgress'].Update(range = (0,voice_length)) #voice progress
        pygame.mixer.init()
        pygame.mixer.music.load(file_path)
        pygame.mixer.music.play()
        
        

   
def pauseSpeech():
    pygame.mixer.music.pause()
    window['pause_speech'].Update(visible=False)
    window['restart_speech'].Update(visible=True)
    
def restartSpeech():
    pygame.mixer.music.unpause()
    window['pause_speech'].Update(visible=True)
    window['restart_speech'].Update(visible=False)
    
def stopSpeech():
    pygame.mixer.music.stop()
    window['pause_speech'].Update(visible=False)
    window['restart_speech'].Update(visible=False)
    window['play_speech'].Update(visible=True)
    
def volume(values):
    pygame.mixer.music.set_volume(values*0.1) #volume range from 0 to 1
    
def voiceProgress(values):
    # pygame.mixer.music.play(start=values*0.546)
    pygame.mixer.music.set_pos(values*0.546)
    
    
def detect_paper(img):
    angle = 0
    img2 = img.copy()
    gray = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    edged = cv2.Canny(gray, 75, 200)
    contours = cv2.findContours(edged.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)[0]
    contours = sorted(contours, key = cv2.contourArea, reverse = True)[:5]
    screenCnt = None
    success = False
    for c in contours:
        approx = cv2.approxPolyDP(c, 0.02 * cv2.arcLength(c, True), True)
        if len(approx) == 4:
            screenCnt = approx
            success = True
            angle = cv2.minAreaRect(screenCnt)[2]
            break

    if not success:
        return img,angle,success
    else:
        box = cv2.boxPoints(cv2.minAreaRect(screenCnt))
        if angle>75 or angle<25:
            img = cv2.drawContours(img, [box.astype("int")], -1, (0, 255, 0), 2)
        else:
            img = cv2.drawContours(img, [box.astype("int")], -1, (0, 255, 255), 2)
        return img,angle,success
    
def spellHelp():
    text = 'This function is to automatically correct words that have been incorrectly identified by the OCR engine.\n\nSpell Correction: Automatic correction based on the spelling of single word.\n\nContextual Correction: Automatic word correction based on context. \n\nWarning: Both methods can not guarantee to generate a accurate result, please choose with caution!!!'
    sg.popup(text,title='About Auto Text Correction',font=('Arial',20),button_type=5)

    

    
       
event_callbacks={
    'get_image':getImage,
    'play_speech':playSpeech,
    'pause_speech':pauseSpeech,
    'restart_speech':restartSpeech,
    'stop_speech':stopSpeech,
    'camera':takeImage,
    'help':spellHelp,
}

if __name__ == '__main__':
    main()
    

    
    
