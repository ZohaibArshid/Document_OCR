import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'
import cv2
from ocr_initializer import initialize_ocr_reader
from UrduOcrCall import Urdu_OCR
import shutil
from config import *
from utils import separate_urdu_english
import re

def remove_special_characters_regex(input_string):
    # Define regex pattern to match special characters
    pattern = r'[^a-zA-Z0-9\s]'
    # Use regex sub() function to replace special characters with an empty string
    clean_string = re.sub(pattern, '', input_string)
    return clean_string
reader=initialize_ocr_reader()
def process_image(input_image,reader=reader):
    
    img= cv2.imread(input_image)
    result = reader.readtext(img, detail=1, paragraph=False)
    final_text=""
    if result:
        english_text, urdu_text,result = separate_urdu_english(result)
        english_text=remove_special_characters_regex(english_text)
        # print("EASYOCR DATA")
        # print('english:',english_text)
        # print('urdu:',urdu_text)
        # print("result:",result)
        if english_text:
            final_text += english_text
        # print('easy ocr language detected:',language)
        # if language=='en':
        #     return text_only
        if urdu_text:
            image_name=os.path.basename(input_image)
            temp_name=0
            urdu_text=""
            directory=os.path.join(BASE_FOLDER,CROPS,image_name)
            os.makedirs(directory, exist_ok=True)
            for i, (bbox, _, _) in enumerate(result):
                (tl, tr, br, bl) = bbox
                tl = (int(tl[0]), int(tl[1]))
                br = (int(br[0]), int(br[1]))
                crop_object =img[tl[1]:br[1], tl[0]:br[0]]
                store_image_path=os.path.join(directory,str(temp_name)+".png")
                cv2.imwrite(store_image_path, crop_object)
                temp_name=temp_name+1
                # print("calling api")
                # print(Urdu_OCR(store_image_path))
                try:
                    urdu_text += " " + Urdu_OCR(store_image_path)
                except:
                    return "Urdu OCR API not live, check it!!!!"
            shutil.rmtree(directory, ignore_errors=True)
        final_text +=urdu_text
        # print("final_text")
        
        return final_text

# file_path=r"TestSample\GrouperTest\18-47-03-312.jpg"
# file_path=r"TestSample\SocialMedia\GEhMxiRbEAAE6rM.jpg"
# # file_path=r"TestSample\Comparing-our-proposed-method-with-the-state-of-the-art-frameworks-on-colorectal-tissue.png"
# print(process_image(file_path))
        