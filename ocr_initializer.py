import easyocr
import torch
def initialize_ocr_reader():
    gpu_available=torch.cuda.is_available()
    print ("GPU Status:",gpu_available)

    reader = easyocr.Reader(['en','ur'], gpu=gpu_available)
    return reader



