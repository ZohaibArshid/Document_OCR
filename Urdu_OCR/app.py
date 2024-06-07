import math
from PIL import Image
import torch
from model import Model
from dataset import NormalizePAD
from utils import CTCLabelConverter, AttnLabelConverter
import os
from fastapi import FastAPI, UploadFile, Depends, HTTPException, Header
import concurrent.futures
import asyncio
app = FastAPI()    
user_api_keys = {
    "user1": "apikey1",
    "user2": "apikey2",
    # Add more users and their API keys as needed
}
# Define a directory to store uploaded videos
upload_dir = "urdu_ocr_uploads"
os.makedirs(upload_dir, exist_ok=True)
# Your model configuration
image_path=f""   
saved_model="best_norm_ED.pth" 
batch_max_length=100 
imgH=32  
imgW=400 
rgb=False
FeatureExtraction="HRNet"
SequenceModeling="DBiLSTM" 
Prediction="CTC" 
num_fiducial=20 
input_channel=1 
output_channel=512
hidden_size=256 
device_id=None
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('device:',device)
if FeatureExtraction == "HRNet":
        output_channel = 32
else:
    output_channel=output_channel
""" 
vocab / character number configuration 
"""
file = open("UrduGlyphs.txt","r",encoding="utf-8")
content = file.readlines()
content = ''.join([str(elem).strip('\n') for elem in content])
character = content+" "
if 'CTC' in Prediction:
    converter = CTCLabelConverter(character)
else:
    converter = AttnLabelConverter(character)
num_class = len(converter.character)
opt=[image_path,saved_model,batch_max_length,imgH,imgW,rgb,FeatureExtraction,SequenceModeling,Prediction,num_fiducial,input_channel,output_channel,hidden_size,device_id,num_class,device]


# Load model
model = Model(opt)
model = model.to(device)

# Load model weights
model.load_state_dict(torch.load(saved_model, map_location=device))
print("urdu ocr loaded!!!")
model.eval()

# Character configuration
file = open("UrduGlyphs.txt", "r", encoding="utf-8")
content = file.readlines()
content = ''.join([str(elem).strip('\n') for elem in content])
character = content + " "

def read_image(image_path):
    """Perform OCR on an image."""
    # Model inference setup
    converter = CTCLabelConverter(character)

    if rgb:
        input_channel = 3
    else:
        input_channel = 1

    # Load and preprocess the image
    img = Image.open(image_path).convert('L')
    img = img.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
    w, h = img.size
    ratio = w / float(h)
    if math.ceil(imgH * ratio) > imgW:
        resized_w = imgW
    else:
        resized_w = math.ceil(imgH * ratio)
    img = img.resize((resized_w, imgH), Image.Resampling.BICUBIC)
    transform = NormalizePAD((1, imgH, imgW))
    img = transform(img)
    img = img.unsqueeze(0)
    img = img.to(device)

    # Model inference
    preds = model(img)
    preds_size = torch.IntTensor([preds.size(1)] * 1)  # Assuming batch size is always 1 for inference
    _, preds_index = preds.max(2)
    preds_str = converter.decode(preds_index.data, preds_size.data)[0]

    return preds_str





def process_file(file: UploadFile):
    try:
        # Read the uploaded file into memory
        file_content = file.file.read()

        # Extract the file extension from the original filename
        file_extension = os.path.splitext(file.filename)[1].lower()

        # Generate a unique filename (e.g., using UUID) or use the original filename
        # Here, I'm using the original filename without the extension
        file_name = os.path.splitext(file.filename)[0]

        # Define the path where the file will be saved
        file_path = os.path.join(upload_dir, f"{file_name}{file_extension}")

       
        # Save the file in the specified directory
        with open(file_path, "wb") as temp_file:
            temp_file.write(file_content)
        # Check if the file already exists
        if not os.path.exists(file_path):
            print(f"File not found/{file_path}")
            return {"error": f"File not found/{file_path}"}
        try:
            # Process the file (e.g., perform OCR)
            result = read_image(image_path=file_path)
            # You can do something with the result here

        except Exception as e:
            return {"error": f"OCR error: {str(e)}"}

        finally:
            # Always try to remove the uploaded file after processing
            try:
                os.remove(file_path)
            except Exception as e:
                return {"error": f"File delete error: {str(e)}"}
        return result

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing file: {str(e)}")

# Dependency to validate the API key
async def get_api_key(api_key: str = Header(None, convert_underscores=False)):
    if api_key not in user_api_keys.values():
        raise HTTPException(status_code=401, detail="Invalid API key")
    return api_key

@app.post("/urdu_ocr/")
async def urdu_ocr_endpoint(
    file: UploadFile,
    api_key: str = Depends(get_api_key),  # Require API key for this route
):
    # Create a new thread for processing each user's video
    with concurrent.futures.ThreadPoolExecutor() as executor:
        result = await asyncio.get_event_loop().run_in_executor(
            executor,
            lambda: process_file(file)
        )
    return result

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=2005, reload=True)
    # uvicorn API:app --host 0.0.0.0 --port 2005 --reload

