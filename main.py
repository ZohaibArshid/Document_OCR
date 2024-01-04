import os
import PyPDF2
import pdfplumber
import easyocr
import torch 
from Extension_Converter import convert_to_pdf
from utils import *
from config import *

gpu_available=torch.cuda.is_available()
# print ("GPU Status:",gpu_available)

temp_paths=create_temp_folders(BASE_FOLDER , SUBFOLDERS)
reader = easyocr.Reader(['en'], gpu=gpu_available)
ocr_results_dict = {}

def extract_text_with_pdfplumber(pdf_path):
    with pdfplumber.open(pdf_path) as pdf:
        text = ''
        tables = []
        for page in pdf.pages:
            text += page.extract_text()
            tables += page.extract_tables()
            
    result_dict = {
        'text': text,
        'tables': {}
    }

    non_empty_tables = [table for table in tables if any(any(cell is not None and cell != '' for cell in row) for row in table)]

    for i, table in enumerate(non_empty_tables, start=1):
        table_key = f"Table{i}"
        result_dict['tables'][table_key] = table

    return result_dict

def extract_text_from_pdf(pdf_path, output_folder):
    with open(pdf_path, 'rb') as file:
        pdf_reader = PyPDF2.PdfReader(file)
        for page_num in range(len(pdf_reader.pages)):
            page = pdf_reader.pages[page_num]

            for img_num, image in enumerate(page.images):
                # Generate a unique filename for each image
                image_filename = f"image_{page_num + 1}_{img_num + 1}.png"
                image_path = os.path.join(output_folder, image_filename)

                with open(image_path, 'wb') as f:
                    f.write(image.data)

def image_ocr(output_folder):
    # Check if the output_folder is empty
    if not os.listdir(output_folder):
        print(f"The folder {output_folder} is empty.")
    else:
        # Initialize the OCR reader  # this needs to run only once to load the model into memory
        ocr_results_dict = {}
        # Loop through all files in the output_folder
        for filename in os.listdir(output_folder):
            if filename.endswith(('.jpg', '.jpeg', '.png', '.gif')):  # Add more image extensions if needed
                image_path = os.path.join(output_folder, filename)
                result = reader.readtext(image_path, detail=0)
                if result:
                    key = f"{os.path.splitext(filename)[0]}"
                    ocr_results_dict[key] = result
        return ocr_results_dict

def remove_files_in_folder(folder_path):
    try:
        # Iterate over all files in the folder
        for file_name in os.listdir(folder_path):
            file_path = os.path.join(folder_path, file_name)

            # Check if it is a file (not a subdirectory)
            if os.path.isfile(file_path):
                # Remove the file
                os.remove(file_path)
                print(f"Removed: {file_path}")

        print("All files removed successfully.")

    except Exception as e:
        print(f"Error: {e}")

def process_pdf(file_path, output_folder=temp_paths['temp_pdf']):
    results = extract_text_with_pdfplumber(file_path)
    extract_text_from_pdf(file_path, output_folder)
    ocr_results_dict = image_ocr(output_folder)
    remove_files_in_folder(output_folder)
    results['images'] = ocr_results_dict
    print(results)

def main():
# Main code
    file_path = 'test1.pdf'
    file_extension = file_path.lower().split('.')[-1]

    if file_extension == 'pdf':
        process_pdf(file_path=file_path)
    elif file_extension in IMAGES_TYPE:
        result = reader.readtext(file_path, detail=0)
        if result:
            results = {'text': '', 'tables': {}, 'images': {'image1': result}}
        print("results",results)

    else:
        # Unsupported file type, try to convert to PDF
        converted_pdf_path=os.path.join(BASE_FOLDER,CONVERTED_PDF_PATH)
        convert_to_pdf(file_path, output_file=converted_pdf_path)
        process_pdf(converted_pdf_path)
        os.remove(converted_pdf_path)

if __name__ == "__main__":
    main()