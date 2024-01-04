import magic
import PyPDF2
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
from PIL import Image
import docx2pdf
import openpyxl
import docx
import textract
import io
import tempfile
import os

def convert_to_pdf(input_files, output_file):
    for input_file in input_files:
        mime = magic.Magic(mime=True)
        file_type = mime.from_file(input_file)

        if "application/msword" in file_type or "application/vnd.openxmlformats-officedocument.wordprocessingml.document" in file_type:
            # Convert Word documents (.doc, .docx) to PDF
            if input_file.endswith('.docx'):
                doc = docx.Document(input_file)

                # Save the document to a temporary file
                temp_docx_path = tempfile.mktemp(suffix=".docx")
                doc.save(temp_docx_path)

                # Convert the temporary document to PDF
                docx2pdf.convert(temp_docx_path, output_file)

                # Delete the temporary document
                os.remove(temp_docx_path)

            elif input_file.endswith('.doc'):
                # Handle .doc files using python-docx or another method
                pass
        elif "text/plain" in file_type or "text/rtf" in file_type or "application/vnd.oasis.opendocument.text" in file_type:
            with open(input_file, 'r', encoding='utf-8') as text_file:
                content = text_file.read()
                c = canvas.Canvas(output_file, pagesize=letter)
                c.drawString(100, 750, content)
                c.save()
        elif "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet" in file_type:
            # Convert Excel spreadsheets (.xlsx) to PDF
            wb = openpyxl.load_workbook(input_file)
            for sheet in wb.worksheets:
                pdf = io.BytesIO()
                sheet.save(pdf, format='pdf')
                with open(output_file, 'wb') as outfile:
                    outfile.write(pdf.getvalue())
        elif "image" in file_type:
            # Convert image file to PDF using PIL
            image = Image.open(input_file)
            pdf_path = output_file
            image.save(pdf_path, "PDF", resolution=100.0)
        elif "pdf" in file_type:
            # If it's already a PDF, just copy it
            with open(input_file, 'rb') as pdf_file:
                reader = PyPDF2.PdfReader(pdf_file)
                writer = PyPDF2.PdfWriter()
                for page in reader.pages:
                    writer.add_page(page)
                with open(output_file, 'wb') as output_pdf:
                    writer.write(output_pdf)
        else:
            # Handle other file types accordingly
            pass

# Example usage
# file_paths = ['test2.docx']
# output_path = 'combined_files.pdf'
# convert_to_pdf(file_paths, output_path)
