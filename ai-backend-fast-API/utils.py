from PyPDF2 import PdfReader

def extract_pdf_text(pdf_content):
    reader = PdfReader(pdf_content)
    text = ""
    for page in reader.pages:
        text += page.extract_text()
    return text
