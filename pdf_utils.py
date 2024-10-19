import PyPDF2

def extract_text_from_pdf(pdf):
    reader = PyPDF2.PdfReader(pdf.file)
    text = ""
    for page in reader.pages:
        text += page.extract_text()
    return text
