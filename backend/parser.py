import fitz  # PyMuPDF

def parse_pdf(file) -> str:
    doc = fitz.open(stream=file.read(), filetype="pdf")
    text = ""
    for page in doc:
        text += page.get_text()
    return text.strip()

def parse_txt(file) -> str:
    return file.read().decode("utf-8").strip()
