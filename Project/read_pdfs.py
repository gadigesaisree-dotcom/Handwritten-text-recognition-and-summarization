import fitz  # PyMuPDF
import os
import glob

pdfs = glob.glob('*.pdf')
for pdf_file in pdfs:
    print(f"\n--- {pdf_file} ---")
    try:
        doc = fitz.open(pdf_file)
        page = doc[0]
        text = page.get_text()
        print(text[:1000])  # Print first 1000 characters
    except Exception as e:
        print(f"Error reading {pdf_file}: {e}")
