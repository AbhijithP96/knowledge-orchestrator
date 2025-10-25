import PyPDF2
from io import BytesIO
import re

def parse_content(content):

    try:
        reader = PyPDF2.PdfReader(BytesIO(content))
        text = ''
        for page in reader.pages:
            text += page.extract_text()

        return text
    except:
        return content.decode('utf-8', errors = 'ignore')
    
def chunk_text(text, chunk_size=100): #fixed size chunking
    chunks = []

    for i in range(0, len(text), chunk_size):
        chunk = ''.join(text[i:i+chunk_size]).strip()
        if chunk:
            chunks.append(chunk)

    return chunks