import requests
import json
import os
from dotenv import load_dotenv

load_dotenv()

OLLAMA_URL = "http://localhost:11434/api/generate"
OLLAMA_MODEL = os.environ['OLLAMA_MODEL']

def generate_with_ollama(prompt: str) -> str:

    payload = {'model' : OLLAMA_MODEL, 'prompt' : prompt, 'stream' : False}

    res = requests.post(OLLAMA_URL, json=payload)
    
    if res.status_code == 200:
        data = res.json()
        return data.get('response', "").strip()
    else:
        return f"error occured {res.status_code}"