from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
import subprocess

app = FastAPI()

origins = [
    "http://127.0.0.1:5500",
    "http://localhost:5500"
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"], 
)

class TextData(BaseModel):
    text: str

@app.post("/process_text")
def process_text(data: TextData):
    input_text = data.text
    print(f"Input Text: {input_text}")
    
    try:
        
        result1 = subprocess.run(
            ["python3", "./task1/class_labels.py"],
            input=input_text,
            capture_output=True,
            text=True,
            check=True
        )
        
        class_labels = None
        with open('./task1/class_labels.txt', 'r') as file:
            class_labels = file.read()
        
    except subprocess.CalledProcessError as e:
        print(f"Error in task1: {e.stderr}")
        raise HTTPException(status_code=500, detail="Error processing task1.")
    
    try:

        result2 = subprocess.run(
            ["python3", "./task2/entities.py"],
            input=input_text,
            capture_output=True,
            text=True,
            check=True
        )
        entities = None
        with open('./task2/entities.txt', 'r') as file:
            entities = file.read()
        
    except subprocess.CalledProcessError as e:
        print(f"Error in task2: {e.stderr}")
        raise HTTPException(status_code=500, detail="Error processing task2.")
    
    print(entities, class_labels)
    return {"entities": entities, "class_labels": class_labels}
