from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from transformers import pipeline
import logging

logging.basicConfig(level=logging.DEBUG)

app = FastAPI()

try:
    generator = pipeline('text-generation', model='gpt2')
    logging.info("Model loaded successfully.")
except Exception as e:
    logging.error(f"Error loading model: {e}")
    raise


class TextGenerationRequest(BaseModel):
    prompt: str
    max_length: int = 50
    num_return_sequences: int = 1
    
class TextGenerationResponse(BaseModel):
    generated_text: str
    

@app.post('/generate', response_model=TextGenerationResponse)
async def generate_text(request: TextGenerationRequest):
    try:
        logging.debug(f"Received request: {request}")
        generated_texts = generator(request.prompt,
                                    max_length=request.max_length,
                                    num_length_sequences=request.num_return_sequences
                                    )
        generated_text = generated_texts[0]['generated_text']
        logging.debug(f"Generated text: {generate_text}")
        return TextGenerationResponse(generate_text=generate_text)
    except Exception as e:
        logging.error(f"Error generating text: {e}")
        raise HTTPException(status_code=500, detail=str(e))