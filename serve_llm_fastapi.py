#%%
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from transformers import pipeline
import logging


logging.basicConfig(level=logging.DEBUG)

app = FastAPI()

try:
    generator = pipeline('text-generation', model='gpt2')


