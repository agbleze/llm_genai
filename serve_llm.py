#%%
from flask import Flask, request, jsonify
from transformers import pipeline
import logging

logging.basicConfig(level=logging.DEBUG)

app = Flask(__name__)

try:
    generator = pipeline('text-generation', model='gpt2')
    logging.info("Model loaded successfully.")
except Exception as e:
    logging.error(f"Error occurred while loading the model: {e}")



