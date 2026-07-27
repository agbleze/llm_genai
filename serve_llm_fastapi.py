#%%
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from transformers import pipeline
import logging


