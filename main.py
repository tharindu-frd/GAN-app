from typing import List
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

# Move imports to the top
import os
import random
import sys
import torch
import google.generativeai as genai
import mediapy as media
from diffusers import AutoPipelineForText2Image

app = FastAPI()

# Define a request body model for the /generateimage endpoint
class ImageRequest(BaseModel):
    text: str

# Define a response model
class ImageResponse(BaseModel):
    images: List[str]

@app.post('/generateimage', response_model=ImageResponse)
def generate_image(request: ImageRequest):
    if os.system('nvidia-smi'):
        raise HTTPException(status_code=500, detail="No GPU found.")
    
    gemini_api_secret_name = 'AIzaSyCZp206SQFV0GnmGPo18-ZzTcKOf_V4ZGM'  # Avoid hardcoding secret names
    
    try:
        GOOGLE_API_KEY = os.getenv(gemini_api_secret_name)
        if not GOOGLE_API_KEY:
            raise HTTPException(status_code=500, detail=f"Secret {gemini_api_secret_name} not found.")
        genai.configure(api_key=GOOGLE_API_KEY)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error configuring Gemini LLM: {str(e)}")
    
    model = genai.GenerativeModel('gemini-pro')

    text = request.text

    prompt = "You are creating a prompt for Stable Diffusion to generate an image. Please generate a text prompt for %s. Only respond with the prompt itself, but embellish it as needed but keep it under 80 tokens." % text
    response = model.generate_content(prompt)
    prompt = response.text
    seed = random.randint(0, sys.maxsize)

    num_inference_steps = 20

    pipe = AutoPipelineForText2Image.from_pretrained(
        "stabilityai/sdxl-turbo",
        torch_dtype=torch.float16,
        use_safetensors=True,
        variant="fp16",
    )

    pipe = pipe.to("cuda")

    images = pipe(
        prompt=prompt,
        guidance_scale=0.0,
        num_inference_steps=num_inference_steps,
        generator=torch.Generator("cuda").manual_seed(seed),
    ).images

    # Convert images to base64 format for transmission
    image_base64 = [media.image_to_base64(image) for image in images]
    
    return {"images": image_base64}

@app.get("/")
def read_root():
    return {"Hello": "World"}