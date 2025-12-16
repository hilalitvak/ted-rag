from fastapi import FastAPI
from pydantic import BaseModel
import os

app = FastAPI()

CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "1024"))
OVERLAP_RATIO = float(os.getenv("OVERLAP_RATIO", "0.2"))
TOP_K = int(os.getenv("TOP_K", "5"))

@app.get("/api/stats")
def stats():
    return {"chunk_size": CHUNK_SIZE, "overlap_ratio": OVERLAP_RATIO, "top_k": TOP_K}

class PromptIn(BaseModel):
    question: str

@app.post("/api/prompt")
def prompt(body: PromptIn):
    return {
        "response": "TODO",
        "context": [],
        "Augmented_prompt": {"System": "TODO", "User": body.question},
    }
