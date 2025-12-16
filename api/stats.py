from fastapi import FastAPI
import os

app = FastAPI()

CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "1024"))
OVERLAP_RATIO = float(os.getenv("OVERLAP_RATIO", "0.2"))
TOP_K = int(os.getenv("TOP_K", "5"))

@app.get("/")
def stats():
    return {"chunk_size": CHUNK_SIZE, "overlap_ratio": OVERLAP_RATIO, "top_k": TOP_K}
