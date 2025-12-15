import os
from fastapi import FastAPI
from pydantic import BaseModel

# =========================
# Config (must match /api/stats)
# =========================
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "1024"))
OVERLAP_RATIO = float(os.getenv("OVERLAP_RATIO", "0.2"))
TOP_K = int(os.getenv("TOP_K", "8"))

# =========================
# Required system prompt (keep this content)
# =========================
SYSTEM_PROMPT = (
    "You are a TED Talk assistant that answers questions strictly and only based on the TED dataset "
    "context provided to you (metadata and transcript passages). You must not use any external knowledge, "
    "the open internet, or information that is not explicitly contained in the retrieved context. "
    "If the answer cannot be determined from the provided context, respond: "
    "“I don’t know based on the provided TED data.” Always explain your answer using the given context, "
    "quoting or paraphrasing the relevant transcript or metadata when helpful."
)

app = FastAPI()

class PromptRequest(BaseModel):
    question: str

@app.get("/stats")
def stats():
    # Must return exactly these keys
    return {
        "chunk_size": CHUNK_SIZE,
        "overlap_ratio": OVERLAP_RATIO,
        "top_k": TOP_K,
    }

@app.post("/prompt")
def prompt(req: PromptRequest):
    # Placeholder until we connect Pinecone + models
    context = []
    user_aug = f"Question: {req.question}\n\nContext:\n{context}"

    return {
        "response": "I don’t know based on the provided TED data.",
        "context": context,
        "Augmented_prompt": {
            "System": SYSTEM_PROMPT,
            "User": user_aug
        }
    }
