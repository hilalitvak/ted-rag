from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI()

class PromptIn(BaseModel):
    question: str

@app.post("/")
def prompt(body: PromptIn):
    return {
        "response": "TODO",
        "context": [],
        "Augmented_prompt": {"System": "TODO", "User": body.question},
    }
