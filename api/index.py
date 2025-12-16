import os
import traceback
from typing import Optional, Tuple

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from dotenv import load_dotenv

from openai import OpenAI
from pinecone import Pinecone

load_dotenv()

app = FastAPI()

# --- RAG hyperparams (stats must always reflect these) ---
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "1024"))
OVERLAP_RATIO = float(os.getenv("OVERLAP_RATIO", "0.2"))
TOP_K = int(os.getenv("TOP_K", "5"))

PINECONE_NAMESPACE = os.getenv("PINECONE_NAMESPACE", "ted")

EMBED_MODEL = os.getenv("EMBED_MODEL", "RPRTHPB-text-embedding-3-small")
CHAT_MODEL = os.getenv("CHAT_MODEL", "RPRTHPB-gpt-5-mini")

SYSTEM_PROMPT = (
    "You are a TED Talk assistant that answers questions strictly and only based on the TED dataset context provided "
    "(metadata and transcript passages). You must not use any external knowledge, the open internet, or information "
    "that is not explicitly contained in the retrieved context. If the answer cannot be determined from the provided "
    "context, respond: \"I don't know based on the provided TED data.\" Always explain your answer using the given "
    "context, quoting or paraphrasing the relevant transcript or metadata when helpful."
)

def normalize_llm_base_url(u: str) -> str:
    u = (u or "").strip().rstrip("/")
    if not u.endswith("/v1"):
        u += "/v1"
    return u

# --- Lazy clients (DON'T create at import time) ---
_llm: Optional[OpenAI] = None
_index = None

def get_clients() -> Tuple[OpenAI, object]:
    global _llm, _index

    missing = []
    for k in ["LLMOD_API_KEY", "LLMOD_BASE_URL", "PINECONE_API_KEY", "PINECONE_HOST"]:
        if not os.getenv(k):
            missing.append(k)
    if missing:
        # Don't crash the function; return a clean 500 with actionable detail
        raise HTTPException(status_code=500, detail=f"missing_env_vars: {', '.join(missing)}")

    if _llm is None:
        _llm = OpenAI(
            api_key=os.environ["LLMOD_API_KEY"],
            base_url=normalize_llm_base_url(os.environ["LLMOD_BASE_URL"]),
        )

    if _index is None:
        pc = Pinecone(api_key=os.environ["PINECONE_API_KEY"])
        _index = pc.Index(host=os.environ["PINECONE_HOST"])

    return _llm, _index

@app.get("/")
def root():
    return {"status": "ok"}

@app.get("/api/stats")
def stats():
    return {
        "chunk_size": CHUNK_SIZE,
        "overlap_ratio": OVERLAP_RATIO,
        "top_k": TOP_K
    }

class PromptIn(BaseModel):
    question: str

@app.post("/api/prompt")
def prompt(body: PromptIn):
    try:
        llm, index = get_clients()
        question = body.question.strip()

        # 1) embed question
        q_vec = llm.embeddings.create(model=EMBED_MODEL, input=question).data[0].embedding

        # 2) retrieve more candidates then dedupe by talk_id
        res = index.query(
            vector=q_vec,
            top_k=min(30, max(TOP_K * 4, TOP_K)),
            include_metadata=True,
            namespace=PINECONE_NAMESPACE,
        )

        context = []
        ctx_text_blocks = []
        seen_talk_ids = set()

        for m in (res.get("matches") or []):
            md = m.get("metadata") or {}
            talk_id = md.get("talk_id", "")
            if not talk_id or talk_id in seen_talk_ids:
                continue
            seen_talk_ids.add(talk_id)

            chunk = md.get("chunk", "")
            item = {
                "talk_id": talk_id,
                "title": md.get("title", ""),
                "chunk": chunk,
                "score": float(m.get("score", 0.0)),
            }
            context.append(item)
            ctx_text_blocks.append(
                f"talk_id={item['talk_id']} | title={item['title']} | score={item['score']:.4f}\n{chunk}"
            )
            if len(context) >= TOP_K:
                break

        user_prompt = (
            "Use ONLY the TED context below to answer.\n"
            "If the answer is not determinable from the context, reply exactly: I don't know based on the provided TED data.\n\n"
            f"Question: {question}\n\n"
            "TED Context:\n"
            + "\n\n---\n\n".join(ctx_text_blocks)
        )

        # gpt-5-mini: don't send temperature
        chat = llm.chat.completions.create(
            model=CHAT_MODEL,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
        )

        answer = chat.choices[0].message.content

        return {
            "response": answer,
            "context": context,
            "Augmented_prompt": {"System": SYSTEM_PROMPT, "User": user_prompt},
        }

    except HTTPException:
        raise
    except Exception:
        print("ERROR in /api/prompt\n", traceback.format_exc())
        raise HTTPException(status_code=500, detail="prompt_failed")
