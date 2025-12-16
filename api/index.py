from fastapi import FastAPI
from pydantic import BaseModel
import os
from dotenv import load_dotenv
load_dotenv()
from openai import OpenAI
from pinecone import Pinecone


app = FastAPI()

# ---- RAG hyperparams (must match /api/stats) ----
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "1024"))
OVERLAP_RATIO = float(os.getenv("OVERLAP_RATIO", "0.2"))
TOP_K = int(os.getenv("TOP_K", "5"))

# ---- Models ----
EMBED_MODEL = os.getenv("EMBED_MODEL", "RPRTHPB-text-embedding-3-small")
CHAT_MODEL = os.getenv("CHAT_MODEL", "RPRTHPB-gpt-5-mini")

# ---- Pinecone ----
PINECONE_NAMESPACE = os.getenv("PINECONE_NAMESPACE", "ted")

# ---- Required system prompt section ----
SYSTEM_PROMPT = (
    "You are a TED Talk assistant that answers questions strictly and only based on the TED dataset context provided "
    "(metadata and transcript passages). You must not use any external knowledge, the open internet, or information "
    "that is not explicitly contained in the retrieved context. If the answer cannot be determined from the provided "
    "context, respond: “I don’t know based on the provided TED data.” Always explain your answer using the given context, "
    "quoting or paraphrasing the relevant transcript or metadata when helpful."
)

# ---- Clients (initialized once per runtime) ----
llm = OpenAI(
    api_key=os.environ["LLMOD_API_KEY"],
    base_url=os.environ["LLMOD_BASE_URL"].rstrip("/") + "/v1",
)

pc = Pinecone(api_key=os.environ["PINECONE_API_KEY"])
index = pc.Index(host=os.environ["PINECONE_HOST"])


@app.get("/api/stats")
def stats():
    return {"chunk_size": CHUNK_SIZE, "overlap_ratio": OVERLAP_RATIO, "top_k": TOP_K}


class PromptIn(BaseModel):
    question: str


@app.post("/api/prompt")
def prompt(body: PromptIn):
    question = body.question.strip()

    # 1) Embed the question
    q_vec = llm.embeddings.create(model=EMBED_MODEL, input=question).data[0].embedding

    # 2) Retrieve top_k chunks from Pinecone
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
        if not talk_id:
            continue
        
        # keep at most 1 chunk per talk (ensures distinct talks)
        if talk_id in seen_talk_ids:
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

    # 3) Build augmented user prompt
    user_prompt = (
        "Use ONLY the TED context below to answer.\n"
        "If the answer is not determinable from the context, reply exactly: I don’t know based on the provided TED data.\n\n"
        f"Question: {question}\n\n"
        "TED Context:\n"
        + "\n\n---\n\n".join(ctx_text_blocks)
    )

    # 4) Ask the chat model
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
