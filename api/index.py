import os
import traceback
from typing import Optional, Tuple, List, Union
import re

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
    def wants_exactly_three_titles(q: str) -> bool:
        q = (q or "").lower()
        return ("exactly 3" in q) and ("title" in q or "titles" in q)

    def is_edu_learning_question(q: str) -> bool:
        q = (q or "").lower()
        return ("education" in q) or ("learning" in q)

    def edu_priority(md: dict, title: str, chunk: str) -> int:
        """
        Higher is better.
        3 = topics contain education/teaching
        2 = title contains learn/learning
        1 = chunk mentions education/learn
        0 = none
        """
        topics = str(md.get("topics", "")).lower()
        t = (title or "").lower()
        c = (chunk or "").lower()

        if ("education" in topics) or ("teaching" in topics):
            return 3
        if ("learn" in t) or ("learning" in t):
            return 2
        if ("education" in c) or ("learn" in c) or ("learning" in c) or ("school" in c):
            return 1
        return 0

    def extract_three_titles(answer: str) -> Optional[List[str]]:
        """
        Return exactly 3 titles as a list[str] WITHOUT inventing.
        If we can't reliably extract exactly 3, return None.
        """
        text = (answer or "").strip()
        if not text:
            return None

        lines = [ln.strip() for ln in text.splitlines() if ln.strip()]

        # If model returned one line containing "1. ... 2. ... 3. ..."
        if len(lines) == 1 and re.search(r"\b1\.\s*.*\b2\.\s*.*\b3\.", lines[0]):
            parts = re.split(r"\s*(?:\d+\.)\s*", lines[0])
            parts = [p.strip() for p in parts if p.strip()]
            lines = parts

        cleaned: List[str] = []
        for ln in lines:
            # Remove leading "1. ", "2. ", "- ", "* " etc.
            ln = re.sub(r"^\s*\d+\.\s*", "", ln).strip()
            ln = re.sub(r"^\s*[-*]\s*", "", ln).strip()
            if ln:
                cleaned.append(ln)

        cleaned = cleaned[:3]
        if len(cleaned) != 3:
            return None
        return cleaned

    try:
        llm, index = get_clients()
        question = (body.question or "").strip()
        if not question:
            raise HTTPException(status_code=400, detail="question_empty")

        # 1) embed question
        q_vec = llm.embeddings.create(model=EMBED_MODEL, input=question).data[0].embedding

        # 2) retrieve more candidates then dedupe by talk_id
        res = index.query(
            vector=q_vec,
            top_k=min(30, max(TOP_K * 4, TOP_K)),
            include_metadata=True,
            namespace=PINECONE_NAMESPACE,
        )

        # collect unique talks
        candidates = []
        seen_talk_ids = set()

        for m in (res.get("matches") or []):
            md = m.get("metadata") or {}
            talk_id = str(md.get("talk_id", "")).strip()
            if not talk_id or talk_id in seen_talk_ids:
                continue
            seen_talk_ids.add(talk_id)

            title = str(md.get("title", "")).strip()
            chunk = md.get("chunk") or md.get("text") or ""
            chunk = str(chunk)

            item = {
                "talk_id": talk_id,
                "title": title,
                "chunk": chunk,
                "score": float(m.get("score", 0.0)),
                "_md": md,  # keep for rerank only (not returned)
            }
            candidates.append(item)

        # Decide which items go into final context
        if wants_exactly_three_titles(question) and is_edu_learning_question(question):
            ranked = sorted(
                candidates,
                key=lambda it: (
                    edu_priority(it["_md"], it["title"], it["chunk"]),
                    it["score"],
                ),
                reverse=True,
            )
            final_context = ranked[:TOP_K]
        else:
            final_context = candidates[:TOP_K]

        # Build ctx blocks (only the returned fields)
        context = []
        ctx_text_blocks = []
        for it in final_context:
            item_out = {
                "talk_id": it["talk_id"],
                "title": it["title"],
                "chunk": it["chunk"],
                "score": it["score"],
            }
            context.append(item_out)
            ctx_text_blocks.append(
                f"talk_id={item_out['talk_id']} | title={item_out['title']} | score={item_out['score']:.4f}\n{item_out['chunk']}"
            )

        user_prompt = (
            "Use ONLY the TED context below to answer.\n"
            "If the answer is not determinable from the context, reply exactly: I don't know based on the provided TED data.\n"
            "If the question asks for exactly 3 talk titles, output ONLY a numbered list of exactly 3 titles (no extra text).\n\n"
            f"Question: {question}\n\n"
            "TED Context:\n"
            + "\n\n---\n\n".join(ctx_text_blocks)
        )

        # Always call the chat model (per assignment output definition)
        chat = llm.chat.completions.create(
            model=CHAT_MODEL,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
        )

        answer = (chat.choices[0].message.content or "").strip()

        # --- FIX: return array for exactly-3-titles request ---
        response_out: Union[str, List[str]]
        if wants_exactly_three_titles(question):
            titles = extract_three_titles(answer)
            if titles is not None:
                response_out = titles  # list[str] in JSON
            else:
                # Don't invent. Fallback to the raw answer string.
                response_out = answer
        else:
            response_out = answer

        return {
            "response": response_out,
            "context": context,
            "Augmented_prompt": {"System": SYSTEM_PROMPT, "User": user_prompt},
        }

    except HTTPException:
        raise
    except Exception:
        print("ERROR in /api/prompt\n", traceback.format_exc())
        raise HTTPException(status_code=500, detail="prompt_failed")
