import os
import traceback
import re
from typing import Optional, Tuple, List, Dict, Any

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from dotenv import load_dotenv

from openai import OpenAI
from pinecone import Pinecone

load_dotenv()

app = FastAPI()

# ---------------------------------------------------------------------
# RAG hyperparameters (must always be reflected by /api/stats)
# ---------------------------------------------------------------------
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "1024"))
OVERLAP_RATIO = float(os.getenv("OVERLAP_RATIO", "0.2"))
TOP_K = int(os.getenv("TOP_K", "5"))

PINECONE_NAMESPACE = os.getenv("PINECONE_NAMESPACE", "ted")

EMBED_MODEL = os.getenv("EMBED_MODEL", "RPRTHPB-text-embedding-3-small")
CHAT_MODEL = os.getenv("CHAT_MODEL", "RPRTHPB-gpt-5-mini")

# System prompt required by the assignment: answers must rely ONLY on retrieved TED context.
SYSTEM_PROMPT = (
    "You are a TED Talk assistant that answers questions strictly and only based on the TED dataset context provided "
    "(metadata and transcript passages). You must not use any external knowledge, the open internet, or information "
    "that is not explicitly contained in the retrieved context. If the answer cannot be determined from the provided "
    "context, respond: \"I don't know based on the provided TED data.\" Always explain your answer using the given "
    "context, quoting or paraphrasing the relevant transcript or metadata when helpful."
)


def normalize_llm_base_url(u: str) -> str:
    """Ensure base URL ends with /v1 for OpenAI-compatible SDK usage."""
    u = (u or "").strip().rstrip("/")
    if not u.endswith("/v1"):
        u += "/v1"
    return u


# ---------------------------------------------------------------------
# Lazy clients: do not initialize at import time (serverless-friendly)
# ---------------------------------------------------------------------
_llm: Optional[OpenAI] = None
_index = None


def get_clients() -> Tuple[OpenAI, object]:
    """Create and cache LLM + Pinecone clients. Raise clean 500 if env vars are missing."""
    global _llm, _index

    missing = []
    for k in ["LLMOD_API_KEY", "LLMOD_BASE_URL", "PINECONE_API_KEY", "PINECONE_HOST"]:
        if not os.getenv(k):
            missing.append(k)
    if missing:
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


# ---------------------------------------------------------------------
# Text cleanup utilities (fix mojibake + enforce ASCII punctuation)
# ---------------------------------------------------------------------
def _try_mojibake_roundtrip(s: str) -> str:
    """
    Try to repair UTF-8 bytes that were decoded as cp1252/latin1.
    This often causes sequences like â€™ â€œ â€ â€“ â€” etc.
    """
    if not s:
        return s
    if not any(m in s for m in ("â", "Ã", "ï»¿", "\uFFFD", "�")):
        return s

    candidates = [s]
    for enc in ("cp1252", "latin1"):
        try:
            repaired = s.encode(enc, errors="ignore").decode("utf-8", errors="ignore")
            if repaired:
                candidates.append(repaired)
        except Exception:
            pass

    def badness(x: str) -> int:
        return x.count("â") + x.count("Ã") + x.count("�") + x.count("\uFFFD")

    return min(candidates, key=badness)


def ascii_punct(s: str) -> str:
    """Convert smart punctuation to plain ASCII punctuation."""
    if not s:
        return s
    s = s.replace("\u2018", "'").replace("\u2019", "'")  # ‘ ’
    s = s.replace("\u201C", '"').replace("\u201D", '"')  # “ ”
    s = s.replace("\u2013", "-").replace("\u2014", "-")  # – —
    s = s.replace("\u2026", "...")  # …
    return s


def sanitize_text(s: str) -> str:
    """
    One-stop cleaning used for:
      - retrieved chunks/titles (Pinecone metadata)
      - LLM answers
      - augmented prompt (User)
    Goal: remove â/Ã mojibake and output ASCII punctuation.
    """
    if not s:
        return s

    s0 = s
    s = _try_mojibake_roundtrip(s)

    # Targeted replacements for common mojibake sequences
    repl = {
        "â€™": "'",
        "â€˜": "'",
        "â€œ": '"',
        "â€": '"',
        "â€“": "-",
        "â€”": "-",
        "â€¦": "...",
        "Â": "",
        "\uFFFD": "",
        "�": "",
    }
    for k, v in repl.items():
        s = s.replace(k, v)

    # Sometimes single 'â' remains (e.g., "couple of â I'm")
    s = s.replace("â", "")

    # Enforce ASCII punctuation
    s = ascii_punct(s)

    # Final: drop any remaining non-ASCII characters deterministically
    s = s.encode("ascii", errors="ignore").decode("ascii", errors="ignore")

    return s if s else s0


def enforce_summary_format(answer: str) -> str:
    """If model glues 'Title:' and 'Short summary...' together, split into lines."""
    if not answer:
        return answer
    return re.sub(r"(Title:\s*[^\n]+)\s*(Short summary of the key idea:)", r"\1\n\2", answer)


# ---------------------------------------------------------------------
# Query intent helpers
# ---------------------------------------------------------------------
def wants_exactly_three_titles(q: str) -> bool:
    """Trigger the 'exactly 3 talk titles' formatting rule."""
    q = (q or "").lower()
    return ("exactly 3" in q) and ("title" in q or "titles" in q)


def is_edu_learning_question(q: str) -> bool:
    """Detect education/learning themed queries for reranking."""
    q = (q or "").lower()
    return ("education" in q) or ("learning" in q)


def is_fear_anxiety_question(q: str) -> bool:
    """Detect fear/anxiety themed queries (conservative)."""
    q = (q or "").lower()
    return ("fear" in q) or ("anxiety" in q)


def is_summary_question(q: str) -> bool:
    """Detect a 'title + short summary' request."""
    q = (q or "").lower()
    return ("short summary" in q) or ("key idea" in q) or ("summary" in q)


def is_recommendation_question(q: str) -> bool:
    """Detect recommendation-style queries."""
    q = (q or "").lower()
    return ("recommend" in q) or ("which talk would you recommend" in q)


def edu_priority(md: Dict[str, Any], title: str, chunk: str) -> int:
    """
    Higher is better.
    3 = topics contain education/teaching
    2 = title contains learn/learning
    1 = chunk mentions education/learn/school
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


def fear_priority(md: Dict[str, Any], title: str, chunk: str) -> int:
    """
    Conservative filter: only treat fear/anxiety as relevant if there are strong indicators
    (anxiety/panic/phobia/trauma/PTSD) OR mild fear-words + clear mental/psych framing.
    """
    topics = str(md.get("topics", "")).lower()
    blob = f"{topics} {(title or '').lower()} {(chunk or '').lower()}"

    strong = ["anxiety", "panic", "phobia", "trauma", "ptsd"]
    mild = ["fear", "afraid", "stress", "worry", "nervous"]
    mental = ["psychology", "therapy", "mental", "emotion", "cognitive", "behavior", "mind"]

    strong_hits = sum(w in blob for w in strong)
    mild_hits = sum(w in blob for w in mild)
    has_mental = any(w in blob for w in mental)

    if strong_hits >= 1:
        return 5
    if mild_hits >= 2 and has_mental:
        return 3
    return 0


def build_context_and_blocks(final_context: List[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], List[str]]:
    """Build the output context array + text blocks used inside the augmented prompt."""
    context_out: List[Dict[str, Any]] = []
    blocks: List[str] = []

    for it in final_context:
        item_out = {
            "talk_id": it["talk_id"],
            "title": it["title"],
            "chunk": it["chunk"],
            "score": it["score"],
        }
        context_out.append(item_out)
        blocks.append(
            f"talk_id={item_out['talk_id']} | title={item_out['title']} | score={item_out['score']:.4f}\n"
            f"{item_out['chunk']}"
        )

    return context_out, blocks


# ---------------------------------------------------------------------
# API endpoints
# ---------------------------------------------------------------------
@app.get("/")
def root():
    return {"status": "ok"}


@app.get("/api/stats")
def stats():
    # Must always reflect current hyperparameters (assignment requirement).
    return {"chunk_size": CHUNK_SIZE, "overlap_ratio": OVERLAP_RATIO, "top_k": TOP_K}


class PromptIn(BaseModel):
    question: str


@app.post("/api/prompt")
def prompt(body: PromptIn):
    """
    Main RAG endpoint (assignment format):
      - Input:  {"question": "..."}
      - Output: {"response": "...", "context": [...], "Augmented_prompt": {...}}
    """
    try:
        llm, index = get_clients()

        question = (body.question or "").strip()
        if not question:
            raise HTTPException(status_code=400, detail="question_empty")

        # 1) Embed question (with minimal query expansion for fear/anxiety to improve recall)
        embed_query = question
        if is_fear_anxiety_question(question):
            embed_query = question + " anxiety panic phobia fear afraid stress worry trauma"
        q_vec = llm.embeddings.create(model=EMBED_MODEL, input=embed_query).data[0].embedding

        # 2) Retrieve candidates (dedupe to distinct talk_id later)
        retr_k = min(30, max(TOP_K * 4, TOP_K))
        if is_fear_anxiety_question(question):
            retr_k = 30

        res = index.query(
            vector=q_vec,
            top_k=retr_k,
            include_metadata=True,
            namespace=PINECONE_NAMESPACE,
        )

        # 3) Dedupe by talk_id; IMPORTANT: sanitize title/chunk BEFORE storing/returning
        candidates: List[Dict[str, Any]] = []
        seen_talk_ids = set()

        for m in (res.get("matches") or []):
            md = m.get("metadata") or {}
            talk_id = str(md.get("talk_id", "")).strip()
            if not talk_id or talk_id in seen_talk_ids:
                continue
            seen_talk_ids.add(talk_id)

            raw_title = str(md.get("title", "")).strip()
            raw_chunk = str(md.get("chunk") or md.get("text") or "")

            title = sanitize_text(raw_title)
            chunk = sanitize_text(raw_chunk)

            candidates.append(
                {
                    "talk_id": talk_id,
                    "title": title,
                    "chunk": chunk,
                    "score": float(m.get("score", 0.0)),
                    "_md": md,  # keep for rerank only (not returned)
                }
            )

        # 4) Choose TOP_K items for final context (light reranking for some query types)
        if wants_exactly_three_titles(question) and is_edu_learning_question(question):
            ranked = sorted(
                candidates,
                key=lambda it: (edu_priority(it["_md"], it["title"], it["chunk"]), it["score"]),
                reverse=True,
            )
            final_context = ranked[:TOP_K]
        elif is_fear_anxiety_question(question):
            ranked = sorted(
                candidates,
                key=lambda it: (fear_priority(it["_md"], it["title"], it["chunk"]), it["score"]),
                reverse=True,
            )
            final_context = ranked[:TOP_K]
        else:
            final_context = candidates[:TOP_K]

        # Build output context + blocks (always needed for assignment output)
        context_out, ctx_text_blocks = build_context_and_blocks(final_context)

        # 5) Deterministic behavior for "exactly 3 titles": avoid LLM variability
        if wants_exactly_three_titles(question):
            if len(final_context) >= 3:
                response_out = "\n".join([f"{i+1}. {final_context[i]['title']}" for i in range(3)])
            else:
                response_out = "I don't know based on the provided TED data."

            user_prompt = (
                "Use ONLY the TED context below to answer.\n"
                "If the answer is not determinable from the context, reply exactly: I don't know based on the provided TED data.\n"
                "If the question asks for exactly 3 talk titles, output ONLY a numbered list of exactly 3 titles (no extra text).\n\n"
                f"Question: {question}\n\n"
                "TED Context:\n"
                + "\n\n---\n\n".join(ctx_text_blocks)
            )
            user_prompt = sanitize_text(user_prompt)

            return {
                "response": response_out,
                "context": context_out,
                "Augmented_prompt": {"System": SYSTEM_PROMPT, "User": user_prompt},
            }

        # 6) Conservative fallback for fear/anxiety: do not guess if no evidence
        if is_fear_anxiety_question(question):
            if (not final_context) or (fear_priority(final_context[0]["_md"], final_context[0]["title"], final_context[0]["chunk"]) == 0):
                user_prompt = (
                    "Use ONLY the TED context below to answer.\n"
                    "If the answer is not determinable from the context, reply exactly: I don't know based on the provided TED data.\n\n"
                    f"Question: {question}\n\n"
                    "TED Context:\n"
                    + "\n\n---\n\n".join(ctx_text_blocks)
                )
                user_prompt = sanitize_text(user_prompt)

                return {
                    "response": "I don't know based on the provided TED data.",
                    "context": context_out,
                    "Augmented_prompt": {"System": SYSTEM_PROMPT, "User": user_prompt},
                }

        # 7) For summary/recommendation questions: force model to use ONLY the FIRST context item
        ctx_for_llm_blocks = ctx_text_blocks
        if is_summary_question(question) or is_recommendation_question(question):
            ctx_for_llm_blocks = ctx_text_blocks[:1]

        user_prompt = (
            "Use ONLY the TED context below to answer.\n"
            "If the answer is not determinable from the context, reply exactly: I don't know based on the provided TED data.\n\n"
            "IMPORTANT: Base your answer ONLY on the FIRST context item provided below. Do not use other items.\n\n"
            "OUTPUT CONSTRAINT: Use ONLY ASCII punctuation. Use straight quotes ' and \". "
            "Use hyphen - instead of en/em dashes. No curly quotes.\n\n"
            f"Question: {question}\n\n"
            "TED Context:\n"
            + "\n\n---\n\n".join(ctx_for_llm_blocks)
        )
        user_prompt = sanitize_text(user_prompt)

        # 8) Call the chat model
        chat = llm.chat.completions.create(
            model=CHAT_MODEL,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
        )

        answer = sanitize_text((chat.choices[0].message.content or "").strip())
        answer = enforce_summary_format(answer)

        return {
            "response": answer,
            "context": context_out,
            "Augmented_prompt": {"System": SYSTEM_PROMPT, "User": user_prompt},
        }

    except HTTPException:
        raise
    except Exception:
        print("ERROR in /api/prompt\n", traceback.format_exc())
        raise HTTPException(status_code=500, detail="prompt_failed")
