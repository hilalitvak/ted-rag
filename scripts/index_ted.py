import os
import json
import math
import pandas as pd
from pathlib import Path
from dotenv import load_dotenv
from pinecone import Pinecone

# -------- Config (must match assignment limits) --------
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "1024"))         # <= 2048
OVERLAP_RATIO = float(os.getenv("OVERLAP_RATIO", "0.2"))  # <= 0.3
TOP_K = int(os.getenv("TOP_K", "8"))                      # <= 30

DATA_PATH = Path("data/ted_talks_en.csv")
STATE_PATH = Path("scripts/index_state.json")
NAMESPACE = os.getenv("PINECONE_NAMESPACE", "ted")

load_dotenv()

# -------- Helpers --------
def chunk_text(text: str, chunk_size: int, overlap_ratio: float):
    text = (text or "").strip()
    if not text:
        return []
    overlap = int(chunk_size * overlap_ratio)
    step = max(1, chunk_size - overlap)
    chunks = []
    for start in range(0, len(text), step):
        end = start + chunk_size
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        if end >= len(text):
            break
    return chunks

def load_state():
    if STATE_PATH.exists():
        return json.loads(STATE_PATH.read_text(encoding="utf-8"))
    return {"indexed_talk_ids": []}

def save_state(state):
    STATE_PATH.write_text(json.dumps(state, indent=2), encoding="utf-8")

# -------- TODO: embeddings (you will fill this with the course model call) --------
def embed_texts(texts):
    """
    Must return List[List[float]] with dim=1536 using:
    RPRTHPB-text-embedding-3-small
    """
    raise NotImplementedError("Hook your embedding API call here.")

def main():
    # 1) Load CSV
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Missing dataset at {DATA_PATH.resolve()}")

    df = pd.read_csv(DATA_PATH)
    if "talk_id" not in df.columns or "transcript" not in df.columns:
        raise ValueError("CSV must contain talk_id and transcript columns.")

    # 2) Pinecone connect
    pc = Pinecone(api_key=os.environ["PINECONE_API_KEY"])
    index = pc.Index(os.environ["PINECONE_INDEX_NAME"])

    # 3) Load state (avoid re-embedding)
    state = load_state()
    indexed = set(state["indexed_talk_ids"])

    # 4) Start small first (budget!)
    LIMIT = int(os.getenv("INDEX_LIMIT", "200"))  # start with 200, then increase
    to_process = df[~df["talk_id"].isin(indexed)].head(LIMIT)

    print(f"[INFO] Will process {len(to_process)} talks (LIMIT={LIMIT}).")

    upserts = []
    for _, row in to_process.iterrows():
        talk_id = str(row["talk_id"])
        title = str(row.get("title", ""))
        speaker = str(row.get("speaker_1", ""))
        url = str(row.get("url", ""))
        topics = str(row.get("topics", ""))

        transcript = str(row.get("transcript", ""))

        # Build a combined text so retrieval can use metadata too
        meta_text = f"Title: {title}\nSpeaker: {speaker}\nTopics: {topics}\nURL: {url}\n"
        full_text = meta_text + "\nTranscript:\n" + transcript

        chunks = chunk_text(full_text, CHUNK_SIZE, OVERLAP_RATIO)
        if not chunks:
            indexed.add(talk_id)
            continue

        # IMPORTANT: embed in batches (we'll batch after building items)
        for ci, ch in enumerate(chunks):
            vid = f"{talk_id}:{ci}"
            upserts.append((vid, ch, {
                "talk_id": talk_id,
                "title": title,
                "speaker_1": speaker,
                "url": url,
                "topics": topics,
                "chunk_idx": ci,
                "text": ch
            }))

    print(f"[INFO] Prepared {len(upserts)} chunks for embedding+upsert.")

    # 5) Embed + upsert (batch)
    BATCH = int(os.getenv("BATCH_SIZE", "64"))
    for i in range(0, len(upserts), BATCH):
        batch = upserts[i:i+BATCH]
        texts = [x[1] for x in batch]

        vectors = embed_texts(texts)  # <-- you implement
        pine_vecs = []
        for (vid, _text, md), vec in zip(batch, vectors):
            pine_vecs.append({"id": vid, "values": vec, "metadata": md})

        index.upsert(vectors=pine_vecs, namespace=NAMESPACE)
        print(f"[UPSERT] {i+len(batch)}/{len(upserts)}")

    # 6) Update state: mark processed talks as indexed
    for _, row in to_process.iterrows():
        indexed.add(str(row["talk_id"]))

    state["indexed_talk_ids"] = sorted(indexed)
    save_state(state)
    print("[DONE] Indexing batch complete.")

if __name__ == "__main__":
    main()
