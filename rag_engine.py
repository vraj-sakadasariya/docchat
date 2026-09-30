"""
rag_engine.py

Handles document parsing, chunking, embedding, and answering questions using Groq.
The approach is basically RAG - split the doc into chunks, find the relevant ones
for a given question, then let the LLM answer using only those chunks.
"""

import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["TRANSFORMERS_VERBOSITY"] = "error"

import io
import re
import numpy as np
import PyPDF2
import docx
from sentence_transformers import SentenceTransformer
from groq import Groq


# ─── Constants ────────────────────────────────────────────────────────────────
EMBED_MODEL = "all-MiniLM-L6-v2"   # Small, fast, good quality
LLM_MODEL   = "openai/gpt-oss-20b"   # Updated: llama3-8b-8192 was decommissioned
CHUNK_SIZE  = 400                   # Words per chunk
CHUNK_OVERLAP = 60                  # Overlap to avoid losing context at edges
TOP_K       = 4                     # Number of chunks to retrieve per query


# ─── RAGEngine Class ──────────────────────────────────────────────────────────
class RAGEngine:
    """
    Full RAG pipeline in one class.

    Usage:
        engine = RAGEngine(api_key="gsk_...")
        engine.load_document(file_bytes, "report.pdf")
        answer, sources = engine.ask("What is the main conclusion?")
    """

    def __init__(self, api_key: str):
        self.groq_client  = Groq(api_key=api_key)
        self.embed_model  = SentenceTransformer(EMBED_MODEL)
        self.chunks: list[str]     = []
        self.embeddings: np.ndarray = None
        self.doc_name: str         = ""
        self.is_ready: bool        = False

    # ── Step 1: Parse ─────────────────────────────────────────────────────────
    def _parse_pdf(self, file_bytes: bytes) -> str:
        reader = PyPDF2.PdfReader(io.BytesIO(file_bytes))
        pages  = [page.extract_text() or "" for page in reader.pages]
        return "\n\n".join(pages)

    def _parse_docx(self, file_bytes: bytes) -> str:
        doc   = docx.Document(io.BytesIO(file_bytes))
        paras = [p.text for p in doc.paragraphs if p.text.strip()]
        return "\n\n".join(paras)

    def _parse_txt(self, file_bytes: bytes) -> str:
        return file_bytes.decode("utf-8", errors="ignore")

    def _parse(self, file_bytes: bytes, file_name: str) -> str:
        ext = file_name.rsplit(".", 1)[-1].lower()
        if ext == "pdf":
            return self._parse_pdf(file_bytes)
        elif ext == "docx":
            return self._parse_docx(file_bytes)
        elif ext == "txt":
            return self._parse_txt(file_bytes)
        else:
            raise ValueError(f"Unsupported file type: .{ext}")

    # ── Step 2: Chunk ─────────────────────────────────────────────────────────
    def _chunk(self, text: str) -> list[str]:
        """Split text into overlapping word-based chunks."""
        # Normalise whitespace
        text  = re.sub(r"\s+", " ", text).strip()
        words = text.split()

        chunks = []
        step   = CHUNK_SIZE - CHUNK_OVERLAP
        for i in range(0, len(words), step):
            chunk = " ".join(words[i : i + CHUNK_SIZE])
            if len(chunk.split()) >= 20:           # Skip tiny fragments
                chunks.append(chunk)
        return chunks

    # ── Step 3: Embed ─────────────────────────────────────────────────────────
    def _embed(self, texts: list[str]) -> np.ndarray:
        return self.embed_model.encode(texts, show_progress_bar=False)

    # ── Public: Load Document ─────────────────────────────────────────────────
    def load_document(self, file_bytes: bytes, file_name: str) -> dict:
        """
        Parse → Chunk → Embed the uploaded document.
        Returns stats dict for the UI.
        """
        raw_text       = self._parse(file_bytes, file_name)
        self.chunks    = self._chunk(raw_text)
        self.embeddings = self._embed(self.chunks)
        self.doc_name  = file_name
        self.is_ready  = True

        return {
            "doc_name"  : file_name,
            "characters": len(raw_text),
            "chunks"    : len(self.chunks),
            "words"     : len(raw_text.split()),
        }

    # ── Step 4: Retrieve ──────────────────────────────────────────────────────
    def retrieve(self, query: str, top_k: int = TOP_K) -> list[tuple[str, float]]:
        """
        Embed the query, compute cosine similarity against all chunk embeddings,
        and return the top-k most relevant (chunk, score) pairs.
        """
        if not self.is_ready:
            raise RuntimeError("No document loaded yet.")

        q_emb = self._embed([query])          # shape (1, dim)

        # Cosine similarity = dot product of unit vectors
        chunk_norms = np.linalg.norm(self.embeddings, axis=1, keepdims=True) + 1e-10
        q_norm      = np.linalg.norm(q_emb) + 1e-10
        normed_chunks = self.embeddings / chunk_norms
        normed_q      = q_emb / q_norm

        scores  = (normed_chunks @ normed_q.T).flatten()   # (n_chunks,)
        top_idx = np.argsort(scores)[::-1][:top_k]

        return [(self.chunks[i], float(scores[i])) for i in top_idx]

    # ── Step 5: Generate ──────────────────────────────────────────────────────
    def ask(self, query: str, chat_history: list[dict] = None):
        """
        Full RAG pipeline:
          Retrieve top-k chunks → Build grounded prompt → Stream LLM response.

        Returns:
            stream   : Groq streaming response (iterate to get tokens)
            sources  : list of (chunk_text, similarity_score) tuples
        """
        sources = self.retrieve(query)
        context = "\n\n---\n\n".join(
            [f"[Excerpt {i+1}]\n{chunk}" for i, (chunk, _) in enumerate(sources)]
        )

        system_prompt = f"""You are DocChat, an AI assistant that answers questions STRICTLY based on the provided document excerpts.

Rules you MUST follow:
1. Only use information that appears in the excerpts below.
2. If the answer is not found in the excerpts, respond with:
   "I couldn't find information about that in this document."
3. Do NOT use outside knowledge or make assumptions.
4. Be concise, clear, and helpful.
5. When relevant, quote short phrases from the document to support your answer.

Document: "{self.doc_name}"

Relevant excerpts from the document:
{context}"""

        messages = [{"role": "system", "content": system_prompt}]

        # Inject previous turns (last 6 to keep context window small)
        if chat_history:
            messages.extend(chat_history[-6:])

        messages.append({"role": "user", "content": query})

        stream = self.groq_client.chat.completions.create(
            model      = LLM_MODEL,
            messages   = messages,
            stream     = True,
            max_tokens = 1024,
            temperature= 0.2,     # Lower = more factual
        )

        return stream, sources
