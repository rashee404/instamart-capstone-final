"""Part D tasks 1-4 - chunking, TF-IDF retrieval, grounded answering and evaluation over data/policies/."""
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from common import POLICIES, RAW
import mock_llm

CHUNK_WORDS, OVERLAP = 40, 10

POLICY_SYSTEM = ("You are a POLICY ASSISTANT for Instamart support. Answer ONLY from the supplied CONTEXT. "
                 "If the context does not contain the answer, reply that you could not find the information "
                 "in the policy documents. Do not use outside knowledge.")


def chunk_words(text, size=CHUNK_WORDS, overlap=OVERLAP):
    """Windows of `size` words, each starting `size - overlap` words after the previous one.
    Stops after the first chunk whose end reaches or passes the last word."""
    words, chunks, start = text.split(), [], 0
    while True:
        chunks.append(" ".join(words[start:start + size]))
        if start + size >= len(words):
            return chunks
        start += size - overlap


class PolicyIndex:
    def __init__(self, size=CHUNK_WORDS, overlap=OVERLAP):
        self.chunks = []                                   # [{"source", "text"}]
        for path in sorted(POLICIES.glob("*.txt")):
            for c in chunk_words(path.read_text(encoding="utf-8"), size, overlap):
                self.chunks.append({"source": path.name, "text": c})
        self.vectorizer = TfidfVectorizer()
        self.matrix = self.vectorizer.fit_transform([c["text"] for c in self.chunks])

    def counts(self):
        return pd.Series([c["source"] for c in self.chunks]).value_counts().sort_index().to_dict()

    def retrieve(self, query, k=3):
        sims = cosine_similarity(self.vectorizer.transform([query]), self.matrix)[0]   # same fitted vectorizer
        top = np.argsort(-sims, kind="stable")[:k]
        return [{"source": self.chunks[i]["source"], "score": round(float(sims[i]), 4),
                 "text": self.chunks[i]["text"]} for i in top]

    def answer(self, question):
        hits = self.retrieve(question, k=3)
        context = "\n".join(f"[{h['source']}]\n{h['text']}" for h in hits)
        messages = [{"role": "system", "content": POLICY_SYSTEM},
                    {"role": "user", "content": f"CONTEXT:\n{context}\nQUESTION:\n{question}"}]
        resp = mock_llm.chat(messages)
        return resp.choices[0].message.content, [h["source"] for h in hits]


_default = None


def default_index():
    global _default
    if _default is None:
        _default = PolicyIndex()
    return _default


def retrieve(query, k=3):
    return default_index().retrieve(query, k)


def answer(question):
    return default_index().answer(question)


def evaluate(index, verbose=True):
    ev = pd.read_csv(RAW / "policy_eval.csv", keep_default_na=False)
    doc_hits = fact_hits = fallback_hits = n_ans = n_unans = 0
    failures = []
    for q in ev.itertuples():
        text, sources = index.answer(q.question)
        if q.expected_doc == "NONE":
            n_unans += 1
            ok = text.startswith("I could not find")
            fallback_hits += ok
            if not ok:
                failures.append((q.question, "no fallback", text, sources))
        else:
            n_ans += 1
            d, f = q.expected_doc in sources, q.expected_fact in text
            doc_hits += d
            fact_hits += f
            if not (d and f):
                failures.append((q.question, f"doc_ok={d} fact_ok={f} expected '{q.expected_fact}'", text, sources))
    res = {"answerable": n_ans, "doc_in_top3": doc_hits, "fact_in_answer": fact_hits,
           "unanswerable": n_unans, "fallback": fallback_hits}
    if verbose:
        print(f"  expected_doc in top-3: {doc_hits}/{n_ans}; expected_fact in answer: {fact_hits}/{n_ans}; "
              f"fallback for unanswerable: {fallback_hits}/{n_unans}")
        for q, why, text, sources in failures:
            print(f"  FAIL: {q}\n        {why}\n        answer: {text}\n        top-3 sources: {sources}")
    return res, failures


if __name__ == "__main__":
    idx = PolicyIndex()
    print("chunks:", len(idx.chunks), idx.counts())
    print("retrieve('refund policy late delivery'):", [(h["source"], h["score"]) for h in idx.retrieve("refund policy late delivery")])
    print("\nevaluation, overlap 10:")
    evaluate(idx)
    idx0 = PolicyIndex(overlap=0)
    print("\nchunks with overlap 0:", len(idx0.chunks), idx0.counts())
    print("evaluation, overlap 0:")
    evaluate(idx0)
