from typing import List, Tuple
from generator import Doc

def retrieve_topk(question: str, docs: List[Doc], top_k: int) -> List[Doc]:
    """
    Простой retrieval (keyword overlap).
    Достаточно реалистично для студенческого стенда, но не усложняет код.
    """
    q = question.lower()
    q_words = [w for w in q.replace("«"," ").replace("»"," ").replace("?"," ").split() if len(w) > 2]

    scored: List[Tuple[int, Doc]] = []
    for d in docs:
        t = d.text.lower()
        score = sum(1 for w in q_words if w in t)
        scored.append((score, d))
    scored.sort(key=lambda x: x[0], reverse=True)
    return [d for s, d in scored[:top_k]]

def build_context(retrieved: List[Doc], overhead_chars: int) -> str:
    parts = []
    for d in retrieved:
        parts.append(f"[DOC {d.doc_id}]\n{d.text.strip()}")
    context = "\n\n---\n\n".join(parts)
    #имитируем служебные токены/разметку/инструкции для RAG
    if overhead_chars > 0:
        context += "\n" + ("#" * min(overhead_chars, 200))  # просто добавляем символы, чтобы росли токены
    return context