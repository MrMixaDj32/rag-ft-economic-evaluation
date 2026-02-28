import numpy as np
from dataclasses import dataclass

@dataclass
class SimResult:
    tokens_in: int
    tokens_out: int
    latency_ms: float
    correct: bool

def clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))

def approx_tokens_from_chars(chars: int, tokens_per_char: float) -> int:
    #минимум 1 токен
    return max(1, int(np.ceil(chars * tokens_per_char)))

def question_difficulty(q_type: str) -> float:
    # 0..1
    return {
        "limit": 0.20,
        "plan": 0.40,
        "fact": 0.40,
        "delta_pct": 0.60
    }.get(q_type, 0.45)

def simulate_llm_call(
    rng: np.random.Generator,
    mode: str,                         # "rag" | "ft"
    q_type: str,
    prompt_chars: int,
    tokens_per_char: float,
    latency_cfg: dict,
    quality_cfg: dict,
    rag_hit: bool = False,
    ft_stale: float = 0.0             # 0..1, где 1 = сильно устарело
) -> SimResult:
    """
    Возвращает имитацию: токены, задержку, correct.
    """
    diff = question_difficulty(q_type)

    #токены входа: по длине промпта
    tokens_in = approx_tokens_from_chars(prompt_chars, tokens_per_char)

    #токены выхода: зависят от типа вопроса (короткий ответ / ответ-число)
    base_out = {"limit": 18, "plan": 22, "fact": 22, "delta_pct": 20}.get(q_type, 22)
    tokens_out = int(max(8, rng.normal(loc=base_out, scale=3)))

    #задержка: base + per_token*(in+out) + rag_extra + jitter
    base_ms = float(latency_cfg["base_ms"])
    per_token_ms = float(latency_cfg["per_token_ms"])
    rag_extra = float(latency_cfg["rag_extra_ms"]) if mode == "rag" else 0.0
    jitter = float(latency_cfg["jitter_ms"])

    latency_ms = base_ms + per_token_ms * (tokens_in + tokens_out) + rag_extra
    latency_ms += float(rng.uniform(-jitter, jitter))
    latency_ms = max(20.0, latency_ms)

    #качество
    base_rag = float(quality_cfg["base_rag"])
    base_ft = float(quality_cfg["base_ft"])
    diff_pen_rag = float(quality_cfg["diff_penalty_rag"])
    diff_pen_ft = float(quality_cfg["diff_penalty_ft"])
    rag_hit_bonus = float(quality_cfg["rag_hit_bonus"])
    ft_stale_penalty = float(quality_cfg["ft_stale_penalty"])
    lo = float(quality_cfg["clamp_min"])
    hi = float(quality_cfg["clamp_max"])

    if mode == "rag":
        p = base_rag - diff_pen_rag * diff + (rag_hit_bonus if rag_hit else 0.0)
    else:
        p = base_ft - diff_pen_ft * diff - ft_stale_penalty * ft_stale

    p = clamp(p, lo, hi)
    correct = bool(rng.random() < p)

    return SimResult(tokens_in=tokens_in, tokens_out=tokens_out, latency_ms=latency_ms, correct=correct)