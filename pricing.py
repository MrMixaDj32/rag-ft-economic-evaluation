from dataclasses import dataclass

@dataclass(frozen=True)
class Pricing:
    llm_in_per_1k: float
    llm_out_per_1k: float

def llm_cost(tokens_in: int, tokens_out: int, p: Pricing) -> float:
    return (tokens_in / 1000.0) * p.llm_in_per_1k + (tokens_out / 1000.0) * p.llm_out_per_1k