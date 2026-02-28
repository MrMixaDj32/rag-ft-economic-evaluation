import numpy as np
from dataclasses import dataclass
from typing import List

@dataclass
class DayLoad:
    day: int
    n_requests: int
    update_share: float  #доля обновления базы в этот день (0..1)

def simulate(days: int, lambda_per_day: int, U_monthly: float, seed: int) -> List[DayLoad]:
    """
    Пуассоновский поток запросов + ежедневные обновления базы.
    update_share = U_monthly / 30 (детерминированно, чтобы U точно влиял).
    """
    rng = np.random.default_rng(seed)
    daily_update = float(U_monthly) / 30.0

    loads: List[DayLoad] = []
    for d in range(1, days + 1):
        n = int(rng.poisson(lambda_per_day))
        loads.append(DayLoad(day=d, n_requests=n, update_share=daily_update))
    return loads