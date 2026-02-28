import os
import yaml
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from generator import generate_corpus, generate_gold_qa, QA
from sim import simulate
from retrieval import retrieve_topk, build_context
from pricing import Pricing, llm_cost
from llm_sim import simulate_llm_call

def percentile(values, q):
    if not values:
        return None
    s = pd.Series(values)
    return float(s.quantile(q))

def build_prompt_chars(system_prompt: str, question: str, context: str | None) -> int:
    if context is None:
        text = f"{system_prompt}\n\nВопрос:\n{question}\nОтвет:"
    else:
        text = f"{system_prompt}\n\nКонтекст:\n{context}\n\nВопрос:\n{question}\nОтвет:"
    return len(text)

def ft_staleness(cum_update_share_since_retrain: float) -> float:
    # c_update_share_since_retrain: примерно 0..U_monthly*(days/retrain_period) усиливаем эффект, чтобы U было видно:
    return min(1.0, cum_update_share_since_retrain * 4.0)

def main():
    cfg = yaml.safe_load(open("config.yaml", "r", encoding="utf-8"))

    seed = int(cfg["seed"])
    exp = cfg["experiment"]
    days = int(exp["days"])
    D_docs = int(exp["D_docs"])
    months = int(exp["months"])
    companies = int(exp["companies"])
    top_k = int(exp["top_k"])
    n_questions = int(exp["questions"])

    U_list = list(cfg["scenarios"]["U_monthly"])
    lam_list = list(cfg["scenarios"]["lambda_per_day"])

    system_prompt = str(cfg["assumptions"]["system_prompt"])
    tokens_per_char = float(cfg["assumptions"]["tokens_per_char_ru"])
    rag_ctx_overhead = int(cfg["assumptions"]["rag_context_overhead_chars"])
    retr_cost = float(cfg["assumptions"]["retr_cost_per_query"])

    ft_policy = cfg["fine_tuning_policy"]
    retrain_every = int(ft_policy["retrain_every_days"])
    ft_train_cost_per_run = float(ft_policy["train_cost_per_run"])

    rag_policy = cfg["rag_policy"]
    rag_index_update_cost_per_1pct = float(rag_policy["index_update_cost_per_1pct"])

    pricing_cfg = cfg["pricing"]
    pricing = Pricing(
        llm_in_per_1k=float(pricing_cfg["llm_input_per_1k"]),
        llm_out_per_1k=float(pricing_cfg["llm_output_per_1k"]),
    )

    latency_cfg = cfg["latency_model"]
    quality_cfg = cfg["quality_model"]

    runtime_cfg = cfg.get("runtime", {})
    max_requests_per_scenario = int(runtime_cfg.get("max_requests_per_scenario", 10**18))

    #генерация корпуса и Q/A
    docs, world = generate_corpus(seed=seed, companies=companies, months=months, docs_total=D_docs)
    qas = generate_gold_qa(seed=seed, world=world, months=months, n_questions=n_questions)

    os.makedirs("results", exist_ok=True)
    rows = []

    for U in U_list:
        for lam in lam_list:
            rng = np.random.default_rng(seed + int(lam) + int(float(U) * 10_000))
            loads = simulate(days=days, lambda_per_day=int(lam), U_monthly=float(U), seed=seed + int(lam))

            #истинное число запросов по сценарию (по пуассону)
            N_total_true = sum(d.n_requests for d in loads)

            #ограничение на реально симулируемые запросы
            req_left = max_requests_per_scenario

            #фиксированные издержки считаем отдельно (накопители!)
            rag_index_cost_total = 0.0
            ft_train_cost_total = 0.0

            #переменные издержки (инференс+retrieval)
            rag_var_cost = 0.0
            ft_var_cost = 0.0

            #метрики по симуляции
            N_sim = 0
            rag_correct = 0
            ft_correct = 0

            rag_lat = []
            ft_lat = []

            rag_tok_in = rag_tok_out = 0
            ft_tok_in = ft_tok_out = 0

            #сколько событий обновления с момента retrain
            update_events_since_retrain = 0

            q_idx = 0

            for day in loads:
                #retrain FT по расписанию
                if day.day == 1 or (day.day - 1) % retrain_every == 0:
                    ft_train_cost_total += ft_train_cost_per_run
                    update_events_since_retrain = 0

                #событие обновления знаний
                #ежедневные обновления: cost и staleness пропорциональны update_share
                daily_update_pct = day.update_share * 100.0
                rag_index_cost_total += rag_index_update_cost_per_1pct * daily_update_pct

                #для FT вместо "числа событий" копим долю обновлений с момента retrain
                update_events_since_retrain += day.update_share  # теперь это накопленная доля (0..)

                #сколько запросов сегодня реально симулируем
                take = min(day.n_requests, req_left)
                req_left -= take

                for _ in range(take):
                    N_sim += 1
                    qa: QA = qas[q_idx % len(qas)]
                    q_idx += 1

                    #RAG
                    retrieved = retrieve_topk(qa.question, docs, top_k=top_k)
                    hit = any(d.doc_id == qa.source_doc_id for d in retrieved)
                    context = build_context(retrieved, overhead_chars=rag_ctx_overhead)

                    rag_prompt_chars = build_prompt_chars(system_prompt, qa.question, context)
                    rag_sim = simulate_llm_call(
                        rng=rng,
                        mode="rag",
                        q_type=qa.q_type,
                        prompt_chars=rag_prompt_chars,
                        tokens_per_char=tokens_per_char,
                        latency_cfg=latency_cfg,
                        quality_cfg=quality_cfg,
                        rag_hit=hit,
                        ft_stale=0.0
                    )

                    rag_tok_in += rag_sim.tokens_in
                    rag_tok_out += rag_sim.tokens_out
                    rag_lat.append(rag_sim.latency_ms)

                    rag_var_cost += llm_cost(rag_sim.tokens_in, rag_sim.tokens_out, pricing) + retr_cost
                    rag_correct += 1 if rag_sim.correct else 0

                    #FT
                    stale = ft_staleness(update_events_since_retrain)
                    ft_prompt_chars = build_prompt_chars(system_prompt, qa.question, None)
                    ft_sim = simulate_llm_call(
                        rng=rng,
                        mode="ft",
                        q_type=qa.q_type,
                        prompt_chars=ft_prompt_chars,
                        tokens_per_char=tokens_per_char,
                        latency_cfg=latency_cfg,
                        quality_cfg=quality_cfg,
                        rag_hit=False,
                        ft_stale=stale
                    )

                    ft_tok_in += ft_sim.tokens_in
                    ft_tok_out += ft_sim.tokens_out
                    ft_lat.append(ft_sim.latency_ms)

                    ft_var_cost += llm_cost(ft_sim.tokens_in, ft_sim.tokens_out, pricing)
                    ft_correct += 1 if ft_sim.correct else 0

                if req_left <= 0:
                    break

            #масштабирование переменных издержек до полной нагрузки
            if N_sim > 0 and N_total_true > N_sim:
                scale = N_total_true / N_sim
            else:
                scale = 1.0

            rag_cost_total = rag_var_cost * scale + rag_index_cost_total
            ft_cost_total = ft_var_cost * scale + ft_train_cost_total

            #cost per query считаем для полной нагрузки
            cost_per_query_rag = rag_cost_total / max(1, N_total_true)
            cost_per_query_ft = ft_cost_total / max(1, N_total_true)

            #accuracy считаем только на симулированной части
            acc_rag = rag_correct / max(1, N_sim)
            acc_ft = ft_correct / max(1, N_sim)

            c_rag_var = rag_var_cost / max(1, N_sim)
            c_ft_var = ft_var_cost / max(1, N_sim)
            denom = (c_rag_var - c_ft_var)

            if abs(denom) < 1e-12:
                N_star = None
            else:
                N_star = (ft_train_cost_total - rag_index_cost_total) / denom
                if N_star < 0:
                    N_star = None

            rows.append({
                "U_monthly": float(U),
                "lambda_per_day": int(lam),
                "days": int(days),

                "N_total_true": int(N_total_true),
                "N_simulated": int(N_sim),
                "scale": float(scale),

                "acc_rag_sim": float(acc_rag),
                "acc_ft_sim": float(acc_ft),

                "cost_total_rag": float(rag_cost_total),
                "cost_total_ft": float(ft_cost_total),
                "cost_per_query_rag": float(cost_per_query_rag),
                "cost_per_query_ft": float(cost_per_query_ft),

                "rag_index_cost_total": float(rag_index_cost_total),
                "ft_train_cost_total": float(ft_train_cost_total),

                "tokens_in_rag_sim": int(rag_tok_in),
                "tokens_out_rag_sim": int(rag_tok_out),
                "tokens_in_ft_sim": int(ft_tok_in),
                "tokens_out_ft_sim": int(ft_tok_out),

                "lat_p50_rag_ms": percentile(rag_lat, 0.50),
                "lat_p95_rag_ms": percentile(rag_lat, 0.95),
                "lat_p50_ft_ms": percentile(ft_lat, 0.50),
                "lat_p95_ft_ms": percentile(ft_lat, 0.95),

                "ft_retrain_every_days": int(retrain_every),

                "N_star_break_even": N_star,
            })

            print(f"[OK] U={U} λ={lam} simulated={N_sim} true={N_total_true} scale={scale:.2f}")

    df = pd.DataFrame(rows)
    df.to_csv("results/summary.csv", index=False)

    #графики
    #TCO vs lambda для каждого U
    for U in sorted(df["U_monthly"].unique()):
        sub = df[df["U_monthly"] == U].sort_values("lambda_per_day")
        plt.figure()
        plt.plot(sub["lambda_per_day"], sub["cost_total_rag"], label="RAG TCO")
        plt.plot(sub["lambda_per_day"], sub["cost_total_ft"], label="FT TCO")
        plt.xlabel("λ (requests/day)")
        plt.ylabel(f"TCO ({pricing_cfg['currency']})")
        plt.title(f"TCO vs load (U={U})")
        plt.legend()
        plt.savefig(f"results/tco_U{U}.png", dpi=200, bbox_inches="tight")
        plt.close()

    #Cost per query across scenarios
    df2 = df.sort_values(["U_monthly", "lambda_per_day"]).copy()
    labels = [f"U={r.U_monthly},λ={r.lambda_per_day}" for r in df2.itertuples()]
    x = np.arange(len(labels))

    plt.figure(figsize=(12, 4))
    plt.plot(x, df2["cost_per_query_rag"], label="RAG cost/query")
    plt.plot(x, df2["cost_per_query_ft"], label="FT cost/query")
    plt.xticks(x, labels, rotation=45, ha="right")
    plt.xlabel("Scenario")
    plt.ylabel(f"Cost per query ({pricing_cfg['currency']})")
    plt.title("Cost per query across scenarios")
    plt.legend()
    plt.tight_layout()
    plt.savefig("results/cost_per_query.png", dpi=200, bbox_inches="tight")
    plt.close()

    print("Готово.")
    print("Файлы:")
    print("- results/summary.csv")
    print("- results/tco_U*.png")
    print("- results/cost_per_query.png")

if __name__ == "__main__":
    main()