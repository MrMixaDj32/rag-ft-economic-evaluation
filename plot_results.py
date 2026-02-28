import os
import pandas as pd
import matplotlib.pyplot as plt

RESULTS_DIR = "results"
CSV_PATH = os.path.join(RESULTS_DIR, "summary.csv")

FIG1_PATH = os.path.join(RESULTS_DIR, "fig1_tco_vs_lambda.png")
FIG2_PATH = os.path.join(RESULTS_DIR, "fig2_cost_per_query_vs_lambda.png")
FIG3_PATH = os.path.join(RESULTS_DIR, "fig3_delta_accuracy_vs_u.png")

def require_columns(df: pd.DataFrame, cols: list[str]) -> None:
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(f"В summary.csv отсутствуют колонки: {missing}")

def sort_key_lambda(x) -> int:
    return int(x)

def sort_key_u(x) -> float:
    return float(x)

def save_figure(path: str, dpi: int = 300) -> None:
    plt.tight_layout()
    plt.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close()

def plot_fig1_tco_vs_lambda(df: pd.DataFrame) -> None:
    #Рисунок 1
    require_columns(df, ["U_monthly", "lambda_per_day", "cost_total_rag", "cost_total_ft"])

    u_vals = sorted(df["U_monthly"].unique(), key=sort_key_u)
    lam_vals = sorted(df["lambda_per_day"].unique(), key=sort_key_lambda)

    plt.figure(figsize=(9, 5))

    for u in u_vals:
        sub = df[df["U_monthly"] == u].sort_values("lambda_per_day")
        plt.plot(sub["lambda_per_day"], sub["cost_total_rag"], marker="o", linewidth=1.7,
                 label=f"RAG, U={u}")
        plt.plot(sub["lambda_per_day"], sub["cost_total_ft"], marker="x", linewidth=1.7,
                 label=f"Fine-tuning, U={u}")

    plt.xlabel("λ, запросов в день")
    plt.ylabel("TCO, усл. ед. (лог. шкала)")
    plt.title("Рисунок 1. TCO в зависимости от интенсивности запросов при различных U")
    plt.xticks(lam_vals, [str(v) for v in lam_vals])
    plt.yscale("log")
    plt.grid(True, linewidth=0.4, alpha=0.5)
    plt.legend(ncol=2, fontsize=9)
    save_figure(FIG1_PATH)

def plot_fig2_cost_per_query_vs_lambda(df: pd.DataFrame) -> None:
    #Рисунок 2
    require_columns(df, ["U_monthly", "lambda_per_day", "cost_per_query_rag", "cost_per_query_ft"])

    u_vals = sorted(df["U_monthly"].unique(), key=sort_key_u)
    lam_vals = sorted(df["lambda_per_day"].unique(), key=sort_key_lambda)

    plt.figure(figsize=(9, 5))

    for u in u_vals:
        sub = df[df["U_monthly"] == u].sort_values("lambda_per_day")
        plt.plot(sub["lambda_per_day"], sub["cost_per_query_rag"], marker="o", linewidth=1.7,
                 label=f"RAG, U={u}")
        plt.plot(sub["lambda_per_day"], sub["cost_per_query_ft"], marker="x", linewidth=1.7,
                 label=f"Fine-tuning, U={u}")

    plt.xlabel("λ, запросов в день")
    plt.ylabel("Средняя стоимость запроса, усл. ед. (лог. шкала)")
    plt.title("Рисунок 2. Средняя стоимость запроса в зависимости от λ")
    plt.xticks(lam_vals, [str(v) for v in lam_vals])
    plt.yscale("log")
    plt.grid(True, linewidth=0.4, alpha=0.5)
    plt.legend(ncol=2, fontsize=9)
    save_figure(FIG2_PATH)

def plot_fig3_delta_accuracy_vs_u(df: pd.DataFrame) -> None:
    #Рисунок 3
    require_columns(df, ["U_monthly", "lambda_per_day", "acc_rag_sim", "acc_ft_sim"])

    lam_vals = sorted(df["lambda_per_day"].unique(), key=sort_key_lambda)
    u_vals = sorted(df["U_monthly"].unique(), key=sort_key_u)

    fig, axes = plt.subplots(nrows=len(lam_vals), ncols=1, figsize=(9, 9), sharex=True)
    if len(lam_vals) == 1:
        axes = [axes]

    for ax, lam in zip(axes, lam_vals):
        sub = df[df["lambda_per_day"] == lam].sort_values("U_monthly")
        delta = sub["acc_rag_sim"].values - sub["acc_ft_sim"].values

        ax.plot(sub["U_monthly"], delta, marker="o", linewidth=1.9, label="Δ = RAG - Fine-tuning")
        ax.axhline(0.0, linewidth=1.0)
        ax.set_title(f"λ = {lam} запросов в день", fontsize=10)
        ax.set_ylabel("Разница долей")
        ax.grid(True, linewidth=0.4, alpha=0.5)
        ax.legend(fontsize=9)

    axes[-1].set_xlabel("U, доля обновления базы знаний за период")
    axes[-1].set_xticks(u_vals, [str(v) for v in u_vals])

    fig.suptitle("Рисунок 3. Преимущество RAG по качеству (Δaccuracy) в зависимости от U", y=0.98)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(FIG3_PATH, dpi=300, bbox_inches="tight")
    plt.close(fig)

def main():
    if not os.path.exists(CSV_PATH):
        raise FileNotFoundError(
            f"Не найден {CSV_PATH}. Сначала запусти run_experiment.py, чтобы он создал summary.csv."
        )

    os.makedirs(RESULTS_DIR, exist_ok=True)
    df = pd.read_csv(CSV_PATH)

    df = df.copy()
    df["lambda_per_day"] = df["lambda_per_day"].astype(int)
    df["U_monthly"] = df["U_monthly"].astype(float)

    plot_fig1_tco_vs_lambda(df)
    plot_fig2_cost_per_query_vs_lambda(df)
    plot_fig3_delta_accuracy_vs_u(df)

    print("Готово. Сгенерированы рисунки для статьи:")
    print(f"- {FIG1_PATH}")
    print(f"- {FIG2_PATH}")
    print(f"- {FIG3_PATH}")

if __name__ == "__main__":
    main()