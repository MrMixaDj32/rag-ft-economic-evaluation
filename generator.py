import random
from dataclasses import dataclass
from typing import List, Dict, Tuple

@dataclass
class Doc:
    doc_id: str
    text: str

@dataclass
class QA:
    q_id: str
    question: str
    answer: str
    source_doc_id: str
    q_type: str  # "limit" | "plan" | "fact" | "delta_pct"

KPI_LIST = [
    "Выручка", "EBITDA", "OPEX", "CAPEX", "Валовая маржа", "Чистая прибыль",
    "Дебиторская задолженность", "Кредиторская задолженность", "CAC", "LTV",
    "Churn", "ARPU", "NPS", "Себестоимость", "Запасы"
]

DEPTS = ["Продажи", "Маркетинг", "ИТ", "Финансы", "Закупки", "HR", "Операции"]

def month_name(m: int) -> str:
    names = ["январь","февраль","март","апрель","май","июнь","июль","август","сентябрь","октябрь","ноябрь","декабрь"]
    return names[(m - 1) % 12]

def generate_corpus(seed: int, companies: int, months: int, docs_total: int) -> Tuple[List[Doc], Dict]:
    """
    Синтетический корпус (русский "корпоративный" стиль).
    Возвращает docs и world (параметры "мира") для генерации gold-QA.
    """
    rnd = random.Random(seed)
    world = {"companies": [], "plan_fact": {}, "limits": {}}

    # мир
    for c in range(companies):
        cname = f"Компания_{c+1}"
        world["companies"].append(cname)

        for d in DEPTS:
            world["limits"][(cname, d)] = rnd.choice([100_000, 200_000, 300_000, 500_000])

        for m in range(1, months + 1):
            for kpi in KPI_LIST:
                plan = rnd.randint(50, 500) * 1_000_000
                fact = int(plan * rnd.uniform(0.85, 1.15))
                world["plan_fact"][(cname, m, kpi)] = {"plan": plan, "fact": fact}

    docs: List[Doc] = []

    #регламенты
    base_docs: List[Doc] = []
    for cname in world["companies"]:
        for d in DEPTS:
            lim = world["limits"][(cname, d)]
            text = (
                f"Регламент согласования расходов.\n"
                f"Организация: {cname}.\n"
                f"Подразделение: {d}.\n"
                f"Лимит единовременного согласования: {lim} руб.\n"
                f"При превышении лимита требуется согласование финансового директора.\n"
            )
            base_docs.append(Doc(doc_id=f"REG_{cname}_{d}", text=text))

    #отчёты
    for cname in world["companies"]:
        for m in range(1, months + 1):
            lines = [f"Управленческий отчёт.\nОрганизация: {cname}.\nПериод: {month_name(m)}.\n\n"]
            # фиксируем 8 KPI в документе
            kpis = rnd.sample(KPI_LIST, k=8)
            for kpi in kpis:
                pf = world["plan_fact"][(cname, m, kpi)]
                lines.append(f"{kpi}: план {pf['plan']} руб., факт {pf['fact']} руб.\n")
            base_docs.append(Doc(doc_id=f"REP_{cname}_{m:02d}", text="".join(lines)))

    #KPI справочник
    for kpi in KPI_LIST:
        text = (
            f"Справочник KPI.\n"
            f"Показатель: {kpi}.\n"
            f"Определение: показатель используется в управленческой аналитике.\n"
            f"Примечание: значения приводятся в рублях, если не указано иное.\n"
        )
        base_docs.append(Doc(doc_id=f"KPI_{kpi}", text=text))

    #добиваем до docs_total вариациями
    docs.extend(base_docs)
    while len(docs) < docs_total:
        src = rnd.choice(base_docs)
        extra = rnd.choice([
            "Примечание: документ актуален на дату формирования.\n",
            "Примечание: отклонения требуют дополнительного анализа.\n",
            "Примечание: источник данных — управленческий контур.\n",
        ])
        docs.append(Doc(doc_id=f"{src.doc_id}_V{len(docs)}", text=src.text + extra))

    return docs[:docs_total], world

def generate_gold_qa(seed: int, world: Dict, months: int, n_questions: int) -> List[QA]:
    rnd = random.Random(seed + 999)
    qas: List[QA] = []
    companies = world["companies"]

    for i in range(n_questions):
        q_type = rnd.choice(["limit", "plan", "fact", "delta_pct"])
        cname = rnd.choice(companies)

        if q_type == "limit":
            dept = rnd.choice(DEPTS)
            lim = world["limits"][(cname, dept)]
            qas.append(QA(
                q_id=f"Q{i:04d}",
                question=f"Какой лимит единовременного согласования расходов для подразделения {dept} в {cname}?",
                answer=f"{lim} руб.",
                source_doc_id=f"REG_{cname}_{dept}",
                q_type="limit"
            ))
        else:
            m = rnd.randint(1, months)
            kpi = rnd.choice(KPI_LIST)
            pf = world["plan_fact"][(cname, m, kpi)]
            rep_id = f"REP_{cname}_{m:02d}"

            if q_type == "plan":
                qas.append(QA(
                    q_id=f"Q{i:04d}",
                    question=f"Какое плановое значение показателя «{kpi}» за {month_name(m)} для {cname}?",
                    answer=f"{pf['plan']} руб.",
                    source_doc_id=rep_id,
                    q_type="plan"
                ))
            elif q_type == "fact":
                qas.append(QA(
                    q_id=f"Q{i:04d}",
                    question=f"Какое фактическое значение показателя «{kpi}» за {month_name(m)} для {cname}?",
                    answer=f"{pf['fact']} руб.",
                    source_doc_id=rep_id,
                    q_type="fact"
                ))
            else:
                plan, fact = pf["plan"], pf["fact"]
                delta = (fact - plan) / plan * 100.0
                qas.append(QA(
                    q_id=f"Q{i:04d}",
                    question=f"На сколько процентов факт отличается от плана по «{kpi}» за {month_name(m)} для {cname}?",
                    answer=f"{delta:.1f}%",
                    source_doc_id=rep_id,
                    q_type="delta_pct"
                ))
    return qas