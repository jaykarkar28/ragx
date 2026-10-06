from pathlib import Path
import json
import pandas as pd
import streamlit as st


BASE_DIR = Path(__file__).resolve().parent.parent
REPORT_DIR = BASE_DIR / "experiments" / "reports"


st.set_page_config(
    page_title="RAGX Evaluation Dashboard",
    page_icon="🔎",
    layout="wide",
)


def load_latest_report():
    reports = sorted(
        REPORT_DIR.glob("retrieval_experiment_*.json"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )

    if not reports:
        return None

    with open(reports[0], encoding="utf-8") as file:
        return json.load(file)


st.title("🔎 RAGX Evaluation Dashboard")
st.caption("Evaluation-driven Retrieval-Augmented Generation system")

report = load_latest_report()

if report is None:
    st.error("No experiment report found.")
    st.stop()


# -------------------------------------------------------------------
# Overview
# -------------------------------------------------------------------

st.header("Evaluation Overview")

answer_eval = report.get("answer_evaluation", {})

col1, col2, col3, col4 = st.columns(4)

col1.metric(
    "Faithfulness",
    f"{answer_eval.get('faithfulness', 0):.2%}",
)

col2.metric(
    "Answer Relevance",
    f"{answer_eval.get('answer_relevance', 0):.2%}",
)

col3.metric(
    "Grounded Rate",
    f"{answer_eval.get('grounded_rate', 0):.2%}",
)

col4.metric(
    "Reference Score",
    f"{answer_eval.get('reference_answer_score', 0):.2%}",
)


# -------------------------------------------------------------------
# Retrieval metrics
# -------------------------------------------------------------------

st.header("Retrieval Performance")

retrieval_runs = report.get("retrieval_runs", [])

rows = []

for run in retrieval_runs:
    k = run.get("k")

    for stage, metrics in run.get("average_metrics", {}).items():
        rows.append(
            {
                "K": k,
                "Stage": stage,
                "MAP": metrics.get("map", 0),
                "MRR": metrics.get("mrr", 0),
                "NDCG": metrics.get(f"ndcg@{k}", 0),
                "Precision": metrics.get(f"precision@{k}", 0),
                "Recall": metrics.get(f"recall@{k}", 0),
            }
        )

if rows:
    retrieval_df = pd.DataFrame(rows)

    selected_k = st.selectbox(
        "Select K",
        sorted(retrieval_df["K"].unique()),
        index=2 if len(retrieval_df["K"].unique()) > 2 else 0,
    )

    filtered = retrieval_df[
        retrieval_df["K"] == selected_k
    ].set_index("Stage")

    st.dataframe(
        filtered.style.format(
            {
                "MAP": "{:.4f}",
                "MRR": "{:.4f}",
                "NDCG": "{:.4f}",
                "Precision": "{:.4f}",
                "Recall": "{:.4f}",
            }
        ),
        use_container_width=True,
    )

    st.subheader("Retrieval Metrics")

    chart_df = filtered[
        ["MAP", "MRR", "NDCG", "Precision", "Recall"]
    ]

    st.bar_chart(chart_df)


# -------------------------------------------------------------------
# Report information
# -------------------------------------------------------------------

st.header("Experiment Information")

info_col1, info_col2, info_col3 = st.columns(3)

info_col1.metric(
    "Queries Evaluated",
    report.get("queries_evaluated", 0),
)

info_col2.metric(
    "Index Size",
    report.get("index_size", 0),
)

info_col3.metric(
    "Generation K",
    report.get("production_generation_k", 0),
)


st.caption(
    f"Latest report: {Path(report.get('dataset_path', '')).name}"
)