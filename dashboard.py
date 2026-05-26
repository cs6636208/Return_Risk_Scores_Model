from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st


ROOT = Path(__file__).resolve().parent
CLEAN_DATA_PATH = ROOT / "data" / "processed" / "clean_dataset.csv"
MODEL_COMPARISON_PATH = ROOT / "reports" / "model_training" / "model_comparison.csv"
V3_METRICS_PATH = ROOT / "reports" / "model_evaluation_v3" / "metrics_summary_v3.csv"
V3_TRADEOFF_PATH = ROOT / "reports" / "model_evaluation_v3" / "accuracy_recall_tradeoff.png"
SHAP_SUMMARY_PATH = ROOT / "reports" / "model_evaluation_v2" / "shap_summary_v2.png"
SHAP_WATERFALL_PATH = ROOT / "reports" / "model_evaluation_v2" / "shap_waterfall_top_risk_v2.png"


st.set_page_config(
    page_title="Return Risk Dashboard",
    layout="wide",
)


@st.cache_data
def load_orders() -> pd.DataFrame:
    df = pd.read_csv(CLEAN_DATA_PATH, parse_dates=["order_date"], low_memory=False)
    if "risk_score" not in df.columns:
        df["risk_score"] = estimate_fallback_risk(df)
    df["risk_score"] = pd.to_numeric(df["risk_score"], errors="coerce").fillna(0.0).clip(0, 1)
    if "risk_tier" not in df.columns:
        df["risk_tier"] = pd.cut(
            df["risk_score"],
            bins=[-0.01, 0.20, 0.40, 1.0],
            labels=["Low", "Medium", "High"],
        ).astype(str)
    return df


@st.cache_data
def load_csv(path: Path) -> pd.DataFrame | None:
    if not path.exists():
        return None
    return pd.read_csv(path)


def estimate_fallback_risk(df: pd.DataFrame) -> pd.Series:
    score = pd.Series(0.05, index=df.index)
    score += (pd.to_numeric(df.get("hist_return_rate", 0), errors="coerce").fillna(0) * 0.35)
    score += (pd.to_numeric(df.get("product_rating", 5), errors="coerce").fillna(5) < 4.0) * 0.15
    score += (pd.to_numeric(df.get("total_discount_pct", 0), errors="coerce").fillna(0) > 0.20) * 0.10
    score += (pd.to_numeric(df.get("delay_days", 0), errors="coerce").fillna(0) > 0) * 0.15
    payment_method = df.get("payment_method", pd.Series("", index=df.index))
    score += payment_method.astype(str).eq("COD") * 0.05
    return score.clip(0, 1)


def money(value: float) -> str:
    return f"{value:,.0f} THB"


def tier_color(tier: str) -> str:
    return {"High": "#d62728", "Medium": "#f0ad4e", "Low": "#2ca02c"}.get(tier, "#6c757d")


def explain_order(row: pd.Series) -> pd.DataFrame:
    factors = []

    hist_return_rate = float(row.get("hist_return_rate", 0) or 0)
    if hist_return_rate >= 0.20:
        factors.append(("Customer return history", "High", hist_return_rate, "Customer has returned a high share of previous orders."))
    elif hist_return_rate > 0:
        factors.append(("Customer return history", "Medium", hist_return_rate, "Customer has some prior return behavior."))

    product_rating = float(row.get("product_rating", 5) or 5)
    if product_rating < 4.0:
        factors.append(("Low product rating", "High", product_rating, "Lower-rated products are more likely to be returned."))

    discount_pct = float(row.get("total_discount_pct", 0) or 0)
    if discount_pct >= 0.20:
        factors.append(("High discount", "Medium", discount_pct, "Large discounts may indicate impulse buying or low commitment."))

    delay_days = float(row.get("delay_days", 0) or 0)
    if delay_days > 0:
        factors.append(("Delivery delay", "High", delay_days, "Late delivery increases cancellation and return risk."))

    if str(row.get("payment_method", "")) == "COD":
        factors.append(("Cash on delivery", "Medium", 1, "COD orders can carry higher non-completion or return risk."))

    if int(row.get("is_repurchased_item", 0) or 0) == 1:
        factors.append(("Repurchased item", "Low", 1, "Repurchase behavior can reduce uncertainty for this product."))

    if not factors:
        factors.append(("No major warning factor", "Low", 0, "This order has no strong risk driver in the current review rules."))

    return pd.DataFrame(factors, columns=["factor", "severity", "value", "interpretation"])


def show_image_if_exists(path: Path, caption: str) -> None:
    if path.exists():
        st.image(str(path), caption=caption, use_container_width=True)
    else:
        st.info(f"Missing artifact: {path.relative_to(ROOT)}")


orders = load_orders()
model_comparison = load_csv(MODEL_COMPARISON_PATH)
v3_metrics = load_csv(V3_METRICS_PATH)

st.title("Return Risk Review")
st.caption("Operational dashboard for prototype return and refund risk scoring.")

with st.sidebar:
    st.header("Filters")
    categories = sorted(orders["category"].dropna().unique()) if "category" in orders else []
    selected_categories = st.multiselect("Category", categories, default=categories)
    tiers = ["High", "Medium", "Low"]
    selected_tiers = st.multiselect("Risk tier", tiers, default=tiers)
    min_score = st.slider("Minimum risk score", 0.0, 1.0, 0.0, 0.01)
    search_order = st.text_input("Order ID contains", "")

filtered = orders.copy()
if selected_categories:
    filtered = filtered[filtered["category"].isin(selected_categories)]
if selected_tiers:
    filtered = filtered[filtered["risk_tier"].isin(selected_tiers)]
filtered = filtered[filtered["risk_score"] >= min_score]
if search_order:
    filtered = filtered[filtered["order_id"].astype(str).str.contains(search_order, case=False, na=False)]

total_orders = len(filtered)
return_rate = filtered["is_returned"].mean() if total_orders else 0
avg_score = filtered["risk_score"].mean() if total_orders else 0
high_risk = int((filtered["risk_tier"] == "High").sum()) if total_orders else 0

metric_cols = st.columns(4)
metric_cols[0].metric("Orders", f"{total_orders:,}")
metric_cols[1].metric("Observed return rate", f"{return_rate:.1%}")
metric_cols[2].metric("Average risk score", f"{avg_score:.2f}")
metric_cols[3].metric("High-risk orders", f"{high_risk:,}")

tab_overview, tab_order, tab_model = st.tabs(["Portfolio", "Order Detail", "Model Evidence"])

with tab_overview:
    left, right = st.columns([1.1, 0.9])
    with left:
        by_tier = (
            filtered.groupby("risk_tier", observed=False)
            .agg(orders=("order_id", "count"), return_rate=("is_returned", "mean"), avg_score=("risk_score", "mean"))
            .reset_index()
        )
        fig = px.bar(
            by_tier,
            x="risk_tier",
            y="orders",
            color="risk_tier",
            color_discrete_map={"High": "#d62728", "Medium": "#f0ad4e", "Low": "#2ca02c"},
            title="Order volume by risk tier",
        )
        st.plotly_chart(fig, use_container_width=True)

    with right:
        by_category = (
            filtered.groupby("category", observed=False)
            .agg(return_rate=("is_returned", "mean"), orders=("order_id", "count"))
            .reset_index()
            .sort_values("return_rate", ascending=False)
        )
        fig = px.scatter(
            by_category,
            x="orders",
            y="return_rate",
            size="orders",
            color="category",
            title="Return rate by category",
            labels={"return_rate": "Return rate"},
        )
        st.plotly_chart(fig, use_container_width=True)

    st.dataframe(
        filtered.sort_values("risk_score", ascending=False)[
            [
                "order_id",
                "customer_id",
                "category",
                "brand",
                "province",
                "payment_method",
                "total_amount",
                "risk_score",
                "risk_tier",
                "is_returned",
            ]
        ].head(5000),
        use_container_width=True,
        hide_index=True,
    )

with tab_order:
    if filtered.empty:
        st.warning("No orders match the current filters.")
    else:
        ordered = filtered.sort_values("risk_score", ascending=False)
        selected_order = st.selectbox("Select order", ordered["order_id"].astype(str), index=0)
        row = orders.loc[orders["order_id"].astype(str).eq(selected_order)].iloc[0]

        detail_cols = st.columns([0.7, 1.3])
        with detail_cols[0]:
            color = tier_color(str(row["risk_tier"]))
            st.markdown(
                f"""
                <div style="border-left: 8px solid {color}; padding: 0.75rem 1rem; background: #f8f9fa;">
                  <div style="font-size: 0.85rem; color: #6c757d;">Risk tier</div>
                  <div style="font-size: 2rem; font-weight: 700;">{row['risk_tier']}</div>
                  <div style="font-size: 1.4rem;">Score {row['risk_score']:.2f}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
            st.write("")
            st.metric("Order amount", money(float(row.get("total_amount", 0) or 0)))
            st.metric("Product rating", f"{float(row.get('product_rating', 0) or 0):.1f}")
            st.metric("History return rate", f"{float(row.get('hist_return_rate', 0) or 0):.1%}")

        with detail_cols[1]:
            st.subheader("Order context")
            context = {
                "Customer": row.get("customer_id", ""),
                "Province": row.get("province", ""),
                "Category": row.get("category", ""),
                "Brand": row.get("brand", ""),
                "Channel": row.get("channel_type", ""),
                "Payment": row.get("payment_method", ""),
                "Courier": row.get("courier_name", ""),
                "Returned": "Yes" if int(row.get("is_returned", 0) or 0) == 1 else "No",
            }
            st.dataframe(pd.DataFrame(context.items(), columns=["field", "value"]), hide_index=True, use_container_width=True)

        st.subheader("Order-level explanation")
        st.dataframe(explain_order(row), use_container_width=True, hide_index=True)

with tab_model:
    metric_area, artifact_area = st.columns([0.9, 1.1])
    with metric_area:
        st.subheader("Model comparison")
        if model_comparison is not None:
            st.dataframe(model_comparison, use_container_width=True, hide_index=True)
        else:
            st.info("Run the training pipeline to create model comparison output.")

        st.subheader("V3 evaluation")
        if v3_metrics is not None:
            display = v3_metrics.copy()
            st.dataframe(display, use_container_width=True, hide_index=True)
        else:
            st.info('Run: python "docs/version 3/model_evaluation_v3.py"')

    with artifact_area:
        st.subheader("Explainability artifacts")
        show_image_if_exists(SHAP_SUMMARY_PATH, "V2 SHAP summary")
        show_image_if_exists(SHAP_WATERFALL_PATH, "V2 top-risk order waterfall")
        show_image_if_exists(V3_TRADEOFF_PATH, "V3 accuracy/recall threshold trade-off")
