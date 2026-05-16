"""
Startup failure analytics dashboard.

Run:
    pip install streamlit pandas numpy plotly scikit-learn openpyxl
    streamlit run startup_analytics_dashboard.py

Default input:
    C:\\Users\\hp\\Desktop\\startups.xlsx
"""

from __future__ import annotations

import sys
from pathlib import Path


try:
    import numpy as np
    import pandas as pd
    import plotly.express as px
    import plotly.graph_objects as go
    import streamlit as st
    from sklearn.compose import ColumnTransformer
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.impute import SimpleImputer
    from sklearn.linear_model import LinearRegression
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
    from sklearn.model_selection import train_test_split
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import OneHotEncoder, StandardScaler
except ModuleNotFoundError as exc:
    print(
        "\nMissing package: "
        f"{exc.name}\n\nInstall dependencies with:\n"
        "pip install streamlit pandas numpy plotly scikit-learn openpyxl\n",
        file=sys.stderr,
    )
    raise


DEFAULT_FILE = Path(r"C:\Users\hp\Desktop\startups.xlsx")
APP_TITLE = "Failed Indian Startups Analytics"


def make_one_hot_encoder() -> OneHotEncoder:
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", sparse=False)


def normalize_col(name: object) -> str:
    text = str(name).strip()
    text = text.replace("->", "to").replace("→", "to")
    text = text.replace("%", "pct").replace("$", "usd")
    text = text.replace("/", "_per_").replace("(", "").replace(")", "")
    text = text.replace("-", "_").replace(" ", "_")
    text = "".join(ch if ch.isalnum() or ch == "_" else "_" for ch in text)
    while "__" in text:
        text = text.replace("__", "_")
    return text.strip("_").lower()


@st.cache_data(show_spinner=False)
def load_startup_data(excel_path: str) -> pd.DataFrame:
    clean_path = str(excel_path).strip().strip('"').strip("'")
    while "\\\\" in clean_path:
        clean_path = clean_path.replace("\\\\", "\\")
    path = Path(clean_path).expanduser()
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")

    sheets = pd.ExcelFile(path).sheet_names
    sheet = "Sheet1" if "Sheet1" in sheets else sheets[0]

    preview = pd.read_excel(path, sheet_name=sheet, header=None, nrows=15)
    header_row = None
    for idx, row in preview.iterrows():
        values = {str(x).strip().lower() for x in row.dropna().tolist()}
        if "company" in values and "sector" in values:
            header_row = idx
            break
    if header_row is None:
        header_row = 0

    raw = pd.read_excel(path, sheet_name=sheet, header=header_row)
    raw = raw.dropna(how="all")
    raw.columns = [str(c).strip() for c in raw.columns]
    raw = raw.loc[:, ~raw.columns.str.startswith("Unnamed")]

    rename_map = {
        "Funding ($M)": "Funding_USD_M",
        "Founded Yr": "Founded_Year",
        "Shutdown Yr": "Shutdown_Year",
        "Shutdown Year": "Shutdown_Year",
        "Years Active": "Years_Active",
        "HQ State": "HQ_State",
        "Stage at Shutdown": "Stage_at_Shutdown",
        "Failure Category": "Failure_Category",
        "Peak Employees": "Peak_Employees",
        "Peak Valuation ($M)": "Peak_Valuation_USD_M",
        "Revenue FY22 ($M)": "Revenue_FY22_USD_M",
        "Revenue FY23 ($M)": "Revenue_FY23_USD_M",
        "Revenue FY24 ($M)": "Revenue_FY24_USD_M",
        "Loss FY22 ($M)": "Loss_FY22_USD_M",
        "Loss FY23 ($M)": "Loss_FY23_USD_M",
        "Loss FY24 ($M)": "Loss_FY24_USD_M",
        "Rev Growth FY22toFY23 (pct)": "Rev_Growth_FY22_to_FY23_pct",
        "Rev Growth FY22→FY23 (%)": "Rev_Growth_FY22_to_FY23_pct",
        "Rev Change FY23toFY24 (pct)": "Rev_Change_FY23_to_FY24_pct",
        "Rev Change FY23→FY24 (%)": "Rev_Change_FY23_to_FY24_pct",
        "Burn Multiple (Loss/Rev FY23)": "Burn_Multiple_Loss_per_Rev_FY23",
        "Monthly Burn ($M)": "Monthly_Burn_USD_M",
        "Valuation Haircut (%)": "Valuation_Haircut_pct",
        "Layoffs (% of workforce)": "Layoffs_pct_of_Workforce",
        "Peak Users (M)": "Peak_Users_M",
    }
    raw = raw.rename(columns={k: v for k, v in rename_map.items() if k in raw.columns})
    raw = raw.loc[:, ~raw.columns.duplicated()]

    if "Company" in raw.columns:
        raw = raw[raw["Company"].notna()]
        raw = raw[raw["Company"].astype(str).str.lower().ne("company")]

    numeric_candidates = [
        "#",
        "Founded",
        "Founded_Year",
        "Shutdown_Year",
        "Funding_USD_M",
        "Peak_Employees",
        "Peak_Valuation_USD_M",
        "Revenue_FY22_USD_M",
        "Revenue_FY23_USD_M",
        "Revenue_FY24_USD_M",
        "Loss_FY22_USD_M",
        "Loss_FY23_USD_M",
        "Loss_FY24_USD_M",
        "Burn_Multiple_Loss_per_Rev_FY23",
        "Monthly_Burn_USD_M",
        "Valuation_Haircut_pct",
        "Layoffs_pct_of_Workforce",
        "Peak_Users_M",
        "Years_Active",
    ]
    for col in numeric_candidates:
        if col in raw.columns:
            raw[col] = pd.to_numeric(raw[col], errors="coerce")

    if "Shutdown_Year" in raw.columns:
        raw["Shutdown_Year"] = pd.to_numeric(raw["Shutdown_Year"], errors="coerce").astype("Int64")

    raw["Analytic_Risk_Score"] = build_risk_score(raw)
    return raw.reset_index(drop=True)


def percentile_score(series: pd.Series, inverse: bool = False) -> pd.Series:
    score = series.rank(pct=True) * 100
    if inverse:
        score = 100 - score
    return score.fillna(score.median()).fillna(50)


def build_risk_score(df: pd.DataFrame) -> pd.Series:
    pieces: list[pd.Series] = []
    weights: list[float] = []
    config = [
        ("Burn_Multiple_Loss_per_Rev_FY23", 0.20, False),
        ("Monthly_Burn_USD_M", 0.18, False),
        ("Loss_FY24_USD_M", 0.16, False),
        ("Valuation_Haircut_pct", 0.18, False),
        ("Layoffs_pct_of_Workforce", 0.12, False),
        ("Rev_Change_FY23_to_FY24_pct", 0.16, True),
    ]
    for col, weight, inverse in config:
        if col in df.columns:
            pieces.append(percentile_score(pd.to_numeric(df[col], errors="coerce"), inverse=inverse))
            weights.append(weight)
    if not pieces:
        return pd.Series(np.nan, index=df.index)
    weighted = sum(piece * weight for piece, weight in zip(pieces, weights)) / sum(weights)
    return weighted.round(1)


def format_money(value: float) -> str:
    if pd.isna(value):
        return "n/a"
    if abs(value) >= 1000:
        return f"${value / 1000:,.1f}B"
    return f"${value:,.1f}M"


def available(df: pd.DataFrame, cols: list[str]) -> list[str]:
    return [col for col in cols if col in df.columns]


def apply_filters(df: pd.DataFrame) -> pd.DataFrame:
    st.sidebar.header("Filters")
    sectors = sorted(df.get("Sector", pd.Series(dtype=str)).dropna().unique())
    categories = sorted(df.get("Failure_Category", pd.Series(dtype=str)).dropna().unique())
    stages = sorted(df.get("Stage_at_Shutdown", pd.Series(dtype=str)).dropna().unique())
    states = sorted(df.get("HQ_State", pd.Series(dtype=str)).dropna().unique())

    selected_sectors = st.sidebar.multiselect("Sector (blank = all)", sectors)
    selected_categories = st.sidebar.multiselect("Failure category (blank = all)", categories)
    selected_stages = st.sidebar.multiselect("Stage (blank = all)", stages)
    selected_states = st.sidebar.multiselect("HQ state (blank = all)", states)

    filtered = df.copy()
    if selected_sectors and "Sector" in filtered:
        filtered = filtered[filtered["Sector"].isin(selected_sectors)]
    if selected_categories and "Failure_Category" in filtered:
        filtered = filtered[filtered["Failure_Category"].isin(selected_categories)]
    if selected_stages and "Stage_at_Shutdown" in filtered:
        filtered = filtered[filtered["Stage_at_Shutdown"].isin(selected_stages)]
    if selected_states and "HQ_State" in filtered:
        filtered = filtered[filtered["HQ_State"].isin(selected_states)]

    if "Shutdown_Year" in filtered.columns and filtered["Shutdown_Year"].notna().any():
        min_year = int(df["Shutdown_Year"].min())
        max_year = int(df["Shutdown_Year"].max())
        year_range = st.sidebar.slider("Shutdown year", min_year, max_year, (min_year, max_year))
        filtered = filtered[
            filtered["Shutdown_Year"].between(year_range[0], year_range[1], inclusive="both")
        ]

    if "Funding_USD_M" in filtered.columns and filtered["Funding_USD_M"].notna().any():
        max_funding = float(df["Funding_USD_M"].max())
        min_funding = st.sidebar.slider("Minimum funding ($M)", 0.0, max_funding, 0.0)
        filtered = filtered[filtered["Funding_USD_M"].fillna(0) >= min_funding]

    return filtered


def overview_tab(df: pd.DataFrame) -> None:
    st.subheader("Executive overview")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Companies", f"{len(df):,}")
    c2.metric("Funding consumed", format_money(df.get("Funding_USD_M", pd.Series(dtype=float)).sum()))
    c3.metric("Avg funding", format_money(df.get("Funding_USD_M", pd.Series(dtype=float)).mean()))
    c4.metric("Median years active", f"{df.get('Years_Active', pd.Series(dtype=float)).median():.1f}")

    chart_cols = st.columns(2)
    if {"Shutdown_Year", "Funding_USD_M"}.issubset(df.columns):
        by_year = (
            df.groupby("Shutdown_Year", dropna=True)
            .agg(Companies=("Company", "count"), Funding_USD_M=("Funding_USD_M", "sum"))
            .reset_index()
        )
        fig = px.bar(
            by_year,
            x="Shutdown_Year",
            y="Companies",
            text="Companies",
            title="Shutdown count by year",
            hover_data={"Funding_USD_M": ":,.1f"},
        )
        chart_cols[0].plotly_chart(fig, width="stretch")

    if {"Failure_Category", "Funding_USD_M"}.issubset(df.columns):
        by_category = (
            df.groupby("Failure_Category")
            .agg(Companies=("Company", "count"), Funding_USD_M=("Funding_USD_M", "sum"))
            .reset_index()
            .sort_values("Funding_USD_M", ascending=False)
        )
        fig = px.treemap(
            by_category,
            path=["Failure_Category"],
            values="Funding_USD_M",
            color="Companies",
            title="Capital lost by failure category",
        )
        chart_cols[1].plotly_chart(fig, width="stretch")

    if {"Company", "Funding_USD_M", "Sector"}.issubset(df.columns):
        top = df.nlargest(15, "Funding_USD_M").sort_values("Funding_USD_M")
        fig = px.bar(
            top,
            x="Funding_USD_M",
            y="Company",
            color="Sector",
            orientation="h",
            title="Top funded failed startups",
            labels={"Funding_USD_M": "Funding ($M)"},
        )
        st.plotly_chart(fig, width="stretch")


def sector_tab(df: pd.DataFrame) -> None:
    st.subheader("Sector analysis")
    metrics = available(
        df,
        [
            "Funding_USD_M",
            "Years_Active",
            "Peak_Valuation_USD_M",
            "Revenue_FY24_USD_M",
            "Loss_FY24_USD_M",
            "Burn_Multiple_Loss_per_Rev_FY23",
            "Monthly_Burn_USD_M",
            "Valuation_Haircut_pct",
            "Analytic_Risk_Score",
        ],
    )
    if "Sector" not in df.columns or not metrics:
        st.info("Sector or numeric columns are unavailable.")
        return

    aggregations = {"Company": "count"}
    aggregations.update({col: "median" for col in metrics})
    sector = df.groupby("Sector").agg(aggregations).rename(columns={"Company": "Companies"})
    if "Funding_USD_M" in df.columns:
        sector["Total_Funding_USD_M"] = df.groupby("Sector")["Funding_USD_M"].sum()
    sector = sector.reset_index().sort_values("Companies", ascending=False)

    fig = px.scatter(
        sector,
        x="Companies",
        y="Total_Funding_USD_M" if "Total_Funding_USD_M" in sector else metrics[0],
        size="Companies",
        color="Analytic_Risk_Score" if "Analytic_Risk_Score" in sector else metrics[0],
        hover_name="Sector",
        title="Sector concentration: count, capital, and risk",
    )
    st.plotly_chart(fig, width="stretch")

    left, right = st.columns(2)
    if "Funding_USD_M" in metrics:
        top_funding = sector.sort_values("Total_Funding_USD_M", ascending=False).head(12)
        left.plotly_chart(
            px.bar(
                top_funding,
                x="Total_Funding_USD_M",
                y="Sector",
                orientation="h",
                title="Highest capital-at-risk sectors",
            ),
            width="stretch",
        )
    if "Analytic_Risk_Score" in metrics:
        risk = sector.sort_values("Analytic_Risk_Score", ascending=False).head(12)
        right.plotly_chart(
            px.bar(
                risk,
                x="Analytic_Risk_Score",
                y="Sector",
                orientation="h",
                title="Highest median analytic risk score",
            ),
            width="stretch",
        )

    st.dataframe(sector, width="stretch")


def correlation_tab(df: pd.DataFrame) -> None:
    st.subheader("Correlation analysis")
    numeric_df = df.select_dtypes(include=[np.number]).drop(columns=["#"], errors="ignore")
    numeric_df = numeric_df.dropna(axis=1, how="all")
    if numeric_df.shape[1] < 2:
        st.info("Not enough numeric columns for correlation.")
        return

    corr = numeric_df.corr(numeric_only=True)
    fig = go.Figure(
        data=go.Heatmap(
            z=corr.values,
            x=corr.columns,
            y=corr.index,
            colorscale="RdBu",
            zmin=-1,
            zmax=1,
            colorbar={"title": "r"},
        )
    )
    fig.update_layout(title="Numeric correlation matrix", height=720)
    st.plotly_chart(fig, width="stretch")

    target = st.selectbox(
        "Show strongest relationships with",
        corr.columns,
        index=list(corr.columns).index("Funding_USD_M") if "Funding_USD_M" in corr.columns else 0,
    )
    relationships = (
        corr[target]
        .drop(labels=[target])
        .dropna()
        .sort_values(key=lambda x: x.abs(), ascending=False)
        .reset_index()
    )
    relationships.columns = ["Metric", "Correlation"]
    st.dataframe(relationships.head(12), width="stretch")


def build_model(
    model_df: pd.DataFrame,
    target_col: str,
    numeric_features: list[str],
    categorical_features: list[str],
) -> tuple[Pipeline, pd.DataFrame, pd.Series, pd.DataFrame, pd.Series, dict]:
    use_cols = numeric_features + categorical_features + [target_col]
    clean = model_df[use_cols].copy()
    clean[target_col] = pd.to_numeric(clean[target_col], errors="coerce")
    clean = clean.dropna(subset=[target_col])

    x = clean[numeric_features + categorical_features]
    y = clean[target_col]
    if len(clean) < 20:
        raise ValueError("Need at least 20 rows with a non-empty target for a reliable split.")

    test_size = 0.25 if len(clean) >= 40 else 0.30
    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=test_size, random_state=42
    )

    numeric_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    categorical_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", make_one_hot_encoder()),
        ]
    )
    preprocess = ColumnTransformer(
        transformers=[
            ("num", numeric_pipe, numeric_features),
            ("cat", categorical_pipe, categorical_features),
        ],
        remainder="drop",
    )

    algorithm = st.session_state.get("algorithm", "Random forest")
    estimator = (
        RandomForestRegressor(n_estimators=500, random_state=42, min_samples_leaf=3)
        if algorithm == "Random forest"
        else LinearRegression()
    )
    model = Pipeline(steps=[("preprocess", preprocess), ("model", estimator)])
    model.fit(x_train, y_train)
    preds = model.predict(x_test)

    metrics = {
        "R2": r2_score(y_test, preds),
        "MAE": mean_absolute_error(y_test, preds),
        "RMSE": float(np.sqrt(mean_squared_error(y_test, preds))),
    }
    return model, x_train, y_train, x_test, y_test, metrics


def model_tab(df: pd.DataFrame) -> None:
    st.subheader("Regression and predictive analytics")
    st.caption(
        "This file only contains failed companies, so this is not a survivor-vs-failure "
        "classifier. The model predicts outcomes within the failed-company population."
    )

    numeric_cols = [
        col
        for col in df.select_dtypes(include=[np.number]).columns
        if col not in {"#", "Founded_Year", "Shutdown_Year"}
    ]
    categorical_cols = available(df, ["Sector", "HQ_State", "Stage_at_Shutdown", "Failure_Category"])
    if len(numeric_cols) < 2:
        st.info("Not enough numeric columns for regression.")
        return

    default_target = "Funding_USD_M" if "Funding_USD_M" in numeric_cols else numeric_cols[0]
    target = st.selectbox(
        "Target variable",
        numeric_cols,
        index=numeric_cols.index(default_target),
    )
    candidate_features = [col for col in numeric_cols if col != target]
    default_features = available(
        df,
        [
            "Years_Active",
            "Peak_Employees",
            "Peak_Valuation_USD_M",
            "Revenue_FY23_USD_M",
            "Revenue_FY24_USD_M",
            "Loss_FY23_USD_M",
            "Loss_FY24_USD_M",
            "Burn_Multiple_Loss_per_Rev_FY23",
            "Monthly_Burn_USD_M",
            "Valuation_Haircut_pct",
            "Layoffs_pct_of_Workforce",
            "Peak_Users_M",
            "Analytic_Risk_Score",
        ],
    )
    default_features = [col for col in default_features if col != target]
    numeric_features = st.multiselect(
        "Numeric predictors",
        candidate_features,
        default=default_features[: min(10, len(default_features))],
    )
    categorical_features = st.multiselect(
        "Categorical predictors",
        categorical_cols,
        default=[col for col in ["Sector", "Stage_at_Shutdown", "Failure_Category"] if col in categorical_cols],
    )
    st.session_state["algorithm"] = st.radio(
        "Regression model",
        ["Random forest", "Linear regression"],
        horizontal=True,
    )

    if not numeric_features and not categorical_features:
        st.warning("Choose at least one predictor.")
        return

    try:
        model, x_train, y_train, x_test, y_test, metrics = build_model(
            df, target, numeric_features, categorical_features
        )
    except Exception as exc:
        st.error(str(exc))
        return

    c1, c2, c3 = st.columns(3)
    c1.metric("Test R2", f"{metrics['R2']:.3f}")
    c2.metric("MAE", f"{metrics['MAE']:,.2f}")
    c3.metric("RMSE", f"{metrics['RMSE']:,.2f}")

    pred_df = pd.DataFrame({"Actual": y_test, "Predicted": model.predict(x_test)})
    fig = px.scatter(pred_df, x="Actual", y="Predicted", title="Actual vs predicted")
    lo = float(np.nanmin([pred_df["Actual"].min(), pred_df["Predicted"].min()]))
    hi = float(np.nanmax([pred_df["Actual"].max(), pred_df["Predicted"].max()]))
    fig.add_trace(
        go.Scatter(
            x=[lo, hi],
            y=[lo, hi],
            mode="lines",
            name="Perfect prediction",
            line={"dash": "dash", "color": "gray"},
        )
    )
    st.plotly_chart(fig, width="stretch")

    estimator = model.named_steps["model"]
    feature_names = list(numeric_features)
    if categorical_features:
        cat_pipe = model.named_steps["preprocess"].named_transformers_["cat"]
        ohe = cat_pipe.named_steps["onehot"]
        feature_names += list(ohe.get_feature_names_out(categorical_features))

    if hasattr(estimator, "feature_importances_"):
        importance = pd.DataFrame(
            {"Feature": feature_names, "Importance": estimator.feature_importances_}
        ).sort_values("Importance", ascending=False)
        st.plotly_chart(
            px.bar(importance.head(20), x="Importance", y="Feature", orientation="h", title="Top drivers"),
            width="stretch",
        )
    elif hasattr(estimator, "coef_"):
        coef = pd.DataFrame({"Feature": feature_names, "Coefficient": estimator.coef_})
        coef["Abs"] = coef["Coefficient"].abs()
        coef = coef.sort_values("Abs", ascending=False).drop(columns="Abs")
        st.dataframe(coef.head(20), width="stretch")

    st.markdown("#### What-if prediction")
    with st.form("prediction_form"):
        values = {}
        cols = st.columns(3)
        for i, col in enumerate(numeric_features):
            series = pd.to_numeric(df[col], errors="coerce")
            min_val = float(series.quantile(0.05)) if series.notna().any() else 0.0
            max_val = float(series.quantile(0.95)) if series.notna().any() else 1.0
            med_val = float(series.median()) if series.notna().any() else 0.0
            if min_val == max_val:
                max_val = min_val + 1.0
            values[col] = cols[i % 3].number_input(col, value=med_val, min_value=min_val, max_value=max_val)
        for i, col in enumerate(categorical_features):
            choices = sorted(df[col].dropna().astype(str).unique())
            values[col] = cols[i % 3].selectbox(col, choices)
        submitted = st.form_submit_button("Predict")
    if submitted:
        one_row = pd.DataFrame([values])
        prediction = float(model.predict(one_row)[0])
        st.success(f"Predicted {target}: {prediction:,.2f}")


def data_tab(df: pd.DataFrame) -> None:
    st.subheader("Data explorer")
    search = st.text_input("Search company, sector, investors, or reason")
    shown = df.copy()
    if search:
        text_cols = shown.select_dtypes(include=["object", "string"]).columns
        mask = pd.Series(False, index=shown.index)
        for col in text_cols:
            mask |= shown[col].astype(str).str.contains(search, case=False, na=False)
        shown = shown[mask]
    st.dataframe(shown, width="stretch", height=520)
    st.download_button(
        "Download filtered CSV",
        shown.to_csv(index=False).encode("utf-8"),
        file_name="filtered_startup_analysis.csv",
        mime="text/csv",
    )


def main() -> None:
    st.set_page_config(page_title=APP_TITLE, layout="wide")
    st.title(APP_TITLE)

    excel_path = st.sidebar.text_input("Excel file path", value=str(DEFAULT_FILE))
    try:
        df = load_startup_data(excel_path)
    except Exception as exc:
        st.error(str(exc))
        st.stop()

    filtered = apply_filters(df)
    st.sidebar.write(f"Rows selected: {len(filtered):,} / {len(df):,}")

    tabs = st.tabs(
        [
            "Overview",
            "Sector analysis",
            "Correlation",
            "Regression and prediction",
            "Data explorer",
        ]
    )
    with tabs[0]:
        overview_tab(filtered)
    with tabs[1]:
        sector_tab(filtered)
    with tabs[2]:
        correlation_tab(filtered)
    with tabs[3]:
        model_tab(filtered)
    with tabs[4]:
        data_tab(filtered)


if __name__ == "__main__":
    main()
