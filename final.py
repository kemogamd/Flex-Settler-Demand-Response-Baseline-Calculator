# flex_settler_app.py
# Flex-Settler: Demand Response Baseline Calculator (v1.4)
# - Robust method parsing (High X of Y + Middle 8 of 10)
# - Safer datetime parsing
# - Similar-day search
# - Baseline calculation with 15-min alignment (nearest timestamp with tolerance)
# - Same-day adjustment with caps
# - Delivered energy, risk analysis, plotting, and exports

import io
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

# -----------------------------
# Page configuration
# -----------------------------
st.set_page_config(
    page_title="Flex-Settler: Baseline Calculator",
    page_icon="⚡",
    layout="wide",
)

# -----------------------------
# Title and description
# -----------------------------
st.title("⚡ Flex-Settler: Demand Response Baseline Calculator")
st.markdown(
    """
This tool calculates energy baselines for demand response events using state-of-the-art methodologies.
Upload your consumption data, define the event period, and get accurate baseline calculations with gaming risk analysis.
"""
)

# -----------------------------
# Sidebar: Methodology and Settings
# -----------------------------
with st.sidebar:
    st.header("📊 Methodology")
    st.info(
        """
**Selected Method: High 5 of 10 with Same-Day Adjustment**

This method:
- Uses the 5 highest consumption days from the last 10 similar days  
- Applies same-day calibration for accuracy  
- Minimizes bias and gaming opportunities  
- Proven accuracy in academic research (Elia study)
"""
    )

    st.header("⚙️ Settings")
    baseline_method = st.selectbox(
        "Baseline Method",
        ["High 5 of 10", "High 4 of 5", "High 10 of 15", "Middle 8 of 10"],
        index=0,
    )

    adjustment_type_label = st.selectbox(
        "Adjustment Type",
        ["Additive (kWh)", "Scalar (%)"],
        index=0,
    )

    adjustment_cap = st.slider(
        "Adjustment Cap (%)",
        min_value=0,
        max_value=50,
        value=20,
        help="Maximum allowed adjustment to prevent gaming",
    )

    adjustment_hours = st.slider(
        "Adjustment Window (hours before event)",
        min_value=1,
        max_value=6,
        value=3,
        help="Period used for same-day adjustment",
    )

# -----------------------------
# Helpers
# -----------------------------


def parse_method(method: str) -> tuple[int, int, bool]:
    """
    Return (x, y, use_middle) for the given method label.
    - use_middle=True indicates 'Middle 8 of 10' logic (exclude min and max).
    """
    mapping = {
        "High 5 of 10": (5, 10, False),
        "High 4 of 5": (4, 5, False),
        "High 10 of 15": (10, 15, False),
        "Middle 8 of 10": (8, 10, True),
    }
    return mapping.get(method, (5, 10, False))


def parse_datetime(dt_str) -> pd.Timestamp:
    """
    Parse datetime string with multiple format support.
    If none of the explicit formats work, fallback to pandas inference.
    """
    if isinstance(dt_str, (pd.Timestamp, datetime)):
        return pd.to_datetime(dt_str)

    if pd.isna(dt_str):
        return pd.NaT

    formats = [
        "%d/%m/%Y %H:%M",
        "%Y-%m-%d %H:%M:%S",
        "%Y-%m-%d %H:%M",
        "%d-%m-%Y %H:%M",
        "%m/%d/%Y %H:%M",
    ]

    for fmt in formats:
        try:
            return pd.to_datetime(dt_str, format=fmt)
        except Exception:
            continue

    # Let pandas try
    return pd.to_datetime(dt_str, errors="coerce")


def identify_similar_days(
    target_ts: pd.Timestamp, all_dates: pd.DatetimeIndex, num_days: int
) -> list[pd.Timestamp]:
    """
    Identify similar days (weekday vs weekend match) looking back from target date.
    - all_dates should be a DatetimeIndex of unique midnights or dates present in data.
    """
    similar_days: list[pd.Timestamp] = []
    target_is_weekend = target_ts.dayofweek >= 5

    # Look back up to 90 days or until we collect num_days
    max_lookback = 90
    current_date = target_ts.normalize() - timedelta(days=1)
    days_checked = 0

    # Normalize all_dates to midnight for robust comparison
    all_dates_norm = pd.to_datetime(all_dates.normalize())

    while len(similar_days) < num_days and days_checked < max_lookback:
        if current_date in all_dates_norm:
            if (current_date.dayofweek >= 5) == target_is_weekend:
                similar_days.append(current_date)
        current_date -= timedelta(days=1)
        days_checked += 1

    return similar_days


def build_event_intervals(
    event_start: pd.Timestamp, event_end: pd.Timestamp, freq: str = "15T"
) -> pd.DatetimeIndex:
    """
    Build event intervals for the specified frequency.
    We treat the end as inclusive to match typical settlement windows and your <= filtering.
    """
    if event_end < event_start:
        return pd.DatetimeIndex([])
    return pd.date_range(event_start, event_end, freq=freq)


def calculate_high_x_of_y(
    data: pd.Series,
    similar_days: list[pd.Timestamp],
    event_start: pd.Timestamp,
    event_end: pd.Timestamp,
    x: int,
    y: int,
    use_middle: bool,
) -> pd.Series:
    """
    Calculate baseline for the event window using High X of Y (or Middle 8 of 10) approach.
    - data: Series indexed by timestamp (15-min intervals), values are kWh.
    - similar_days: List of previous days (midnight timestamps) in descending recency.
    - For each event interval, gather the same time from each reference day and apply the selection logic.
    """
    # Ensure regular 15-min grid; enables nearest matching below
    data = data.copy().sort_index().resample("15T").mean()

    reference_days = similar_days[:y]
    event_intervals = build_event_intervals(event_start, event_end, freq="15T")

    values = []
    tol_seconds = 7.5 * 60  # ±7.5 minutes tolerance

    for interval in event_intervals:
        interval_time = interval.time()
        daily_values = []

        for ref_day in reference_days:
            ref_ts = pd.Timestamp.combine(ref_day.date(), interval_time)

            # Nearest timestamp lookup with tolerance
            idxer = data.index.get_indexer([ref_ts], method="nearest")
            if idxer.size > 0 and idxer[0] != -1:
                nearest_ts = data.index[idxer[0]]
                if abs((nearest_ts - ref_ts).total_seconds()) <= tol_seconds:
                    v = data.loc[nearest_ts]
                    if not pd.isna(v):
                        daily_values.append(float(v))

        if len(daily_values) == 0:
            values.append(0.0)
            continue

        if use_middle:
            # Middle 8 of 10: exclude min and max if possible
            if len(daily_values) >= 3:
                sorted_vals = sorted(daily_values)
                selected = sorted_vals[1:-1]
                values.append(
                    float(np.mean(selected)) if selected else float(
                        np.mean(daily_values))
                )
            else:
                values.append(float(np.mean(daily_values)))
        else:
            # High X of Y: take highest X if at least X points exist; otherwise average available
            if len(daily_values) >= x:
                selected = sorted(daily_values, reverse=True)[:x]
                values.append(float(np.mean(selected)))
            else:
                values.append(float(np.mean(daily_values)))

    return pd.Series(values, index=event_intervals, dtype=float)


def calculate_same_day_adjustment(
    data: pd.Series,
    baseline_series: pd.Series,
    event_start: pd.Timestamp,
    adjustment_window_hours: int,
    adj_type_label: str,
    cap_pct: int,
) -> tuple[float, str]:
    """
    Calculate same-day adjustment factor or offset.
    - For Additive (kWh): return offset (bounded by cap%) and type "Additive".
    - For Scalar (%): return multiplier (bounded by cap%) and type "Scalar".
    """
    adjustment_end = event_start
    adjustment_start = event_start - timedelta(hours=adjustment_window_hours)

    actual_adjustment = data[(data.index >= adjustment_start) & (
        data.index < adjustment_end)]
    baseline_adjustment = baseline_series.reindex(actual_adjustment.index)

    if (
        actual_adjustment.empty
        or baseline_adjustment.empty
        or baseline_adjustment.mean() == 0
        or pd.isna(baseline_adjustment.mean())
    ):
        if adj_type_label == "Additive (kWh)":
            return 0.0, "Additive"
        else:
            return 1.0, "Scalar"

    if adj_type_label == "Additive (kWh)":
        # Difference in kWh
        adj = float(actual_adjustment.mean() - baseline_adjustment.mean())
        max_adj = float(baseline_adjustment.mean() * (cap_pct / 100))
        adj = float(np.clip(adj, -max_adj, max_adj))
        return adj, "Additive"
    else:
        # Ratio
        ratio = float(actual_adjustment.mean() / baseline_adjustment.mean())
        cap_factor = 1 + (cap_pct / 100)
        ratio = float(np.clip(ratio, 1 / cap_factor, cap_factor))
        return ratio, "Scalar"


def apply_adjustment(baseline_series: pd.Series, factor: float, adj_type: str) -> pd.Series:
    if adj_type == "Additive":
        return baseline_series + factor
    else:
        return baseline_series * factor


def calculate_gaming_risk(data, event_start, adjustment_start, historical_mean):
    """Heuristic-based risk scoring on pre-event adjustment period (enhanced sensitivity)."""
    risk_score = 0
    risk_factors = []

    adj_data = data[(data.index >= adjustment_start)
                    & (data.index < event_start)]

    # Fallback if historical_mean too small
    eps = 1e-6
    if (historical_mean is None) or (np.isnan(historical_mean)) or (historical_mean < eps):
        # Use a rolling mean from earlier window (e.g., 6–12 hours before adjustment_start)
        backup_start = adjustment_start - timedelta(hours=12)
        backup_end = adjustment_start - timedelta(hours=6)
        backup = data[(data.index >= backup_start) & (data.index < backup_end)]
        if not backup.empty:
            historical_mean = float(backup.mean())
        else:
            historical_mean = float(adj_data.mean() or 0.0)

    if not adj_data.empty and historical_mean > eps:
        adj_mean = float(adj_data.mean())

        # Factor 1: stronger sensitivity
        ratio = adj_mean / historical_mean
        pct = (ratio - 1.0) * 100
        if ratio >= 1.6:
            risk_score += 55
            risk_factors.append(
                f"Adjustment period {pct:.1f}% above historical average")
        elif ratio >= 1.4:
            risk_score += 40
            risk_factors.append(
                f"Adjustment period {pct:.1f}% above historical average")
        elif ratio >= 1.25:
            risk_score += 25
            risk_factors.append(
                f"Adjustment period {pct:.1f}% above historical average")

        # Factor 2: last-hour spike with lower threshold and partial credit
        if len(adj_data) >= 4:
            last_hour = adj_data.iloc[-4:]
            first_hours = adj_data.iloc[:-4]
            if len(first_hours) > 0:
                spike_ratio = float(last_hour.mean()) / \
                    float(first_hours.mean() or eps)
                if spike_ratio >= 1.5:
                    risk_score += 30
                    risk_factors.append(
                        "Strong spike in last hour before event")
                elif spike_ratio >= 1.3:
                    risk_score += 20
                    risk_factors.append(
                        "Moderate spike in last hour before event")
                elif spike_ratio >= 1.15:
                    risk_score += 10
                    risk_factors.append(
                        "Small spike in last hour before event")

        # Factor 3: variability with lower threshold
        variability = float(adj_data.std())
        if variability >= historical_mean * 0.7:
            risk_score += 20
            risk_factors.append(
                "Very high variability during adjustment period")
        elif variability >= historical_mean * 0.5:
            risk_score += 12
            risk_factors.append("High variability during adjustment period")
        elif variability >= historical_mean * 0.35:
            risk_score += 6
            risk_factors.append(
                "Moderate variability during adjustment period")

    risk_score = min(int(risk_score), 100)

    if risk_score >= 60:
        risk_level = "🔴 HIGH"
        risk_color = "red"
    elif risk_score >= 30:
        risk_level = "🟡 MEDIUM"
        risk_color = "orange"
    else:
        risk_level = "🟢 LOW"
        risk_color = "green"

    return risk_score, risk_level, risk_factors, risk_color


# -----------------------------
# Main UI
# -----------------------------
col1, col2 = st.columns([2, 1])

with col1:
    st.header("📁 Data Upload")
    uploaded_file = st.file_uploader(
        "Upload consumption data (CSV or Excel)",
        type=["csv", "xlsx", "xls"],
        help="File should contain datetime and consumption columns",
    )

with col2:
    st.header("🗓️ Event Definition")
    event_date = st.date_input("Event Date", datetime.now().date())

    c1, c2 = st.columns(2)
    with c1:
        event_start_time = st.time_input(
            "Start Time", datetime.strptime("14:00", "%H:%M").time())
    with c2:
        event_end_time = st.time_input(
            "End Time", datetime.strptime("18:00", "%H:%M").time())

# -----------------------------
# Processing
# -----------------------------
if uploaded_file is not None:
    try:
        if uploaded_file.name.lower().endswith(".csv"):
            df_raw = pd.read_csv(uploaded_file)
        else:
            df_raw = pd.read_excel(uploaded_file)

        st.success(f"✅ File loaded successfully! {len(df_raw)} rows found.")

        st.header("🔗 Column Mapping")
        st.info("Map your data columns to the required fields")

        m1, m2, m3 = st.columns(3)
        with m1:
            datetime_col = st.selectbox(
                "Datetime Column", options=df_raw.columns.tolist(), index=0)
        with m2:
            consumption_col = st.selectbox(
                "Consumption Column (kWh)",
                options=df_raw.columns.tolist(),
                index=min(1, len(df_raw.columns) - 1),
            )
        with m3:
            activation_col = st.selectbox(
                "Activation Column (optional)", options=["None"] + df_raw.columns.tolist(), index=0
            )

        if st.button("🚀 Calculate Baseline", type="primary"):
            with st.spinner("Processing data and calculating baseline..."):
                try:
                    df = df_raw.copy()
                    df["datetime"] = df[datetime_col].apply(parse_datetime)
                    df["consumption"] = pd.to_numeric(
                        df[consumption_col], errors="coerce")
                    df = df.dropna(subset=["datetime", "consumption"])
                    df = df.set_index("datetime").sort_index()
                    consumption_data: pd.Series = df["consumption"].astype(
                        float)

                    # Ensure 15-minute grid throughout
                    consumption_data = consumption_data.resample("15T").mean()

                    event_start = pd.Timestamp.combine(
                        event_date, event_start_time)
                    event_end = pd.Timestamp.combine(
                        event_date, event_end_time)

                    x_val, y_val, use_middle = parse_method(baseline_method)

                    # Identify similar days based on days present in data
                    all_dates = pd.DatetimeIndex(
                        consumption_data.index.normalize().unique())
                    similar_days = identify_similar_days(
                        event_start, all_dates, num_days=y_val)
                    if len(similar_days) < y_val:
                        st.warning(
                            f"⚠️ Only {len(similar_days)} similar days found (need {y_val}). Results may be less accurate."
                        )

                    # Baseline (raw)
                    baseline_raw = calculate_high_x_of_y(
                        consumption_data,
                        similar_days,
                        event_start,
                        event_end,
                        x=x_val,
                        y=y_val,
                        use_middle=use_middle,
                    )

                    # Historical mean (before event)
                    historical_mean = float(
                        consumption_data[consumption_data.index <
                                         event_start].mean() or 0.0
                    )

                    # Adjustment
                    adjustment_start = event_start - \
                        timedelta(hours=adjustment_hours)
                    adjustment_factor, adj_type_used = calculate_same_day_adjustment(
                        consumption_data,
                        baseline_raw,
                        event_start,
                        adjustment_hours,
                        adjustment_type_label,
                        adjustment_cap,
                    )

                    baseline_adjusted = apply_adjustment(
                        baseline_raw, adjustment_factor, adj_type_used)

                    # Actual consumption during event reindexed to baseline times for comparability
                    actual_event = consumption_data[
                        (consumption_data.index >= event_start) & (
                            consumption_data.index <= event_end)
                    ]
                    actual_event_aligned = actual_event.reindex(
                        baseline_adjusted.index)

                    # Delivered energy (align indices; missing actual -> treated as 0)
                    actual_for_calc = actual_event_aligned.fillna(0.0)
                    delivered_energy = float(
                        (baseline_adjusted - actual_for_calc).sum())
                    baseline_total = float(baseline_adjusted.sum())
                    actual_total = float(actual_for_calc.sum())
                    delivery_pct = (
                        delivered_energy / baseline_total * 100) if baseline_total > 0 else 0.0

                    # Risk analysis
                    risk_score, risk_level, risk_factors, risk_color = calculate_gaming_risk(
                        consumption_data, event_start, adjustment_start, historical_mean
                    )

                    st.success("✅ Baseline calculation completed!")

                    # -----------------------------
                    # Results Dashboard
                    # -----------------------------
                    st.header("📊 Results Dashboard")
                    k1, k2, k3, k4 = st.columns(4)
                    with k1:
                        st.metric(
                            "Calculated Baseline",
                            f"{baseline_total:.2f} kWh",
                            help="Total baseline energy for event period",
                        )
                    with k2:
                        st.metric(
                            "Actual Consumption",
                            f"{actual_total:.2f} kWh",
                            help="Total actual energy consumed during event",
                        )
                    with k3:
                        st.metric(
                            "Delivered Energy",
                            f"{delivered_energy:.2f} kWh",
                            delta=f"{delivery_pct:.1f}%",
                            help="Energy saved during demand response event",
                        )
                    with k4:
                        # Use text with color for clarity
                        st.markdown(
                            f"<div><b>Gaming Risk:</b> <span style='color:{risk_color}'>{risk_level} ({risk_score}/100)</span></div>",
                            unsafe_allow_html=True,
                        )

                    # -----------------------------
                    # Technical Details
                    # -----------------------------
                    st.header("🔧 Technical Details")
                    d1, d2 = st.columns(2)
                    with d1:
                        st.markdown(
                            f"""
- Method Used: {baseline_method}  
- Adjustment Type: {adj_type_used}  
- Adjustment Value: {adjustment_factor:.4f} {'kWh' if adj_type_used == 'Additive' else '×'}  
- Adjustment Cap: ±{adjustment_cap}%  
- Similar Days Used: {len(similar_days)}
"""
                        )
                    with d2:
                        st.markdown(
                            f"""
- Event Duration: {(event_end - event_start).total_seconds()/3600:.1f} hours  
- Baseline (Raw): {baseline_raw.sum():.2f} kWh  
- Baseline (Adjusted): {baseline_total:.2f} kWh  
- Historical Average: {historical_mean:.3f} kWh per interval  
- Adjustment Window: {adjustment_hours} hours before event
"""
                        )

                    if risk_factors:
                        st.subheader("⚠️ Gaming Risk Analysis")
                        st.markdown(
                            f"<b>Risk Score:</b> <span style='color:{risk_color}'>{risk_score}/100 - {risk_level}</span>",
                            unsafe_allow_html=True,
                        )
                        st.markdown("**Risk Factors Detected:**")
                        for f in risk_factors:
                            st.markdown(f"- {f}")

                    # -----------------------------
                    # Visualization
                    # -----------------------------
                    st.header("📈 Visualization")

                    # Build area shading using common index to avoid mismatch
                    area_index = baseline_adjusted.index.intersection(
                        actual_event.index)
                    area_baseline = baseline_adjusted.reindex(area_index)
                    area_actual = actual_event.reindex(area_index)

                    fig = make_subplots(
                        rows=2,
                        cols=1,
                        subplot_titles=(
                            "Baseline vs Actual Consumption", "Gaming Risk Analysis"),
                        row_heights=[0.7, 0.3],
                        vertical_spacing=0.12,
                    )

                    # Plot 1: Main comparison
                    fig.add_trace(
                        go.Scatter(
                            x=baseline_adjusted.index,
                            y=baseline_adjusted.values,
                            name="Adjusted Baseline",
                            line=dict(color="blue", width=2),
                            mode="lines",
                        ),
                        row=1,
                        col=1,
                    )
                    fig.add_trace(
                        go.Scatter(
                            x=actual_event.index,
                            y=actual_event.values,
                            name="Actual Consumption",
                            line=dict(color="red", width=2),
                            mode="lines+markers",
                        ),
                        row=1,
                        col=1,
                    )
                    fig.add_trace(
                        go.Scatter(
                            x=baseline_raw.index,
                            y=baseline_raw.values,
                            name="Raw Baseline (no adjustment)",
                            line=dict(color="lightblue", width=1, dash="dash"),
                            mode="lines",
                        ),
                        row=1,
                        col=1,
                    )

                    # Area for delivered energy (only where both series exist)
                    if not area_index.empty:
                        fig.add_trace(
                            go.Scatter(
                                x=area_baseline.index.tolist(
                                ) + area_baseline.index.tolist()[::-1],
                                y=area_baseline.values.tolist(
                                ) + area_actual.values.tolist()[::-1],
                                fill="toself",
                                fillcolor="rgba(0,255,0,0.2)",
                                line=dict(color="rgba(255,255,255,0)"),
                                name="Delivered Energy",
                                showlegend=True,
                            ),
                            row=1,
                            col=1,
                        )

                    # Plot 2: Risk analysis timeline
                    lookback_hours = 12
                    adjustment_start = event_start - \
                        timedelta(hours=adjustment_hours)
                    lookback_start = adjustment_start - \
                        timedelta(hours=lookback_hours)
                    analysis_data = consumption_data[
                        (consumption_data.index >= lookback_start) & (
                            consumption_data.index <= event_end)
                    ]

                    fig.add_trace(
                        go.Scatter(
                            x=analysis_data.index,
                            y=analysis_data.values,
                            name="Consumption Pattern",
                            line=dict(color="purple", width=2),
                            mode="lines",
                        ),
                        row=2,
                        col=1,
                    )

                    # Vertical markers (use datetime directly)
                    adj_x = (adjustment_start.to_pydatetime() if hasattr(
                        adjustment_start, "to_pydatetime") else adjustment_start)
                    evt_x = (event_start.to_pydatetime() if hasattr(
                        event_start, "to_pydatetime") else event_start)

                    adj_x_str = adj_x.isoformat()
                    evt_x_str = evt_x.isoformat()

                    # Add vertical line shapes spanning the subplot's x-axis
                    fig.add_shape(
                        type="line",
                        x0=adj_x_str,
                        x1=adj_x_str,
                        y0=0,
                        y1=1,
                        xref="x2",
                        yref="y2 domain",  # row 2, col 1 axes
                        line=dict(color="orange", width=2, dash="dash"),
                    )
                    fig.add_shape(
                        type="line",
                        x0=evt_x_str,
                        x1=evt_x_str,
                        y0=0,
                        y1=1,
                        xref="x2",
                        yref="y2 domain",
                        line=dict(color="red", width=2, dash="dash"),
                    )

                    # Add text annotations near the top of subplot 2
                    fig.add_annotation(
                        x=adj_x_str,
                        y=1.02,
                        xref="x2",
                        yref="y2 domain",
                        text="Adjustment Start",
                        showarrow=False,
                        font=dict(color="orange"),
                    )
                    fig.add_annotation(
                        x=evt_x_str,
                        y=1.02,
                        xref="x2",
                        yref="y2 domain",
                        text="Event Start",
                        showarrow=False,
                        font=dict(color="red"),
                    )

                    # Horizontal reference (historical mean)
                    fig.add_hline(
                        y=historical_mean,
                        line_dash="dot",
                        line_color="gray",
                        annotation_text=f"Historical Avg: {historical_mean:.3f}",
                        row=2,
                        col=1,
                    )

                    fig.update_xaxes(title_text="Time", row=2, col=1)
                    fig.update_yaxes(
                        title_text="Consumption (kWh)", row=1, col=1)
                    fig.update_yaxes(
                        title_text="Consumption (kWh)", row=2, col=1)
                    fig.update_layout(
                        height=800,
                        showlegend=True,
                        hovermode="x unified",
                        title_text=f"Demand Response Event Analysis - {event_date}",
                    )

                    st.plotly_chart(fig, use_container_width=True)

                    # -----------------------------
                    # Export Results
                    # -----------------------------
                    st.header("💾 Export Results")

                    results_df = pd.DataFrame(
                        {
                            "Timestamp": baseline_adjusted.index,
                            "Raw_Baseline_kWh": baseline_raw.values,
                            "Adjusted_Baseline_kWh": baseline_adjusted.values,
                            "Actual_Consumption_kWh": actual_event_aligned.values,
                        }
                    )
                    results_df["Delivered_Energy_kWh"] = (
                        results_df["Adjusted_Baseline_kWh"] -
                        results_df["Actual_Consumption_kWh"]
                    )

                    summary_df = pd.DataFrame(
                        {
                            "Metric": [
                                "Event Date",
                                "Event Start",
                                "Event End",
                                "Method Used",
                                "Adjustment Type",
                                "Adjustment Value",
                                "Total Baseline (Raw)",
                                "Total Baseline (Adjusted)",
                                "Total Actual Consumption",
                                "Total Delivered Energy",
                                "Delivery Percentage",
                                "Gaming Risk Score",
                                "Gaming Risk Level",
                            ],
                            "Value": [
                                event_date.strftime("%Y-%m-%d"),
                                event_start_time.strftime("%H:%M"),
                                event_end_time.strftime("%H:%M"),
                                baseline_method,
                                adj_type_used,
                                f"{adjustment_factor:.4f} {'kWh' if adj_type_used == 'Additive' else '×'}",
                                f"{baseline_raw.sum():.2f} kWh",
                                f"{baseline_total:.2f} kWh",
                                f"{actual_total:.2f} kWh",
                                f"{delivered_energy:.2f} kWh",
                                f"{delivery_pct:.1f}%",
                                f"{risk_score}/100",
                                risk_level,
                            ],
                        }
                    )

                    output = io.BytesIO()
                    try:
                        with pd.ExcelWriter(output, engine="openpyxl") as writer:
                            summary_df.to_excel(
                                writer, sheet_name="Summary", index=False)
                            results_df.to_excel(
                                writer, sheet_name="Detailed_Results", index=False)
                            if risk_factors:
                                pd.DataFrame({"Risk_Factors": risk_factors}).to_excel(
                                    writer, sheet_name="Risk_Analysis", index=False
                                )
                        excel_data = output.getvalue()
                        c1, c2 = st.columns(2)
                        with c1:
                            st.download_button(
                                label="📥 Download Excel Report",
                                data=excel_data,
                                file_name=f"baseline_results_{event_date.strftime('%Y%m%d')}.xlsx",
                                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                            )
                    except Exception:
                        st.warning(
                            "openpyxl not installed; Excel export unavailable.")

                    csv_data = results_df.to_csv(index=False)
                    st.download_button(
                        label="📥 Download CSV Data",
                        data=csv_data,
                        file_name=f"baseline_data_{event_date.strftime('%Y%m%d')}.csv",
                        mime="text/csv",
                    )

                except Exception as e:
                    st.error(f"❌ Error processing data: {e}")
                    st.exception(e)

    except Exception as e:
        st.error(f"❌ Error loading file: {e}")
        st.exception(e)

else:
    # -----------------------------
    # Instructions
    # -----------------------------
    st.info(
        """
### 👋 Welcome to Flex-Settler!

**To get started:**
1. Upload your data file (CSV or Excel)  
2. Map your columns  
3. Define the event period (date and time)  
4. Click 'Calculate Baseline' to see results

The tool will:
- Calculate an accurate baseline using proven methodologies
- Apply same-day adjustments
- Analyze gaming risks
- Generate reports and visualizations

**Expected data format:**
- A datetime column (various formats supported)
- A consumption/production column (in kWh)
- Optional: activation status column
"""
    )

    with st.expander("📋 See sample data format"):
        st.dataframe(
            pd.DataFrame(
                {
                    "Date_time": ["01/01/2023 00:15", "01/01/2023 00:30", "01/01/2023 00:45"],
                    "Consumption [kWh]": [0.403, 0.338, 0.342],
                    "Activation": ["W", "W", "W"],
                    "Solar_irradiance": [0, 0, 0],
                }
            )
        )

# -----------------------------
# Footer
# -----------------------------
st.markdown("---")
st.markdown(
    """
<div style='text-align: center'>
    <p><strong>Flex-Settler</strong> | Baseline Calculator v1.4</p>
    <p>Built with Streamlit | Methodology based on peer-reviewed research</p>
</div>
""",
    unsafe_allow_html=True,
)
