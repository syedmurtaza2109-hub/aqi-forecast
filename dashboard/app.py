"""
dashboard/app.py
─────────────────
Streamlit web dashboard.

Run:  streamlit run dashboard/app.py

Shows:
  • Current AQI + level badge
  • 3-day hourly forecast chart
  • Daily summary table
  • Historical trend (7 days)
  • Pollutant breakdown
  • SHAP feature importance bar chart
  • Hazard alerts
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import streamlit as st
import requests
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime
import config

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title=f"AQI Forecast — {config.CITY_NAME}",
    page_icon="🌫️",
    layout="wide",
    initial_sidebar_state="expanded",
)

API_BASE = os.getenv("API_BASE_URL", "http://localhost:5000")

# ── Helpers ───────────────────────────────────────────────────────────────────

def api_get(path, params=None):
    try:
        r = requests.get(f"{API_BASE}{path}", params=params, timeout=20)
        r.raise_for_status()
        return r.json()
    except Exception as e:
        st.error(f"API error ({path}): {e}")
        return None


def aqi_badge(value, level, color):
    st.markdown(
        f"""
        <div style="background:{color};border-radius:12px;padding:20px 30px;
                    display:inline-block;text-align:center;min-width:160px;">
            <div style="font-size:3rem;font-weight:800;color:{'#000' if color in ('#ffff00','#00e400') else '#fff'};
                        line-height:1;">{int(value)}</div>
            <div style="font-size:1rem;font-weight:600;color:{'#000' if color in ('#ffff00','#00e400') else '#fff'};">
                {level}
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("🌫️ AQI Forecast")
    st.markdown(f"**City:** {config.CITY_NAME}")
    st.markdown(f"**Lat/Lon:** {config.CITY_LAT}, {config.CITY_LON}")
    st.divider()

    forecast_days = st.slider("Forecast horizon (days)", 1, 3, 3)
    history_days  = st.slider("History to show (days)", 1, 14, 7)
    show_shap     = st.checkbox("Show SHAP explanations", value=True)

    if st.button("🔄 Refresh data"):
        st.cache_data.clear()
        st.rerun()

    st.divider()
    st.caption("Data: AQICN + OpenWeatherMap")
    st.caption("Model: Hopsworks Feature Store")
    st.caption(f"Updated: {datetime.utcnow().strftime('%H:%M UTC')}")


# ── Main ──────────────────────────────────────────────────────────────────────
st.title(f"🌫️ Air Quality Forecast — {config.CITY_NAME}")

# ── Current conditions ────────────────────────────────────────────────────────
st.subheader("Current Conditions")
current = api_get("/api/current")

if current:
    col1, col2, col3, col4, col5 = st.columns([2, 1, 1, 1, 1])
    with col1:
        aqi_badge(current["aqi"], current["level"], current["color"])
    with col2:
        st.metric("PM2.5",       f"{current['pm25']} µg/m³")
        st.metric("PM10",        f"{current['pm10']} µg/m³")
    with col3:
        st.metric("O₃",         f"{current['o3']} ppb")
        st.metric("NO₂",        f"{current['no2']} ppb")
    with col4:
        st.metric("Temperature", f"{current['temperature']}°C")
        st.metric("Humidity",    f"{current['humidity']}%")
    with col5:
        st.metric("Wind",        f"{current['wind_speed']} m/s")

    # ── Hazard alert ─────────────────────────────────────────────────────────
    if current["aqi"] > 150:
        st.error(
            f"⚠️ **AIR QUALITY ALERT** — AQI is {int(current['aqi'])} ({current['level']}). "
            "Avoid prolonged outdoor activity. Wear N95 masks outdoors."
        )
    elif current["aqi"] > 100:
        st.warning(
            f"⚠️ AQI is {int(current['aqi'])} ({current['level']}). "
            "Sensitive groups should limit outdoor exposure."
        )
    else:
        st.success(f"✅ Air quality is **{current['level']}** (AQI {int(current['aqi'])})")

st.divider()

# ── 3-Day Forecast ────────────────────────────────────────────────────────────
st.subheader(f"📈 {forecast_days}-Day Hourly Forecast")
forecast_data = api_get("/api/forecast", {"hours": forecast_days * 24})

if forecast_data:
    df_fc = pd.DataFrame(forecast_data)
    df_fc["datetime"] = pd.to_datetime(df_fc["datetime"])

    # Colour each bar by AQI level
    fig = go.Figure()
    for _, row in df_fc.iterrows():
        fig.add_trace(go.Bar(
            x=[row["datetime"]],
            y=[row["aqi_predicted"]],
            marker_color=row["color"],
            name=row["level"],
            showlegend=False,
            hovertemplate=(
                f"<b>{row['datetime'].strftime('%a %b %d %H:%M')}</b><br>"
                f"AQI: {row['aqi_predicted']}<br>"
                f"Level: {row['level']}<extra></extra>"
            ),
        ))

    # AQI threshold lines
    for val, label, color in [(50, "Good", "#00e400"), (100, "Moderate", "#ffff00"),
                               (150, "Unhealthy (Sens)", "#ff7e00"), (200, "Unhealthy", "#ff0000")]:
        fig.add_hline(y=val, line_dash="dot", line_color=color,
                      annotation_text=label, annotation_position="right")

    fig.update_layout(
        xaxis_title="Date / Time",
        yaxis_title="AQI",
        height=380,
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        bargap=0.1,
        margin=dict(t=10, b=10),
    )
    st.plotly_chart(fig, use_container_width=True)

    # Daily summary table
    st.subheader("Daily Summary")
    daily = df_fc.groupby("date")["aqi_predicted"].agg(["mean", "min", "max"]).reset_index()
    daily.columns = ["Date", "Avg AQI", "Min AQI", "Max AQI"]
    daily["Avg AQI"] = daily["Avg AQI"].round(1)
    daily["Min AQI"] = daily["Min AQI"].round(1)
    daily["Max AQI"] = daily["Max AQI"].round(1)
    daily["Level"] = daily["Avg AQI"].apply(lambda v: config.get_aqi_level(v)[0])

    def highlight_aqi(row):
        _, color = config.get_aqi_level(row["Avg AQI"])
        text_color = "#000" if color in ("#ffff00", "#00e400") else "#fff"
        return [f"background-color:{color};color:{text_color}"] * len(row)

    st.dataframe(daily.style.apply(highlight_aqi, axis=1), use_container_width=True, hide_index=True)

st.divider()

# ── Historical trend ──────────────────────────────────────────────────────────
st.subheader(f"📊 Historical AQI — Past {history_days} Days")
hist_data = api_get("/api/history", {"days": history_days})

if hist_data:
    df_h = pd.DataFrame(hist_data)
    df_h["datetime"] = pd.to_datetime(df_h["datetime"])

    col_l, col_r = st.columns(2)

    with col_l:
        fig_hist = px.line(df_h, x="datetime", y="aqi",
                           title="AQI over time",
                           labels={"aqi": "AQI", "datetime": ""})
        fig_hist.update_traces(line_color="#4f8ef7")
        fig_hist.update_layout(height=300, margin=dict(t=30, b=10),
                                plot_bgcolor="rgba(0,0,0,0)",
                                paper_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig_hist, use_container_width=True)

    with col_r:
        if "temperature" in df_h.columns and "humidity" in df_h.columns:
            fig_wx = go.Figure()
            fig_wx.add_trace(go.Scatter(x=df_h["datetime"], y=df_h["temperature"],
                                         name="Temp (°C)", line=dict(color="#ff7f50")))
            fig_wx.add_trace(go.Scatter(x=df_h["datetime"], y=df_h["humidity"],
                                         name="Humidity (%)", line=dict(color="#6495ed"),
                                         yaxis="y2"))
            fig_wx.update_layout(
                title="Temperature & Humidity",
                yaxis=dict(title="°C"),
                yaxis2=dict(title="%", overlaying="y", side="right"),
                height=300, margin=dict(t=30, b=10),
                plot_bgcolor="rgba(0,0,0,0)",
                paper_bgcolor="rgba(0,0,0,0)",
                legend=dict(orientation="h"),
            )
            st.plotly_chart(fig_wx, use_container_width=True)

    # Pollutant breakdown
    if all(c in df_h.columns for c in ["pm25", "pm10"]):
        st.subheader("Pollutant Breakdown")
        poll_cols = [c for c in ["pm25", "pm10"] if c in df_h.columns]
        df_poll = df_h[["datetime"] + poll_cols].melt("datetime",
                                                       var_name="Pollutant",
                                                       value_name="Value")
        fig_p = px.line(df_poll, x="datetime", y="Value", color="Pollutant",
                        labels={"Value": "µg/m³", "datetime": ""},
                        color_discrete_sequence=["#e05e5e", "#5e9de0"])
        fig_p.update_layout(height=260, margin=dict(t=10, b=10),
                             plot_bgcolor="rgba(0,0,0,0)",
                             paper_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig_p, use_container_width=True)

st.divider()

# ── SHAP explanations ─────────────────────────────────────────────────────────
if show_shap:
    st.subheader("🔍 Feature Importance (SHAP)")
    shap_data = api_get("/api/shap")
    if shap_data:
        df_shap = pd.DataFrame(shap_data).head(15)
        fig_shap = px.bar(df_shap, x="importance", y="feature", orientation="h",
                          title="Top 15 features driving AQI predictions",
                          labels={"importance": "Mean |SHAP value|", "feature": ""},
                          color="importance", color_continuous_scale="Blues")
        fig_shap.update_layout(height=420, margin=dict(t=40, b=10),
                                yaxis=dict(autorange="reversed"),
                                plot_bgcolor="rgba(0,0,0,0)",
                                paper_bgcolor="rgba(0,0,0,0)",
                                coloraxis_showscale=False)
        st.plotly_chart(fig_shap, use_container_width=True)
        st.caption(
            "SHAP (SHapley Additive exPlanations) measures how much each feature "
            "contributes to the model's predictions. Higher = more influential."
        )

# ── AQI Scale legend ──────────────────────────────────────────────────────────
st.divider()
st.subheader("AQI Scale Reference")
cols = st.columns(len(config.AQI_LEVELS))
for i, (lo, hi, label, color) in enumerate(config.AQI_LEVELS):
    text_c = "#000" if color in ("#ffff00", "#00e400") else "#fff"
    cols[i].markdown(
        f'<div style="background:{color};border-radius:8px;padding:10px;'
        f'text-align:center;color:{text_c};">'
        f"<b>{lo}–{hi}</b><br>{label}</div>",
        unsafe_allow_html=True,
    )
