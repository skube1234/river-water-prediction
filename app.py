
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go

# 페이지 기본 설정
st.set_page_config(
    page_title="중랑천 AI 수위 예측",
    page_icon="🌊",
    layout="wide"
)

st.title("🌊 중랑천 AI 수위 예측 시스템")
st.caption(
    "의정부 신곡교 | LSTM · Bi-LSTM 기반 "
    "하천 수위 예측 및 홍수 위험도 평가"
)

# CSV 업로드
uploaded = st.file_uploader(
    "예측 결과 CSV 파일 업로드",
    type=["csv", "gz"]
)

# CSV가 없으면 시연용 데이터 사용
if uploaded is None:
    st.warning(
        "현재는 가상의 시연용 데이터를 표시합니다. "
        "실제 연구 결과가 아닙니다."
    )

    times = pd.date_range(
        "2025-08-13 00:00",
        periods=72,
        freq="h"
    )

    x = np.arange(72)
    actual = 0.8 + 4.4 * np.exp(-((x-30)/6)**2)

    rows = []
    for model in ["LSTM", "BiLSTM"]:
        for horizon in [1, 3, 6, 12]:
            shift = horizon / 3
            peak = 4.1 if model == "LSTM" else 4.2
            pred = 0.8 + peak * np.exp(
                -((x-30-shift)/6.5)**2
            )

            for t, a, p in zip(times, actual, pred):
                rows.append({
                    "target_time": t,
                    "observed": a,
                    "predicted": p,
                    "model": model,
                    "horizon_h": horizon
                })

    df = pd.DataFrame(rows)

else:
    df = pd.read_csv(uploaded)

# 데이터 형식 확인
required = [
    "target_time", "observed",
    "predicted", "model", "horizon_h"
]

if not all(c in df.columns for c in required):
    st.error("CSV 파일의 열 이름을 확인해 주세요.")
    st.stop()

df["target_time"] = pd.to_datetime(df["target_time"])
df["observed"] = pd.to_numeric(
    df["observed"], errors="coerce"
)
df["predicted"] = pd.to_numeric(
    df["predicted"], errors="coerce"
)
df["horizon_h"] = pd.to_numeric(
    df["horizon_h"], errors="coerce"
)
df = df.dropna(subset=required)

if df.empty:
    st.error("표시할 수 있는 데이터가 없습니다.")
    st.stop()

# 사이드바
st.sidebar.header("⚙️ 분석 조건")

model = st.sidebar.selectbox(
    "예측 모델",
    sorted(df["model"].unique())
)

horizon = st.sidebar.selectbox(
    "예측시간",
    sorted(df["horizon_h"].unique()),
    format_func=lambda h: f"{int(h)}시간 후"
)

data = df[
    (df["model"] == model) &
    (df["horizon_h"] == horizon)
].copy()

# 여러 시드의 예측값 평균
data = data.groupby(
    "target_time", as_index=False
).agg(
    observed=("observed", "mean"),
    predicted=("predicted", "mean")
).sort_values("target_time")

start = data["target_time"].min().date()
end = data["target_time"].max().date()

period = st.sidebar.date_input(
    "조회 기간",
    value=(start, end),
    min_value=start,
    max_value=end
)

if len(period) != 2:
    st.stop()

data = data[
    (data["target_time"].dt.date >= period[0]) &
    (data["target_time"].dt.date <= period[1])
]

if data.empty:
    st.warning("선택한 기간에 데이터가 없습니다.")
    st.stop()

# 성능 지표 계산
actual = data["observed"].to_numpy()
pred = data["predicted"].to_numpy()

mae = np.mean(np.abs(actual - pred))
rmse = np.sqrt(np.mean((actual - pred)**2))

ss = np.sum((actual - actual.mean())**2)
r2 = (
    1 - np.sum((actual-pred)**2)/ss
    if ss > 0 else np.nan
)

# 성능 지표 표시
st.subheader("📊 AI 모델 예측 성능")

c1, c2, c3 = st.columns(3)
c1.metric("MAE", f"{mae:.4f} m")
c2.metric("RMSE", f"{rmse:.4f} m")
c3.metric("R²", f"{r2:.4f}")

# 수위 그래프
st.subheader("🌊 실제 수위와 AI 예측 수위")

fig = go.Figure()

fig.add_trace(go.Scatter(
    x=data["target_time"],
    y=actual,
    name="실제 관측 수위",
    line=dict(color="#2563eb", width=3)
))

fig.add_trace(go.Scatter(
    x=data["target_time"],
    y=pred,
    name=f"{model} 예측 수위",
    line=dict(color="#ef4444", width=2)
))

# 홍수 위험수위 기준선
thresholds = [
    ("관심", 2.60, "#eab308"),
    ("주의", 5.10, "#f97316"),
    ("경계", 6.00, "#ef4444"),
    ("심각", 7.00, "#991b1b")
]

for name, level, color in thresholds:
    fig.add_hline(
        y=level,
        line_dash="dash",
        line_color=color,
        annotation_text=name
    )

fig.update_layout(
    height=500,
    xaxis_title="날짜 및 시간",
    yaxis_title="수위 (m)",
    hovermode="x unified"
)

st.plotly_chart(fig, use_container_width=True)

# 홍수 위험도 평가
st.subheader("🚨 AI 예측 홍수 위험도")

level = float(pred[-1])

if level >= 7.0:
    risk = "심각"
elif level >= 6.0:
    risk = "경계"
elif level >= 5.10:
    risk = "주의"
elif level >= 2.60:
    risk = "관심"
else:
    risk = "기준 미만"

c1, c2 = st.columns(2)

with c1:
    st.metric("선택 기간 마지막 예측 수위", f"{level:.2f} m")

with c2:
    st.metric("위험 단계", risk)

st.info(
    "본 시스템은 과거 관측자료 기반 연구용 "
    "시각화 시스템입니다. 실시간 홍수 예경보가 "
    "아니며, 위험수위 기준면 일치 여부를 "
    "확인한 후 위험도를 해석해야 합니다."
)
