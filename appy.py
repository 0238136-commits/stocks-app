import os
from datetime import date, timedelta

import pandas as pd
import yfinance as yf
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import streamlit as st
from dotenv import load_dotenv

import appy  # nuestro módulo auxiliar

# ==============================
# Branding / Disclaimer
# ==============================
AUTHOR_NAME = "Esteban González Guerra"
PERSONAL_BRAND = "TuMarca"
DISCLAIMER = f"© {date.today().year} {AUTHOR_NAME} — Todos los derechos reservados. Marca personal: {PERSONAL_BRAND}.\n\nLa información presentada es con fines educativos y no constituye asesoría financiera."

# ==============================
# Configuración inicial
# ==============================
load_dotenv(override=True)
st.set_page_config(page_title="Stocks Ticket App — OHLCV", page_icon="📈", layout="wide")
sns.set_theme(style="whitegrid")

st.title("📈 Stocks Ticket App — OHLCV con Seaborn + Gemini")
st.caption("Crea gráficos OHLCV, traduce descripciones con Gemini y exporta datos.")

# ==============================
# Sidebar — Parámetros
# ==============================
with st.sidebar:
    st.header("Parámetros")
    ticker = st.text_input("Ticker (ej. AAPL, MSFT, TSLA)", value="AAPL").strip()
    col1, col2 = st.columns(2)
    with col1:
        start = st.date_input("Inicio", value=date.today() - timedelta(days=180))
    with col2:
        end = st.date_input("Fin", value=date.today())
    interval = st.selectbox("Intervalo", ["1d", "1wk", "1mo"], index=0)
    btn_load = st.button("Cargar datos")

# ==============================
# Traducción con Gemini
# ==============================
st.subheader("🌐 Traducción de descripción (Gemini)")
col_a, col_b = st.columns([2,1])
with col_a:
    user_text = st.text_area(
        "Pega aquí la descripción de la empresa/activo a traducir",
        placeholder="Apple Inc. designs, manufactures, and markets smartphones...",
        height=150,
    )
with col_b:
    target_lang = st.selectbox("Traducir a", ["es", "en", "pt", "fr", "de"], index=0)
    translate_now = st.button("Traducir con Gemini")

if translate_now:
    try:
        translated = appy.translate_text(user_text, target_lang=target_lang)
        if translated:
            st.success("Traducción lista")
            st.text_area("Resultado", translated, height=150)
        else:
            st.info("No hay texto para traducir.")
    except Exception as e:
        st.error(f"Error al traducir: {e}")

st.divider()

# ==============================
# Descargar datos de yfinance
# ==============================
@st.cache_data(show_spinner=False)
def load_data(tkr: str, start_d: date, end_d: date, interval: str) -> pd.DataFrame:
    df = yf.download(tkr, start=start_d, end=end_d + timedelta(days=1), interval=interval, progress=False)
    if df.empty:
        return pd.DataFrame()
    df = df.rename(columns=str.title)  # Open, High, Low, Close, Volume
    df.index = pd.to_datetime(df.index)
    return df

if btn_load:
    df = load_data(ticker, start, end, interval)
    if df.empty:
        st.warning("No se obtuvieron datos. Verifica el ticker/fechas.")
    else:
        st.subheader(f"📊 OHLCV — {ticker}")
        st.caption(f"Desde {df.index.min().date()} hasta {df.index.max().date()} | {interval}")

        # Gráfico Seaborn O/H/L/C
        fig1, ax1 = plt.subplots(figsize=(11, 5))
        sns.lineplot(data=df[["Open", "High", "Low", "Close"]], ax=ax1)
        ax1.set_title(f"{ticker} — Open/High/Low/Close")
        ax1.set_xlabel("Fecha")
        ax1.set_ylabel("Precio")
        st.pyplot(fig1)

        # Gráfico Volume
        fig2, ax2 = plt.subplots(figsize=(11, 3))
        ax2.bar(df.index, df["Volume"].values)
        ax2.set_title(f"{ticker} — Volume")
        ax2.set_xlabel("Fecha")
        ax2.set_ylabel("Volumen")
        st.pyplot(fig2)

        # Candlestick con Plotly
        with st.expander("Ver Candlestick (Plotly)"):
            fig_candle = go.Figure(data=[go.Candlestick(
                x=df.index,
                open=df['Open'], high=df['High'], low=df['Low'], close=df['Close']
            )])
            fig_candle.update_layout(height=500, xaxis_rangeslider_visible=False)
            st.plotly_chart(fig_candle, use_container_width=True)

        # Resumen estadístico
        st.subheader("📐 Resumen estadístico")
        st.dataframe(df.describe().T, use_container_width=True)

        # Descargar CSV
        csv = df.to_csv().encode("utf-8")
        st.download_button("⬇️ Descargar CSV", data=csv, file_name=f"{ticker}_{interval}.csv", mime="text/csv")

st.divider()
st.markdown(DISCLAIMER)
