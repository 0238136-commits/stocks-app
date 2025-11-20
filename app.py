import os
from datetime import date, timedelta
import math
import numpy as np
import pandas as pd
import yfinance as yf
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from dotenv import load_dotenv
import appy  # Traducción con Gemini

# =============================
# Branding / Disclaimer
# =============================
AUTHOR_NAME = "Esteban González Guerra"
PERSONAL_BRAND = "TuMarca"
DISCLAIMER = (
    f"© {date.today().year} {AUTHOR_NAME} — Todos los derechos reservados. "
    f"Marca personal: {PERSONAL_BRAND}.\n\n"
    "La información presentada es con fines educativos y no constituye asesoría financiera."
)

# =============================
# Configuración inicial
# =============================
load_dotenv(override=True)
st.set_page_config(
    page_title="Stocks Ticket App — Análisis Completo",
    page_icon="📈",
    layout="wide"
)

st.title("📈 Stocks Ticket App — Análisis Completo de Acciones")
st.caption(DISCLAIMER)

# =============================
# Sidebar — Parámetros
# =============================
st.sidebar.header("Parámetros del análisis")

ticker = st.sidebar.text_input("Ticker principal", "AAPL").upper()
benchmark = st.sidebar.text_input("Índice de referencia (benchmark)", "SPY").upper()
years_history = st.sidebar.slider("Años de historia", min_value=1, max_value=10, value=5)
rf_rate = st.sidebar.number_input(
    "Tasa libre de riesgo anual (%)",
    min_value=0.0, max_value=15.0, value=4.0, step=0.25
)

# =============================
# Funciones auxiliares
# =============================
@st.cache_data(show_spinner=True)
def cargar_precios(ticker: str, benchmark: str, years: int):
    end = date.today()
    start = end - timedelta(days=365 * years)

    hist_ticker = yf.download(ticker, start=start, end=end, progress=False)
    hist_bench = yf.download(benchmark, start=start, end=end, progress=False)

    return hist_ticker, hist_bench


def calcular_riesgo_tabla(hist_prices, rf_rate_annual: float):
    """
    hist_prices: precios de cierre (Series o DataFrame)
    rf_rate_annual: tasa libre de riesgo en %
    """
    # Aseguramos que sea Serie 1D
    if isinstance(hist_prices, pd.DataFrame):
        if "Close" in hist_prices.columns:
            hist_prices = hist_prices["Close"]
        else:
            hist_prices = hist_prices.iloc[:, 0]

    hist_prices = hist_prices.dropna()

    periodos = {
        "3M": 63,
        "6M": 126,
        "9M": 189,
        "1Y": 252,
        "3Y": 252 * 3,
        "5Y": 252 * 5,
        "YTD": None,
    }

    filas = []
    rf = rf_rate_annual / 100.0

    for nombre, dias in periodos.items():

        # ---------- YTD ----------
        if nombre == "YTD":
            mask = hist_prices.index.year == date.today().year
            precios = hist_prices[mask]
            if len(precios) < 2:
                filas.append([nombre, "N/A", "N/A", "N/A"])
                continue

            retornos = precios.pct_change().dropna()
            if retornos.empty:
                filas.append([nombre, "N/A", "N/A", "N/A"])
                continue

            retorno_acum = float(precios.iloc[-1] / precios.iloc[0] - 1)
            vol_anual = float(retornos.std() * math.sqrt(252)) if not np.isnan(retornos.std()) else float("nan")
            mean_daily = float(retornos.mean())

        # ---------- OTROS PERIODOS ----------
        else:
            if len(hist_prices) < dias:
                filas.append([nombre, "N/A", "N/A", "N/A"])
                continue

            ventana = hist_prices.tail(dias)
            retornos = ventana.pct_change().dropna()
            if retornos.empty:
                filas.append([nombre, "N/A", "N/A", "N/A"])
                continue

            retorno_acum = float(ventana.iloc[-1] / ventana.iloc[0] - 1)
            vol_anual = float(retornos.std() * math.sqrt(252)) if not np.isnan(retornos.std()) else float("nan")
            mean_daily = float(retornos.mean())

        # Rendimiento anualizado
        rendimiento_anualizado = (1 + mean_daily) ** 252 - 1
        exceso = rendimiento_anualizado - rf

        if np.isnan(vol_anual) or vol_anual <= 0:
            sharpe = "N/A"
        else:
            sharpe = exceso / vol_anual

        filas.append([
            nombre,
            round(retorno_acum * 100, 2),
            round(vol_anual * 100, 2) if not np.isnan(vol_anual) and vol_anual > 0 else "N/A",
            round(sharpe, 2) if sharpe != "N/A" else "N/A"
        ])

    return pd.DataFrame(
        filas,
        columns=["Periodo", "Rendimiento (%)", "Volatilidad (%)", "Sharpe Ratio"]
    )

# =============================
# Cargar datos
# =============================
hist_ticker, hist_bench = cargar_precios(ticker, benchmark, years_history)

if hist_ticker.empty:
    st.error("No se encontraron datos del ticker.")
    st.stop()

data_ticker = yf.Ticker(ticker)

# =============================
# Layout con pestañas
# =============================
tab_resumen, tab_graficos, tab_riesgo = st.tabs(
    ["📌 Resumen", "📉 Gráficos", "📐 Riesgos"]
)

# =============================
# TAB 1 — RESUMEN
# =============================
with tab_resumen:
    info = data_ticker.info if hasattr(data_ticker, "info") else {}

    st.subheader(f"Resumen de {ticker}")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.write("**Nombre:**", info.get("longName", "No disponible"))
        st.write("**Sector:**", info.get("sector", "No disponible"))
        st.write("**Industria:**", info.get("industry", "No disponible"))

    with col2:
        st.write("**País:**", info.get("country", "No disponible"))
        website = info.get("website")
        if website:
            st.write("**Sitio web:**", f"[{website}]({website})")
        else:
            st.write("**Sitio web:** No disponible")

    with col3:
        mc = info.get("marketCap")
        st.write("**Market Cap:**", f"${mc:,.0f}" if mc else "No disponible")
        beta = info.get("beta")
        st.write("**Beta:**", round(beta, 2) if isinstance(beta, (int, float)) else "No disponible")

    # Métricas de precio
    close_series = hist_ticker["Close"]
    if isinstance(close_series, pd.DataFrame):
        close_series = close_series.iloc[:, 0]

    precio_actual = float(close_series.iloc[-1])
    precio_52w_max = float(close_series.tail(252).max())
    precio_52w_min = float(close_series.tail(252).min())
    if len(close_series) > 1:
        cambio_hoy = float(close_series.iloc[-1] / close_series.iloc[-2] - 1)
    else:
        cambio_hoy = 0.0

    c1, c2, c3 = st.columns(3)
    c1.metric("Precio actual", f"${precio_actual:,.2f}", f"{cambio_hoy*100:.2f}% hoy")
    c2.metric("Máx 52 semanas", f"${precio_52w_max:,.2f}")
    c3.metric("Mín 52 semanas", f"${precio_52w_min:,.2f}")

    # Descripción traducida con Gemini
    with st.expander("📘 Descripción (traducida con Gemini)"):
        try:
            descripcion = info.get("longBusinessSummary", "")
            if descripcion:
                traduccion = appy.traducir(descripcion)
                st.write(traduccion)
            else:
                st.write("No hay descripción disponible.")
        except Exception:
            st.write("No se pudo traducir la descripción en este momento.")

# =============================
# TAB 2 — GRÁFICOS
# =============================
with tab_graficos:
    st.subheader("📉 Velas Japonesas (1 año)")

    hist_1y = hist_ticker.tail(252)

    fig = go.Figure(
        data=[go.Candlestick(
            x=hist_1y.index,
            open=hist_1y["Open"],
            high=hist_1y["High"],
            low=hist_1y["Low"],
            close=hist_1y["Close"],
            name=ticker
        )]
    )

    fig.update_layout(
        height=500,
        template="plotly_white",
        xaxis_rangeslider_visible=False,
        xaxis_title="Fecha",
        yaxis_title="Precio"
    )

    st.plotly_chart(fig, use_container_width=True)

    # Comparación base cero
    st.subheader(f"📊 Comparación base cero: {ticker} vs {benchmark}")

    if hist_bench.empty or "Close" not in hist_bench.columns:
        st.warning("No se pudo descargar el benchmark.")
    else:
        s_ticker = hist_ticker["Close"]
        if isinstance(s_ticker, pd.DataFrame):
            s_ticker = s_ticker.iloc[:, 0]
        s_ticker.name = ticker

        s_bench = hist_bench["Close"]
        if isinstance(s_bench, pd.DataFrame):
            s_bench = s_bench.iloc[:, 0]
        s_bench.name = benchmark

        combined = pd.concat([s_ticker, s_bench], axis=1).dropna()

        if combined.empty:
            st.warning("No se pudieron alinear los datos.")
        else:
            base = combined / combined.iloc[0] - 1

            fig2 = px.line(
                base,
                labels={
                    "value": "Rendimiento acumulado",
                    "index": "Fecha",
                    "variable": "Activo"
                }
            )
            fig2.update_layout(template="plotly_white")
            st.plotly_chart(fig2, use_container_width=True)

# =============================
# TAB 3 — RIESGO
# =============================
with tab_riesgo:
    st.subheader("📐 Rendimientos, volatilidad y Sharpe")

    df_riesgo = calcular_riesgo_tabla(hist_ticker["Close"], rf_rate)
    st.dataframe(df_riesgo, use_container_width=True)

# =============================
# Footer
# =============================
st.markdown("---")
st.caption(DISCLAIMER)








