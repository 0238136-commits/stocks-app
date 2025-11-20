import os
from datetime import date, datetime, timedelta

import numpy as np
import pandas as pd
import yfinance as yf
import plotly.graph_objects as go
import streamlit as st
from dotenv import load_dotenv
import google.generativeai as genai

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
st.set_page_config(
    page_title="Stocks Ticket App — OHLCV + Riesgo + Gemini",
    page_icon="📈",
    layout="wide",
)

# Config Gemini
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
if GOOGLE_API_KEY:
    genai.configure(api_key=GOOGLE_API_KEY)


# ==============================
# Funciones auxiliares
# ==============================
@st.cache_data(show_spinner=False)
def descargar_historial(ticker, start, end, interval="1d"):
    return yf.download(ticker, start=start, end=end, interval=interval, progress=False)


@st.cache_data(show_spinner=False)
def descargar_multi(tickers, start, end, interval="1d"):
    return yf.download(
        tickers,
        start=start,
        end=end,
        interval=interval,
        auto_adjust=False,
        progress=False,
    )


def calcular_kpis(returns, benchmark_returns=None):
    """KPIs básicos de riesgo/rendimiento a partir de rendimientos diarios."""
    returns = returns.dropna()
    if returns.empty:
        return None, None, None, None

    ann_return = (1 + returns.mean()) ** 252 - 1
    ann_vol = returns.std() * np.sqrt(252)

    # Max drawdown con curva de equity
    equity = (1 + returns).cumprod()
    running_max = equity.cummax()
    drawdown = equity / running_max - 1
    max_dd = drawdown.min()  # < 0

    beta = None
    if benchmark_returns is not None:
        benchmark_returns = benchmark_returns.dropna()
        common = returns.index.intersection(benchmark_returns.index)
        if len(common) > 10:
            r = returns.loc[common]
            m = benchmark_returns.loc[common]
            cov = np.cov(r, m)[0, 1]
            var_m = np.var(m)
            if var_m != 0:
                beta = cov / var_m

    return ann_return, ann_vol, max_dd, beta


def rendimiento_periodo(series, start_date=None):
    """Rendimiento simple entre primera y última observación en el rango."""
    series = series.dropna()
    if series.empty:
        return np.nan

    if start_date is not None:
        series = series[series.index >= start_date]

    if len(series) < 2:
        return np.nan

    return series.iloc[-1] / series.iloc[0] - 1


def tabla_rendimientos(prices, hoy):
    """Crea tabla de rendimientos para varios periodos estándar."""
    resultados = []

    # Periodos
    periodos = [
        ("3M", hoy - timedelta(days=90)),
        ("6M", hoy - timedelta(days=180)),
        ("9M", hoy - timedelta(days=270)),
        ("YTD", date(hoy.year, 1, 1)),
        ("1Y", hoy - timedelta(days=365)),
        ("3Y", hoy - timedelta(days=365 * 3)),
        ("5Y", hoy - timedelta(days=365 * 5)),
    ]

    for col in prices.columns:
        serie = prices[col]
        fila = {"Activo": col}
        for nombre, inicio in periodos:
            fila[nombre] = rendimiento_periodo(serie, inicio)
        resultados.append(fila)

    df = pd.DataFrame(resultados)
    df.set_index("Activo", inplace=True)
    return df


def traducir_con_gemini(texto, target_language="es"):
    if not GOOGLE_API_KEY:
        return "⚠️ No hay GOOGLE_API_KEY configurada en los Secrets."

    try:
        model = genai.GenerativeModel("gemini-1.5-flash")
        prompt = f"Traduce el siguiente texto al idioma '{target_language}' manteniendo el sentido financiero y profesional:\n\n{texto}"
        respuesta = model.generate_content(prompt)
        return respuesta.text.strip()
    except Exception as e:
        return f"No se pudo traducir la descripción en este momento. Detalle técnico: {e}"


def panel_fundamentales(ticker):
    st.subheader("📚 Datos fundamentales")
    try:
        tk = yf.Ticker(ticker)
        info = tk.info if hasattr(tk, "info") else tk.get_info()

        def g(key, default=None):
            return info.get(key, default)

        col1, col2, col3 = st.columns(3)
        col1.metric("P/E (ttm)", f"{g('trailingPE', np.nan):.2f}" if g("trailingPE") else "N/A")
        col2.metric("P/B", f"{g('priceToBook', np.nan):.2f}" if g("priceToBook") else "N/A")
        col3.metric(
            "Margen neto",
            f"{g('profitMargins', 0):.2%}" if g("profitMargins") is not None else "N/A",
        )

        col4, col5, col6 = st.columns(3)
        col4.metric(
            "ROE",
            f"{g('returnOnEquity', 0):.2%}" if g("returnOnEquity") is not None else "N/A",
        )
        col5.metric(
            "Deuda / Equity",
            f"{g('debtToEquity', np.nan):.2f}" if g("debtToEquity") else "N/A",
        )
        col6.metric(
            "Cap. de mercado",
            f"{g('marketCap', 0):,.0f}" if g("marketCap") else "N/A",
        )
    except Exception as e:
        st.info(f"No fue posible obtener datos fundamentales para {ticker}. ({e})")


# ==============================
# Layout principal
# ==============================
st.title("📈 Stocks Ticket App — OHLCV, Riesgo y Gemini")
st.caption(
    "Analiza acciones con gráficos OHLCV, comparación con índice, métricas de riesgo, fundamentales y traducción con Gemini."
)

# ------------
# Sidebar
# ------------
hoy = date.today()
default_end = hoy
default_start = hoy - timedelta(days=365 * 5)

st.sidebar.header("Parámetros")

tickers = st.sidebar.multiselect(
    "Tickers a analizar (máx 3)",
    ["AAPL", "MSFT", "TSLA", "AMZN", "NVDA", "GOOGL"],
    default=["AAPL"],
    max_selections=3,
)

ticker_principal = tickers[0] if tickers else "AAPL"

benchmark = st.sidebar.text_input("Índice de referencia (benchmark)", "SPY")

col_fecha_1, col_fecha_2 = st.sidebar.columns(2)
start_date = col_fecha_1.date_input("Inicio", default_start)
end_date = col_fecha_2.date_input("Fin", default_end)

interval = st.sidebar.selectbox(
    "Intervalo",
    ["1d", "1wk", "1mo"],
    index=0,
)

st.sidebar.write(" ")
cargar = st.sidebar.button("📥 Cargar datos")

st.sidebar.markdown("---")
st.sidebar.markdown(DISCLAIMER)

# ==============================
# Contenido principal
# ==============================
if cargar:
    if not tickers:
        st.warning("Selecciona al menos un ticker.")
    else:
        with st.spinner("Descargando datos históricos..."):
            # OHLCV para ticker principal
            hist_principal = descargar_historial(
                ticker_principal, start_date, end_date, interval
            )

            # Multi para comparación
            activos = list(set(tickers + [benchmark]))
            multi = descargar_multi(activos, start_date, end_date, interval)

        if hist_principal.empty or multi.empty:
            st.error("No se pudieron descargar datos. Verifica los símbolos y las fechas.")
        else:
            # ==========================
            # Columna izquierda: Gráficos
            # ==========================
            col_left, col_right = st.columns([2, 1], gap="large")

            with col_left:
                st.subheader(f"🕯️ Velas OHLCV — {ticker_principal}")

                fig_candle = go.Figure(
                    data=[
                        go.Candlestick(
                            x=hist_principal.index,
                            open=hist_principal["Open"],
                            high=hist_principal["High"],
                            low=hist_principal["Low"],
                            close=hist_principal["Close"],
                            name=ticker_principal,
                        )
                    ]
                )
                fig_candle.update_layout(
                    xaxis_title="Fecha",
                    yaxis_title="Precio",
                    height=500,
                )
                st.plotly_chart(fig_candle, use_container_width=True)

                # Base cero comparación
                st.subheader("📊 Comparación base cero vs benchmark")

                prices = multi["Adj Close"] if isinstance(multi.columns, pd.MultiIndex) else multi["Adj Close"]

                # Asegurar DataFrame
                if isinstance(prices, pd.Series):
                    prices = prices.to_frame()

                prices = prices.dropna(how="all")
                base = prices / prices.iloc[0]

                fig_base = go.Figure()
                for col in base.columns:
                    fig_base.add_trace(
                        go.Scatter(
                            x=base.index,
                            y=base[col],
                            mode="lines",
                            name=col,
                        )
                    )
                fig_base.update_layout(
                    xaxis_title="Fecha",
                    yaxis_title="Índice base 1.0",
                    height=450,
                )
                st.plotly_chart(fig_base, use_container_width=True)

                # Correlación
                st.subheader("🔗 Correlación entre activos")
                returns_all = prices.pct_change().dropna()
                corr = returns_all.corr()
                st.dataframe(corr.style.format("{:.2f}"))

                # Descargar CSV
                st.subheader("📥 Descargar precios históricos")
                csv_data = prices.to_csv().encode("utf-8")
                st.download_button(
                    label="Descargar CSV",
                    data=csv_data,
                    file_name=f"precios_{'_'.join(activos)}.csv",
                    mime="text/csv",
                )

            # ==========================
            # Columna derecha: Riesgo + Fundamentales
            # ==========================
            with col_right:
                # KPIs
                st.subheader("📌 KPIs de riesgo")

                # Rendimientos diarios del principal y benchmark
                prices_principal = prices[ticker_principal]
                prices_bench = prices[benchmark]

                r_ticker = prices_principal.pct_change().dropna()
                r_bench = prices_bench.pct_change().dropna()

                ann_ret, ann_vol, max_dd, beta = calcular_kpis(
                    r_ticker, r_bench
                )

                c1, c2 = st.columns(2)
                c1.metric(
                    "Rendimiento anualizado",
                    f"{ann_ret:.2%}" if ann_ret is not None else "N/A",
                )
                c2.metric(
                    "Volatilidad anualizada",
                    f"{ann_vol:.2%}" if ann_vol is not None else "N/A",
                )

                c3, c4 = st.columns(2)
                c3.metric(
                    "Max drawdown",
                    f"{max_dd:.2%}" if max_dd is not None else "N/A",
                )
                c4.metric(
                    "Beta vs benchmark",
                    f"{beta:.2f}" if beta is not None else "N/A",
                )

                st.markdown("---")
                panel_fundamentales(ticker_principal)

            # ==========================
            # Tabla de rendimientos
            # ==========================
            st.subheader("📅 Rendimientos por periodo")
            tabla = tabla_rendimientos(prices, hoy)
            st.dataframe(
                tabla.style.format("{:.2%}"),
                use_container_width=True,
            )

# ==============================
# Sección: Traducción con Gemini
# ==============================
st.markdown("---")
st.header("🌐 Traducción de descripción (Gemini)")

col_t1, col_t2 = st.columns([3, 1])

with col_t1:
    texto_original = st.text_area(
        "Pega aquí la descripción de la empresa/activo a traducir",
        value="Apple Inc. designs, manufactures, and markets smartphones, personal computers, tablets and wearables.",
        height=150,
    )

with col_t2:
    idioma_destino = st.selectbox(
        "Traducir a",
        options=["es", "en", "pt", "fr"],
        index=0,
    )
    traducir = st.button("Traducir con Gemini")

if traducir:
    with st.spinner("Consultando a Gemini..."):
        traduccion = traducir_con_gemini(texto_original, idioma_destino)

    st.subheader("📖 Descripción (traducida con Gemini)")
    st.write(traduccion)

# Footer
st.markdown("---")
st.caption(DISCLAIMER)
