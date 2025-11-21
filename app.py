import os
from datetime import date, datetime, timedelta
import math
import warnings
import requests
from collections import Counter
import re

import numpy as np
import pandas as pd
import yfinance as yf
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st
from dotenv import load_dotenv
from bs4 import BeautifulSoup
from scipy import stats
import ta

warnings.filterwarnings('ignore')

# =============================
# Branding / Disclaimer
# =============================
AUTHOR_NAME = "Esteban González Guerra"
PERSONAL_BRAND = "TuMarca"
DISCLAIMER = (
    f"© {date.today().year} {AUTHOR_NAME} — Todos los derechos reservados. "
    f"Marca personal: {PERSONAL_BRAND}.\n\n"
    "La información presentada es con fines educativos y no constituye asesoría financiera. "
    "Las proyecciones son estimaciones basadas en datos históricos y no garantizan resultados futuros."
)

# =============================
# Configuración inicial
# =============================
load_dotenv(override=True)

st.set_page_config(
    page_title="🚀 Elite Stock Analytics Platform",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS personalizado
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        padding: 1rem 0;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    .stMetric {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #667eea;
    }
</style>
""", unsafe_allow_html=True)

st.markdown('<h1 class="main-header">🚀 ELITE STOCK ANALYTICS PLATFORM</h1>', unsafe_allow_html=True)
st.caption(DISCLAIMER)

# =============================
# Sidebar — Parámetros
# =============================
st.sidebar.image("https://img.icons8.com/color/96/000000/stock-share.png", width=100)
st.sidebar.title("⚙️ Panel de Control")

default_ticker = "AAPL"
ticker = st.sidebar.text_input(
    "🎯 Ticker de la acción:",
    value=default_ticker,
    help="Ejemplo: AAPL, TSLA, MSFT, GOOGL"
).upper()

benchmark_default = "SPY"
benchmark_ticker = st.sidebar.text_input(
    "📊 Índice de referencia:",
    value=benchmark_default,
    help="Benchmark para comparación (SPY, ^GSPC, QQQ)"
).upper()

years_back = st.sidebar.slider(
    "📅 Años de datos históricos:",
    min_value=1,
    max_value=10,
    value=5
)

risk_free_rate = st.sidebar.number_input(
    "💵 Tasa libre de riesgo (%):",
    min_value=0.0,
    max_value=10.0,
    value=4.5,
    step=0.1,
    help="Para cálculo de Sharpe Ratio"
) / 100

st.sidebar.markdown("---")
st.sidebar.markdown("### 🎨 Opciones de visualización")

show_technical = st.sidebar.checkbox("📈 Análisis Técnico Avanzado", value=True)
show_montecarlo = st.sidebar.checkbox("🎲 Simulación Monte Carlo", value=True)
show_news_sentiment = st.sidebar.checkbox("📰 Análisis de Sentimiento", value=True)

periods_to_show = ["1M", "3M", "6M", "YTD", "1Y", "3Y", "5Y"]

# =============================
# Funciones auxiliares mejoradas
# =============================
@st.cache_data(show_spinner=False, ttl=3600)
def descargar_precios(ticker: str, years: int = 5) -> pd.DataFrame:
    """Descarga datos históricos optimizados"""
    try:
        end_date = datetime.now()
        start_date = end_date - timedelta(days=years*365)
        data = yf.download(ticker, start=start_date, end=end_date, progress=False)
        if data.empty:
            return pd.DataFrame()
        data = data.dropna()
        data.index = pd.to_datetime(data.index)
        return data
    except Exception as e:
        st.error(f"Error descargando {ticker}: {str(e)}")
        return pd.DataFrame()


def calcular_metricas_avanzadas(precios: pd.Series, benchmark: pd.Series = None, rf_rate: float = 0.045) -> dict:
    """Calcula métricas financieras avanzadas"""
    if precios is None or precios.empty:
        return {}

    precios = precios.sort_index()
    returns = precios.pct_change().dropna()
    
    # Rendimientos básicos
    total_return = (precios.iloc[-1] / precios.iloc[0]) - 1
    days = (precios.index[-1] - precios.index[0]).days
    years = days / 365.0
    cagr = (precios.iloc[-1] / precios.iloc[0]) ** (1 / years) - 1 if years > 0 else 0
    
    # Riesgo
    vol_anual = returns.std() * np.sqrt(252)
    
    # Sharpe Ratio
    excess_returns = returns - (rf_rate / 252)
    sharpe = np.sqrt(252) * excess_returns.mean() / returns.std() if returns.std() != 0 else 0
    
    # Sortino Ratio (solo considera downside volatility)
    downside_returns = returns[returns < 0]
    downside_std = downside_returns.std() * np.sqrt(252)
    sortino = (cagr - rf_rate) / downside_std if downside_std != 0 else 0
    
    # Maximum Drawdown
    cumulative = (1 + returns).cumprod()
    running_max = cumulative.expanding().max()
    drawdown = (cumulative - running_max) / running_max
    max_drawdown = drawdown.min()
    
    # Calmar Ratio
    calmar = cagr / abs(max_drawdown) if max_drawdown != 0 else 0
    
    # VaR y CVaR (95% confidence)
    var_95 = returns.quantile(0.05)
    cvar_95 = returns[returns <= var_95].mean()
    
    # Beta y Alpha (si hay benchmark)
    beta, alpha = np.nan, np.nan
    if benchmark is not None and not benchmark.empty:
        try:
            bench_returns = benchmark.pct_change().dropna()
            common_dates = returns.index.intersection(bench_returns.index)
            if len(common_dates) > 30:
                ret_aligned = returns.loc[common_dates]
                bench_aligned = bench_returns.loc[common_dates]
                covariance = ret_aligned.cov(bench_aligned)
                bench_variance = bench_aligned.var()
                beta = covariance / bench_variance if bench_variance != 0 else 1
                alpha = (ret_aligned.mean() * 252) - (rf_rate + beta * (bench_aligned.mean() * 252 - rf_rate))
        except:
            pass
    
    # Win Rate
    win_rate = (returns > 0).sum() / len(returns) if len(returns) > 0 else 0
    
    # Average Gain/Loss
    avg_gain = returns[returns > 0].mean() if len(returns[returns > 0]) > 0 else 0
    avg_loss = returns[returns < 0].mean() if len(returns[returns < 0]) > 0 else 0
    
    return {
        "total_return": total_return,
        "cagr": cagr,
        "vol_anual": vol_anual,
        "sharpe": sharpe,
        "sortino": sortino,
        "max_drawdown": max_drawdown,
        "calmar": calmar,
        "var_95": var_95,
        "cvar_95": cvar_95,
        "beta": beta,
        "alpha": alpha,
        "win_rate": win_rate,
        "avg_gain": avg_gain,
        "avg_loss": avg_loss,
    }


def calcular_indicadores_tecnicos(data: pd.DataFrame) -> pd.DataFrame:
    """Calcula indicadores técnicos avanzados"""
    df = data.copy()
    
    # RSI
    df['RSI'] = ta.momentum.RSIIndicator(df['Close'], window=14).rsi()
    
    # MACD
    macd = ta.trend.MACD(df['Close'])
    df['MACD'] = macd.macd()
    df['MACD_Signal'] = macd.macd_signal()
    df['MACD_Hist'] = macd.macd_diff()
    
    # Bollinger Bands
    bollinger = ta.volatility.BollingerBands(df['Close'])
    df['BB_High'] = bollinger.bollinger_hband()
    df['BB_Mid'] = bollinger.bollinger_mavg()
    df['BB_Low'] = bollinger.bollinger_lband()
    
    # Moving Averages
    df['SMA_20'] = ta.trend.SMAIndicator(df['Close'], window=20).sma_indicator()
    df['SMA_50'] = ta.trend.SMAIndicator(df['Close'], window=50).sma_indicator()
    df['SMA_200'] = ta.trend.SMAIndicator(df['Close'], window=200).sma_indicator()
    df['EMA_12'] = ta.trend.EMAIndicator(df['Close'], window=12).ema_indicator()
    df['EMA_26'] = ta.trend.EMAIndicator(df['Close'], window=26).ema_indicator()
    
    # ATR (Average True Range)
    df['ATR'] = ta.volatility.AverageTrueRange(df['High'], df['Low'], df['Close']).average_true_range()
    
    # Volume indicators
    df['Volume_SMA'] = df['Volume'].rolling(window=20).mean()
    
    # Stochastic Oscillator
    stoch = ta.momentum.StochasticOscillator(df['High'], df['Low'], df['Close'])
    df['Stoch_K'] = stoch.stoch()
    df['Stoch_D'] = stoch.stoch_signal()
    
    return df


@st.cache_data(show_spinner=False, ttl=1800)
def scrape_yahoo_news(ticker: str) -> list:
    """Scraping de noticias de Yahoo Finance"""
    try:
        url = f"https://finance.yahoo.com/quote/{ticker}/news"
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        }
        response = requests.get(url, headers=headers, timeout=10)
        soup = BeautifulSoup(response.content, 'html.parser')
        
        news_items = []
        for item in soup.find_all('h3', limit=10):
            title = item.get_text(strip=True)
            link_tag = item.find_parent('a')
            link = link_tag['href'] if link_tag and 'href' in link_tag.attrs else ''
            if title and len(title) > 10:
                news_items.append({'title': title, 'link': link, 'source': 'Yahoo Finance'})
        
        return news_items
    except Exception as e:
        return []


@st.cache_data(show_spinner=False, ttl=1800)
def scrape_finviz_data(ticker: str) -> dict:
    """Scraping de datos fundamentales de Finviz"""
    try:
        url = f"https://finviz.com/quote.ashx?t={ticker}"
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        }
        response = requests.get(url, headers=headers, timeout=10)
        soup = BeautifulSoup(response.content, 'html.parser')
        
        fundamentals = {}
        table = soup.find('table', {'class': 'snapshot-table2'})
        
        if table:
            rows = table.find_all('tr')
            for row in rows:
                cols = row.find_all('td')
                for i in range(0, len(cols), 2):
                    if i + 1 < len(cols):
                        key = cols[i].get_text(strip=True)
                        value = cols[i + 1].get_text(strip=True)
                        fundamentals[key] = value
        
        # News from Finviz
        news_table = soup.find('table', {'id': 'news-table'})
        news_items = []
        if news_table:
            for row in news_table.find_all('tr', limit=10):
                title_cell = row.find('a')
                if title_cell:
                    news_items.append({
                        'title': title_cell.get_text(strip=True),
                        'link': title_cell['href'],
                        'source': 'Finviz'
                    })
        
        return {'fundamentals': fundamentals, 'news': news_items}
    except Exception as e:
        return {'fundamentals': {}, 'news': []}


def analisis_sentimiento_simple(news_list: list) -> dict:
    """Análisis básico de sentimiento en titulares"""
    positive_words = ['surge', 'gain', 'profit', 'growth', 'up', 'high', 'buy', 'bull', 
                     'strong', 'beat', 'success', 'rise', 'rally', 'positive', 'upgrade']
    negative_words = ['fall', 'drop', 'loss', 'down', 'low', 'sell', 'bear', 'weak', 
                     'miss', 'fail', 'decline', 'crash', 'negative', 'downgrade']
    
    sentiment_scores = []
    for news in news_list:
        title_lower = news['title'].lower()
        pos_count = sum(1 for word in positive_words if word in title_lower)
        neg_count = sum(1 for word in negative_words if word in title_lower)
        
        if pos_count > neg_count:
            sentiment_scores.append('Positivo')
        elif neg_count > pos_count:
            sentiment_scores.append('Negativo')
        else:
            sentiment_scores.append('Neutral')
    
    if not sentiment_scores:
        return {'general': 'Neutral', 'distribution': {}}
    
    counter = Counter(sentiment_scores)
    total = len(sentiment_scores)
    
    return {
        'general': max(counter, key=counter.get),
        'distribution': {k: v/total for k, v in counter.items()}
    }


def simulacion_monte_carlo(precios: pd.Series, dias_futuros: int = 252, simulaciones: int = 1000) -> pd.DataFrame:
    """Simulación Monte Carlo para proyecciones"""
    if precios.empty:
        return pd.DataFrame()
    
    returns = precios.pct_change().dropna()
    media = returns.mean()
    std = returns.std()
    
    ultimo_precio = precios.iloc[-1]
    
    simulaciones_array = np.zeros((dias_futuros, simulaciones))
    
    for i in range(simulaciones):
        precios_simulados = [ultimo_precio]
        for j in range(dias_futuros):
            rendimiento = np.random.normal(media, std)
            precio = precios_simulados[-1] * (1 + rendimiento)
            precios_simulados.append(precio)
        simulaciones_array[:, i] = precios_simulados[1:]
    
    fechas_futuras = pd.date_range(start=precios.index[-1], periods=dias_futuros+1, freq='D')[1:]
    
    df_sim = pd.DataFrame(simulaciones_array, index=fechas_futuras)
    
    return df_sim


def filtrar_por_periodo(data: pd.DataFrame, periodo: str) -> pd.DataFrame:
    """Filtra datos por periodo"""
    if data.empty:
        return data

    end_date = data.index[-1]
    if periodo == "YTD":
        start_date = datetime(end_date.year, 1, 1)
    elif periodo == "1M":
        start_date = end_date - pd.DateOffset(months=1)
    elif periodo == "3M":
        start_date = end_date - pd.DateOffset(months=3)
    elif periodo == "6M":
        start_date = end_date - pd.DateOffset(months=6)
    elif periodo == "1Y":
        start_date = end_date - pd.DateOffset(years=1)
    elif periodo == "3Y":
        start_date = end_date - pd.DateOffset(years=3)
    elif periodo == "5Y":
        start_date = end_date - pd.DateOffset(years=5)
    else:
        return data

    return data[data.index >= start_date]


# =============================
# Descarga de datos
# =============================
if not ticker:
    st.warning("⚠️ Ingresa un ticker válido en la barra lateral para comenzar.")
    st.stop()

with st.spinner(f"🔄 Descargando datos de {ticker}..."):
    data_accion = descargar_precios(ticker, years_back)

with st.spinner(f"🔄 Descargando datos del benchmark {benchmark_ticker}..."):
    data_bench = descargar_precios(benchmark_ticker, years_back)

if data_accion.empty:
    st.error("❌ No se pudieron descargar datos para el ticker ingresado. Verifica el símbolo.")
    st.stop()

# Aplicar indicadores técnicos
if show_technical:
    data_accion = calcular_indicadores_tecnicos(data_accion)

# =============================
# Info de la empresa
# =============================
ticker_yf = yf.Ticker(ticker)
info = ticker_yf.info if hasattr(ticker_yf, "info") else {}

nombre_largo = info.get("longName", ticker)
sector = info.get("sector", "N/A")
industria = info.get("industry", "N/A")
pais = info.get("country", "N/A")
market_cap = info.get("marketCap", None)
pe_ratio = info.get("trailingPE", None)
forward_pe = info.get("forwardPE", None)
peg_ratio = info.get("pegRatio", None)
price_to_book = info.get("priceToBook", None)
dividend_yield = info.get("dividendYield", None)
profit_margins = info.get("profitMargins", None)
revenue_growth = info.get("revenueGrowth", None)

# =============================
# Header con métricas clave
# =============================
st.markdown("---")
ultimo_precio = data_accion['Close'].iloc[-1]
cambio_diario = data_accion['Close'].pct_change().iloc[-1]
volumen = data_accion['Volume'].iloc[-1]

col1, col2, col3, col4, col5 = st.columns(5)

with col1:
    st.metric(
        "💰 Precio Actual",
        f"${ultimo_precio:,.2f}",
        f"{cambio_diario*100:+.2f}%"
    )

with col2:
    if market_cap:
        mc_b = market_cap / 1e9
        st.metric("🏢 Market Cap", f"${mc_b:,.1f}B")
    else:
        st.metric("🏢 Market Cap", "N/D")

with col3:
    if pe_ratio:
        st.metric("📊 P/E Ratio", f"{pe_ratio:.2f}x")
    else:
        st.metric("📊 P/E Ratio", "N/D")

with col4:
    if dividend_yield:
        st.metric("💵 Dividend Yield", f"{dividend_yield*100:.2f}%")
    else:
        st.metric("💵 Dividend Yield", "N/D")

with col5:
    st.metric("📦 Volumen", f"{volumen/1e6:.1f}M")

# =============================
# Información corporativa
# =============================
st.markdown("---")
st.subheader("🏛️ Información Corporativa")

col1, col2, col3, col4 = st.columns(4)
col1.metric("🏢 Compañía", nombre_largo[:30])
col2.metric("🏭 Sector", sector)
col3.metric("🔧 Industria", industria[:25])
col4.metric("🌍 País", pais)

# Fundamentales adicionales
st.markdown("### 📈 Métricas Fundamentales")
cols = st.columns(6)

fundamentals = [
    ("Forward P/E", forward_pe, "x", 0),
    ("PEG Ratio", peg_ratio, "x", 1),
    ("Price/Book", price_to_book, "x", 2),
    ("Profit Margin", profit_margins, "%", 3),
    ("Revenue Growth", revenue_growth, "%", 4),
]

for i, (label, value, unit, col_idx) in enumerate(fundamentals):
    if col_idx < len(cols):
        with cols[col_idx]:
            if value is not None:
                if unit == "%":
                    st.metric(label, f"{value*100:.2f}%")
                else:
                    st.metric(label, f"{value:.2f}{unit}")
            else:
                st.metric(label, "N/D")

# Descripción
descripcion = info.get("longBusinessSummary", "Descripción no disponible.")
with st.expander("📄 Ver descripción completa de la empresa"):
    st.write(descripcion)

# =============================
# Scraping de noticias
# =============================
if show_news_sentiment:
    st.markdown("---")
    st.subheader("📰 Noticias Recientes & Análisis de Sentimiento")
    
    with st.spinner("🔍 Recopilando noticias desde Yahoo Finance y Finviz..."):
        yahoo_news = scrape_yahoo_news(ticker)
        finviz_data = scrape_finviz_data(ticker)
        finviz_news = finviz_data.get('news', [])
        
        todas_noticias = yahoo_news + finviz_news
        
        if todas_noticias:
            sentimiento = analisis_sentimiento_simple(todas_noticias)
            
            col1, col2 = st.columns([2, 1])
            
            with col1:
                st.markdown(f"### Sentimiento General: **{sentimiento['general']}**")
                if sentimiento['distribution']:
                    fig_sent = go.Figure(data=[
                        go.Bar(
                            x=list(sentimiento['distribution'].keys()),
                            y=[v*100 for v in sentimiento['distribution'].values()],
                            marker_color=['#00cc96', '#ffa15a', '#ef553b']
                        )
                    ])
                    fig_sent.update_layout(
                        title="Distribución de Sentimiento (%)",
                        xaxis_title="Sentimiento",
                        yaxis_title="Porcentaje",
                        height=300,
                        template="plotly_white"
                    )
                    st.plotly_chart(fig_sent, use_container_width=True)
            
            with col2:
                st.markdown("### 📊 Resumen")
                st.metric("Total de noticias", len(todas_noticias))
                for sent, pct in sentimiento['distribution'].items():
                    st.metric(sent, f"{pct*100:.1f}%")
            
            st.markdown("### 📋 Titulares Recientes")
            for i, news in enumerate(todas_noticias[:15], 1):
                with st.expander(f"{i}. {news['title'][:100]}..."):
                    st.markdown(f"**Fuente:** {news['source']}")
                    if news['link']:
                        st.markdown(f"[🔗 Leer más]({news['link']})")
        else:
            st.info("No se pudieron obtener noticias en este momento.")

# =============================
# Datos fundamentales de Finviz
# =============================
if finviz_data.get('fundamentals'):
    st.markdown("---")
    st.subheader("📊 Datos Fundamentales (Finviz)")
    
    fund_data = finviz_data['fundamentals']
    
    # Organizar en tabla
    if fund_data:
        cols_fund = st.columns(4)
        items = list(fund_data.items())
        
        for idx, (key, value) in enumerate(items[:20]):  # Mostrar primeros 20
            col_idx = idx % 4
            with cols_fund[col_idx]:
                st.metric(key, value)

# =============================
# Métricas avanzadas
# =============================
st.markdown("---")
st.subheader("🎯 Métricas de Riesgo y Rendimiento (Análisis Completo)")

# Calcular métricas para diferentes periodos
metricas_por_periodo = {}
for periodo in periods_to_show:
    data_periodo = filtrar_por_periodo(data_accion, periodo)
    bench_periodo = filtrar_por_periodo(data_bench, periodo) if not data_bench.empty else None
    
    if not data_periodo.empty and "Adj Close" in data_periodo:
        bench_series = bench_periodo["Adj Close"] if bench_periodo is not None and "Adj Close" in bench_periodo else None
        metricas = calcular_metricas_avanzadas(
            data_periodo["Adj Close"],
            bench_series,
            risk_free_rate
        )
        metricas_por_periodo[periodo] = metricas

# Crear tabla comparativa
if metricas_por_periodo:
    rows = []
    for periodo, metricas in metricas_por_periodo.items():
        rows.append({
            "Periodo": periodo,
            "Rend. Total": f"{metricas.get('total_return', 0)*100:.2f}%",
            "CAGR": f"{metricas.get('cagr', 0)*100:.2f}%",
            "Volatilidad": f"{metricas.get('vol_anual', 0)*100:.2f}%",
            "Sharpe": f"{metricas.get('sharpe', 0):.2f}",
            "Sortino": f"{metricas.get('sortino', 0):.2f}",
            "Max DD": f"{metricas.get('max_drawdown', 0)*100:.2f}%",
            "Calmar": f"{metricas.get('calmar', 0):.2f}",
            "Beta": f"{metricas.get('beta', 0):.2f}",
            "Alpha": f"{metricas.get('alpha', 0)*100:.2f}%",
        })
    
    df_metricas = pd.DataFrame(rows)
    
    st.dataframe(
        df_metricas,
        use_container_width=True,
        hide_index=True
    )
    
    # Métricas destacadas del último año
    if "1Y" in metricas_por_periodo:
        met_1y = metricas_por_periodo["1Y"]
        
        st.markdown("### 🏆 Métricas Destacadas (1 año)")
        col1, col2, col3, col4, col5 = st.columns(5)
        
        col1.metric("📊 Sharpe Ratio", f"{met_1y.get('sharpe', 0):.2f}")
        col2.metric("🎯 Sortino Ratio", f"{met_1y.get('sortino', 0):.2f}")
        col3.metric("📉 Max Drawdown", f"{met_1y.get('max_drawdown', 0)*100:.2f}%")
        col4.metric("🎲 VaR (95%)", f"{met_1y.get('var_95', 0)*100:.2f}%")
        col5.metric("✅ Win Rate", f"{met_1y.get('win_rate', 0)*100:.1f}%")

# =============================
# Gráfico de velas avanzado
# =============================
st.markdown("---")
st.subheader("📊 Análisis de Precio con Velas Japonesas")

periodo_velas = st.selectbox("Selecciona periodo:", ["1M", "3M", "6M", "1Y", "3Y", "5Y", "Todo"], index=3)

if periodo_velas == "Todo":
    data_velas = data_accion.copy()
else:
    data_velas = filtrar_por_periodo(data_accion, periodo_velas)

if not data_velas.empty:
    fig = make_subplots(
        rows=3, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.03,
        row_heights=[0.6, 0.2, 0.2],
        subplot_titles=('Precio', 'Volumen', 'RSI')
    )
    
    # Velas
    fig.add_trace(
        go.Candlestick(
            x=data_velas.index,
            open=data_velas['Open'],
            high=data_velas['High'],
            low=data_velas['Low'],
            close=data_velas['Close'],
            name='Precio',
            increasing_line_color='#00cc96',
            decreasing_line_color='#ef553b'
        ),
        row=1, col=1
    )
    
    # Medias móviles
    if 'SMA_20' in data_velas.columns:
        fig.add_trace(
            go.Scatter(x=data_velas.index, y=data_velas['SMA_20'], 
                      name='SMA 20', line=dict(color='orange', width=1)),
            row=1, col=1
        )
    if 'SMA_50' in data_velas.columns:
        fig.add_trace(
            go.Scatter(x=data_velas.index, y=data_velas['SMA_50'], 
                      name='SMA 50', line=dict(color='blue', width=1)),
            row=1, col=1
        )
    if 'SMA_200' in data_velas.columns:
        fig.add_trace(
            go.Scatter(x=data_velas.index, y=data_velas['SMA_200'], 
                      name='SMA 200', line=dict(color='red', width=1.5)),
            row=1, col=1
        )
    
    # Bollinger Bands
    if 'BB_High' in data_velas.columns:
        fig.add_trace(
            go.Scatter(x=data_velas.index, y=data_velas['BB_High'], 
                      name='BB High', line=dict(color='gray', width=0.5, dash='dash')),
            row=1, col=1
        )
        fig.add_trace(
            go.Scatter(x=data_velas.index, y=data_velas['BB_Low'], 
                      name='BB Low', line=dict(color='gray', width=0.5, dash='dash'),
                      fill='tonexty', fillcolor='rgba(128,128,128,0.1)'),
            row=1, col=1
        )
    
    # Volumen
    colors = ['red' if data_velas['Close'].iloc[i] < data_velas['Open'].iloc[i] 
              else 'green' for i in range(len(data_velas))]
    fig.add_trace(
        go.Bar(x=data_velas.index, y=data_velas['Volume'], 
               name='Volumen', marker_color=colors),
        row=2, col=1
    )
    
    # RSI
    if 'RSI' in data_velas.columns:
        fig.add_trace(
            go.Scatter(x=data_velas.index, y=data_velas['RSI'], 
                      name='RSI', line=dict(color='purple', width=2)),
            row=3, col=1
        )
        fig.add_hline(y=70, line_dash="dash", line_color="red", row=3, col=1)
        fig.add_hline(y=30, line_dash="dash", line_color="green", row=3, col=1)
    
    fig.update_layout(
        height=900,
        xaxis_rangeslider_visible=False,
        template="plotly_white",
        showlegend=True,
        hovermode='x unified'
    )
    
    fig.update_yaxes(title_text="Precio ($)", row=1, col=1)
    fig.update_yaxes(title_text="Volumen", row=2, col=1)
    fig.update_yaxes(title_text="RSI", row=3, col=1)
    
    st.plotly_chart(fig, use_container_width=True)

# =============================
# Análisis técnico detallado
# =============================
if show_technical and 'RSI' in data_accion.columns:
    st.markdown("---")
    st.subheader("🔧 Análisis Técnico Detallado")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### 📊 Indicadores Actuales")
        ultimo_rsi = data_accion['RSI'].iloc[-1]
        ultimo_macd = data_accion['MACD'].iloc[-1]
        ultimo_signal = data_accion['MACD_Signal'].iloc[-1]
        
        st.metric("RSI (14)", f"{ultimo_rsi:.2f}")
        if ultimo_rsi > 70:
            st.warning("⚠️ Sobrecomprado")
        elif ultimo_rsi < 30:
            st.success("✅ Sobrevendido")
        else:
            st.info("➡️ Neutral")
        
        st.metric("MACD", f"{ultimo_macd:.2f}")
        st.metric("MACD Signal", f"{ultimo_signal:.2f}")
        
        if ultimo_macd > ultimo_signal:
            st.success("📈 Señal alcista (MACD > Signal)")
        else:
            st.warning("📉 Señal bajista (MACD < Signal)")
    
    with col2:
        st.markdown("### 🎯 Niveles de Soporte/Resistencia")
        
        # Calcular niveles aproximados
        precio_actual = data_accion['Close'].iloc[-1]
        high_52w = data_accion['High'].tail(252).max()
        low_52w = data_accion['Low'].tail(252).min()
        
        st.metric("Máximo 52 semanas", f"${high_52w:.2f}")
        st.metric("Mínimo 52 semanas", f"${low_52w:.2f}")
        st.metric("Rango", f"${high_52w - low_52w:.2f}")
        
        pct_from_high = ((precio_actual - high_52w) / high_52w) * 100
        pct_from_low = ((precio_actual - low_52w) / low_52w) * 100
        
        st.metric("% desde máximo", f"{pct_from_high:.2f}%")
        st.metric("% desde mínimo", f"{pct_from_low:.2f}%")
    
    # MACD Chart
    st.markdown("### 📈 MACD")
    fig_macd = go.Figure()
    fig_macd.add_trace(go.Scatter(
        x=data_accion.index, y=data_accion['MACD'],
        name='MACD', line=dict(color='blue', width=2)
    ))
    fig_macd.add_trace(go.Scatter(
        x=data_accion.index, y=data_accion['MACD_Signal'],
        name='Signal', line=dict(color='red', width=2)
    ))
    fig_macd.add_trace(go.Bar(
        x=data_accion.index, y=data_accion['MACD_Hist'],
        name='Histogram', marker_color='gray'
    ))
    fig_macd.update_layout(height=400, template="plotly_white")
    st.plotly_chart(fig_macd, use_container_width=True)

# =============================
# Comparación con benchmark
# =============================
st.markdown("---")
st.subheader(f"⚖️ Comparación vs {benchmark_ticker}")

if not data_bench.empty:
    # Base 100
    common_accion = data_accion[["Adj Close"]].rename(columns={"Adj Close": "Accion"})
    common_bench = data_bench[["Adj Close"]].rename(columns={"Adj Close": "Benchmark"})
    common = common_accion.join(common_bench, how="inner")
    
    if not common.empty:
        serie_accion = (common["Accion"] / common["Accion"].iloc[0]) * 100
        serie_bench = (common["Benchmark"] / common["Benchmark"].iloc[0]) * 100
        
        fig_comp = go.Figure()
        fig_comp.add_trace(go.Scatter(
            x=serie_accion.index, y=serie_accion,
            name=f"{ticker}", line=dict(color='#667eea', width=3)
        ))
        fig_comp.add_trace(go.Scatter(
            x=serie_bench.index, y=serie_bench,
            name=f"{benchmark_ticker}", line=dict(color='#764ba2', width=3)
        ))
        
        fig_comp.update_layout(
            title="Rendimiento Normalizado (Base 100)",
            xaxis_title="Fecha",
            yaxis_title="Índice",
            template="plotly_white",
            height=500,
            hovermode='x unified'
        )
        
        st.plotly_chart(fig_comp, use_container_width=True)
        
        # Métricas comparativas
        rend_accion = (serie_accion.iloc[-1] / 100 - 1) * 100
        rend_bench = (serie_bench.iloc[-1] / 100 - 1) * 100
        outperformance = rend_accion - rend_bench
        
        col1, col2, col3 = st.columns(3)
        col1.metric(f"Rendimiento {ticker}", f"{rend_accion:.2f}%")
        col2.metric(f"Rendimiento {benchmark_ticker}", f"{rend_bench:.2f}%")
        col3.metric("Outperformance", f"{outperformance:+.2f}%", 
                   delta=f"{outperformance:.2f}%" if outperformance > 0 else None)

# =============================
# Simulación Monte Carlo
# =============================
if show_montecarlo:
    st.markdown("---")
    st.subheader("🎲 Proyección Monte Carlo")
    
    col1, col2 = st.columns(2)
    with col1:
        dias_proyeccion = st.slider("Días a proyectar:", 30, 365, 252)
    with col2:
        num_simulaciones = st.slider("Número de simulaciones:", 100, 5000, 1000, step=100)
    
    with st.spinner("Ejecutando simulación Monte Carlo..."):
        df_montecarlo = simulacion_monte_carlo(
            data_accion["Adj Close"],
            dias_futuros=dias_proyeccion,
            simulaciones=num_simulaciones
        )
    
    if not df_montecarlo.empty:
        # Estadísticas
        precio_final_medio = df_montecarlo.iloc[-1].mean()
        precio_final_p5 = df_montecarlo.iloc[-1].quantile(0.05)
        precio_final_p95 = df_montecarlo.iloc[-1].quantile(0.95)
        precio_actual = data_accion["Adj Close"].iloc[-1]
        
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Precio Actual", f"${precio_actual:.2f}")
        col2.metric("Proyección Media", f"${precio_final_medio:.2f}")
        col3.metric("Escenario Pesimista (5%)", f"${precio_final_p5:.2f}")
        col4.metric("Escenario Optimista (95%)", f"${precio_final_p95:.2f}")
        
        # Gráfico
        fig_mc = go.Figure()
        
        # Algunas trayectorias de muestra
        for i in range(min(100, num_simulaciones)):
            fig_mc.add_trace(go.Scatter(
                x=df_montecarlo.index,
                y=df_montecarlo.iloc[:, i],
                mode='lines',
                line=dict(width=0.5, color='rgba(102, 126, 234, 0.1)'),
                showlegend=False,
                hoverinfo='skip'
            ))
        
        # Media
        fig_mc.add_trace(go.Scatter(
            x=df_montecarlo.index,
            y=df_montecarlo.mean(axis=1),
            mode='lines',
            name='Media',
            line=dict(color='red', width=3)
        ))
        
        # Percentiles
        fig_mc.add_trace(go.Scatter(
            x=df_montecarlo.index,
            y=df_montecarlo.quantile(0.95, axis=1),
            mode='lines',
            name='Percentil 95',
            line=dict(color='green', width=2, dash='dash')
        ))
        
        fig_mc.add_trace(go.Scatter(
            x=df_montecarlo.index,
            y=df_montecarlo.quantile(0.05, axis=1),
            mode='lines',
            name='Percentil 5',
            line=dict(color='orange', width=2, dash='dash')
        ))
        
        fig_mc.update_layout(
            title=f"Simulación Monte Carlo - {num_simulaciones} escenarios",
            xaxis_title="Fecha",
            yaxis_title="Precio ($)",
            template="plotly_white",
            height=600,
            hovermode='x unified'
        )
        
        st.plotly_chart(fig_mc, use_container_width=True)
        
        # Distribución final
        st.markdown("### 📊 Distribución de Precios Finales")
        fig_hist = go.Figure()
        fig_hist.add_trace(go.Histogram(
            x=df_montecarlo.iloc[-1],
            nbinsx=50,
            name='Distribución',
            marker_color='#667eea'
        ))
        fig_hist.update_layout(
            xaxis_title="Precio Final ($)",
            yaxis_title="Frecuencia",
            template="plotly_white",
            height=400
        )
        st.plotly_chart(fig_hist, use_container_width=True)
        
        # Probabilidades
        prob_gain = (df_montecarlo.iloc[-1] > precio_actual).sum() / num_simulaciones * 100
        st.metric(
            "🎯 Probabilidad de ganancia",
            f"{prob_gain:.1f}%",
            help="Porcentaje de simulaciones donde el precio final es mayor al actual"
        )

# =============================
# Análisis de correlaciones
# =============================
st.markdown("---")
st.subheader("🔗 Matriz de Correlación (Indicadores Técnicos)")

if show_technical:
    cols_correlacion = ['Close', 'Volume', 'RSI', 'MACD', 'ATR']
    cols_disponibles = [col for col in cols_correlacion if col in data_accion.columns]
    
    if len(cols_disponibles) > 2:
        corr_matrix = data_accion[cols_disponibles].corr()
        
        fig_corr = go.Figure(data=go.Heatmap(
            z=corr_matrix.values,
            x=corr_matrix.columns,
            y=corr_matrix.columns,
            colorscale='RdBu',
            zmid=0,
            text=corr_matrix.values.round(2),
            texttemplate='%{text}',
            textfont={"size": 10},
        ))
        
        fig_corr.update_layout(
            title="Matriz de Correlación",
            height=500,
            template="plotly_white"
        )
        
        st.plotly_chart(fig_corr, use_container_width=True)

# =============================
# Análisis de volatilidad
# =============================
st.markdown("---")
st.subheader("📉 Análisis de Volatilidad Histórica")

returns_vol = data_accion['Adj Close'].pct_change().dropna()
rolling_vol = returns_vol.rolling(window=30).std() * np.sqrt(252) * 100

fig_vol = go.Figure()
fig_vol.add_trace(go.Scatter(
    x=rolling_vol.index,
    y=rolling_vol,
    fill='tozeroy',
    name='Volatilidad 30 días',
    line=dict(color='#764ba2', width=2)
))

fig_vol.update_layout(
    title="Volatilidad Rodante (30 días, anualizada)",
    xaxis_title="Fecha",
    yaxis_title="Volatilidad (%)",
    template="plotly_white",
    height=400
)

st.plotly_chart(fig_vol, use_container_width=True)

# Estadísticas de volatilidad
vol_actual = rolling_vol.iloc[-1]
vol_promedio = rolling_vol.mean()
vol_max = rolling_vol.max()
vol_min = rolling_vol.min()

col1, col2, col3, col4 = st.columns(4)
col1.metric("Vol. Actual", f"{vol_actual:.2f}%")
col2.metric("Vol. Promedio", f"{vol_promedio:.2f}%")
col3.metric("Vol. Máxima", f"{vol_max:.2f}%")
col4.metric("Vol. Mínima", f"{vol_min:.2f}%")

# =============================
# Drawdown analysis
# =============================
st.markdown("---")
st.subheader("📊 Análisis de Drawdown")

cumulative_returns = (1 + returns_vol).cumprod()
running_max = cumulative_returns.expanding().max()
drawdown = (cumulative_returns - running_max) / running_max * 100

fig_dd = go.Figure()
fig_dd.add_trace(go.Scatter(
    x=drawdown.index,
    y=drawdown,
    fill='tozeroy',
    name='Drawdown',
    line=dict(color='red', width=2),
    fillcolor='rgba(255, 0, 0, 0.3)'
))

fig_dd.update_layout(
    title="Drawdown Histórico (%)",
    xaxis_title="Fecha",
    yaxis_title="Drawdown (%)",
    template="plotly_white",
    height=400
)

st.plotly_chart(fig_dd, use_container_width=True)

max_dd = drawdown.min()
st.metric("📉 Máximo Drawdown Histórico", f"{max_dd:.2f}%")

# =============================
# Distribución de retornos
# =============================
st.markdown("---")
st.subheader("📊 Distribución de Retornos Diarios")

col1, col2 = st.columns(2)

with col1:
    fig_dist = go.Figure()
    fig_dist.add_trace(go.Histogram(
        x=returns_vol * 100,
        nbinsx=50,
        name='Retornos',
        marker_color='#667eea'
    ))
    fig_dist.update_layout(
        title="Histograma de Retornos Diarios",
        xaxis_title="Retorno (%)",
        yaxis_title="Frecuencia",
        template="plotly_white",
        height=400
    )
    st.plotly_chart(fig_dist, use_container_width=True)

with col2:
    # Estadísticas
    mean_return = returns_vol.mean() * 100
    median_return = returns_vol.median() * 100
    std_return = returns_vol.std() * 100
    skewness = returns_vol.skew()
    kurtosis = returns_vol.kurtosis()
    
    st.markdown("### 📈 Estadísticas")
    st.metric("Media", f"{mean_return:.3f}%")
    st.metric("Mediana", f"{median_return:.3f}%")
    st.metric("Desv. Estándar", f"{std_return:.3f}%")
    st.metric("Asimetría (Skewness)", f"{skewness:.2f}")
    st.metric("Curtosis (Kurtosis)", f"{kurtosis:.2f}")

# =============================
# Recomendaciones basadas en análisis
# =============================
st.markdown("---")
st.subheader("💡 Resumen Ejecutivo & Señales")

col1, col2 = st.columns(2)

with col1:
    st.markdown("### 🎯 Señales Técnicas")
    
    señales = []
    
    if 'RSI' in data_accion.columns:
        ultimo_rsi = data_accion['RSI'].iloc[-1]
        if ultimo_rsi > 70:
            señales.append("⚠️ RSI indica sobrecompra")
        elif ultimo_rsi < 30:
            señales.append("✅ RSI indica sobreventa (oportunidad)")
        else:
            señales.append("➡️ RSI en zona neutral")
    
    if 'MACD' in data_accion.columns and 'MACD_Signal' in data_accion.columns:
        ultimo_macd = data_accion['MACD'].iloc[-1]
        ultimo_signal = data_accion['MACD_Signal'].iloc[-1]
        if ultimo_macd > ultimo_signal:
            señales.append("📈 MACD cruzó al alza (señal alcista)")
        else:
            señales.append("📉 MACD cruzó a la baja (señal bajista)")
    
    if 'SMA_50' in data_accion.columns and 'SMA_200' in data_accion.columns:
        sma_50 = data_accion['SMA_50'].iloc[-1]
        sma_200 = data_accion['SMA_200'].iloc[-1]
        if sma_50 > sma_200:
            señales.append("✅ Cruz dorada: SMA50 > SMA200 (tendencia alcista)")
        else:
            señales.append("⚠️ Cruz de muerte: SMA50 < SMA200 (tendencia bajista)")
    
    precio_actual = data_accion['Close'].iloc[-1]
    if 'BB_High' in data_accion.columns and 'BB_Low' in data_accion.columns:
        bb_high = data_accion['BB_High'].iloc[-1]
        bb_low = data_accion['BB_Low'].iloc[-1]
        if precio_actual > bb_high:
            señales.append("⚠️ Precio por encima de Banda de Bollinger superior")
        elif precio_actual < bb_low:
            señales.append("✅ Precio por debajo de Banda de Bollinger inferior")
    
    for señal in señales:
        st.write(señal)

with col2:
    st.markdown("### 📊 Resumen de Métricas")
    
    if "1Y" in metricas_por_periodo:
        met = metricas_por_periodo["1Y"]
        
        resumen = []
        
        sharpe = met.get('sharpe', 0)
        if sharpe > 1:
            resumen.append("✅ Sharpe Ratio superior a 1 (bueno)")
        elif sharpe > 0:
            resumen.append("➡️ Sharpe Ratio positivo pero bajo")
        else:
            resumen.append("⚠️ Sharpe Ratio negativo")
        
        max_dd = met.get('max_drawdown', 0)
        if max_dd > -0.10:
            resumen.append("✅ Drawdown máximo menor al 10%")
        elif max_dd > -0.20:
            resumen.append("➡️ Drawdown máximo moderado (10-20%)")
        else:
            resumen.append("⚠️ Drawdown máximo significativo (>20%)")
        
        beta = met.get('beta', 1)
        if not np.isnan(beta):
            if beta > 1:
                resumen.append(f"📈 Beta {beta:.2f}: Más volátil que el mercado")
            elif beta < 1:
                resumen.append(f"📉 Beta {beta:.2f}: Menos volátil que el mercado")
            else:
                resumen.append("➡️ Beta cercano a 1: Volatilidad similar al mercado")
        
        vol = met.get('vol_anual', 0)
        if vol < 0.20:
            resumen.append("✅ Volatilidad baja (<20%)")
        elif vol < 0.40:
            resumen.append("➡️ Volatilidad moderada (20-40%)")
        else:
            resumen.append("⚠️ Volatilidad alta (>40%)")
        
        for item in resumen:
            st.write(item)

# =============================
# Descarga de datos
# =============================
st.markdown("---")
st.subheader("💾 Exportar Datos")

col1, col2, col3 = st.columns(3)

with col1:
    csv_data = data_accion.to_csv()
    st.download_button(
        label="📥 Descargar datos históricos (CSV)",
        data=csv_data,
        file_name=f"{ticker}_historical_data.csv",
        mime="text/csv"
    )

with col2:
    if metricas_por_periodo:
        df_export = pd.DataFrame(metricas_por_periodo).T
        csv_metricas = df_export.to_csv()
        st.download_button(
            label="📥 Descargar métricas (CSV)",
            data=csv_metricas,
            file_name=f"{ticker}_metrics.csv",
            mime="text/csv"
        )

with col3:
    if not df_montecarlo.empty and show_montecarlo:
        csv_mc = df_montecarlo.to_csv()
        st.download_button(
            label="📥 Descargar simulación MC (CSV)",
            data=csv_mc,
            file_name=f"{ticker}_montecarlo.csv",
            mime="text/csv"
        )

# =============================
# Footer profesional
# =============================
st.markdown("---")
st.markdown(f"""
<div style='text-align: center; padding: 2rem; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
            border-radius: 10px; color: white;'>
    <h3>🚀 Elite Stock Analytics Platform</h3>
    <p>{DISCLAIMER}</p>
    <p><strong>Análisis generado:</strong> {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
    <p><em>Desarrollado por {AUTHOR_NAME} | {PERSONAL_BRAND}</em></p>
</div>
""", unsafe_allow_html=True)

st.balloons()