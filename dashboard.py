import streamlit as st
import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, f1_score, confusion_matrix

st.set_page_config(page_title="Dashboard Preditivo - Ambev", layout="wide")

st.markdown("""
    <style>
    .stApp { background-color: #FFF0F5; }
    h1, h2, h3 { color: #C154C1; }
    .stSidebar { background-color: #FFB6C1; }
    div[data-testid="stMetricValue"] { color: #C154C1; }
    </style>
""", unsafe_allow_html=True)

st.title("Análise Preditiva e Retorno Financeiro - Ambev (ABEV3.SA)")
st.markdown("---")


@st.cache_data
def carregar_dados():
    dados = yf.download("ABEV3.SA", start="2023-01-01", end="2025-12-31")
    if isinstance(dados.columns, pd.MultiIndex):
        dados.columns = dados.columns.droplevel(1)
    
    dados['Diferenca'] = dados['Close'] - dados['Open']
    dados['Alvo'] = (dados['Diferenca'] > 0).astype(int)
    return dados

dados = carregar_dados()

st.sidebar.header("Painel de Controle")
st.sidebar.write("Defina os parâmetros do modelo:")

# Filtro para selecionar os anos de treinamento
anos_disponiveis = [2023, 2024]
anos_treino = st.sidebar.multiselect(
    "Selecione os anos de Treinamento:",
    options=anos_disponiveis,
    default=anos_disponiveis
)

if not anos_treino:
    st.warning("Selecione pelo menos um ano para o treinamento na barra lateral.")
    st.stop() # Pausa o app até o usuário selecionar um ano

# --- 4. PREPARAÇÃO DOS DADOS E TREINAMENTO ---
# Filtrando os dados com base na escolha do usuário
dados_treino = dados[dados.index.year.isin(anos_treino)].copy()
dados_teste = dados[dados.index.year == 2025].copy() # Teste fixo em 2025 conforme projeto

features = ['Open', 'High', 'Low', 'Volume']
X_treino = dados_treino[features]
y_treino = dados_treino['Alvo']
X_teste = dados_teste[features]
y_teste = dados_teste['Alvo']

# Normalização
scaler = StandardScaler()
X_treino_norm = scaler.fit_transform(X_treino)
X_teste_norm = scaler.transform(X_teste)

# Treinamento do Modelo
modelo_knn = KNeighborsClassifier(n_neighbors=5)
modelo_knn.fit(X_treino_norm, y_treino)
previsoes = modelo_knn.predict(X_teste_norm)

# --- 5. CÁLCULO DAS MÉTRICAS E RESULTADOS FINANCEIROS ---
# Métricas do Robô
acuracia = accuracy_score(y_teste, previsoes) * 100
precisao = precision_score(y_teste, previsoes, zero_division=0) * 100
f1 = f1_score(y_teste, previsoes, zero_division=0) * 100

tn, fp, fn, tp = confusion_matrix(y_teste, previsoes).ravel()
especificidade = (tn / (tn + fp)) * 100 if (tn + fp) > 0 else 0

# Simulação Financeira
dados_teste['Rendimento_Diario_%'] = ((dados_teste['Close'] - dados_teste['Open']) / dados_teste['Open']) * 100
dados_teste['Previsao_Robo'] = previsoes
dias_operados = dados_teste[dados_teste['Previsao_Robo'] == 1]

retorno_ganhos = dias_operados[dias_operados['Rendimento_Diario_%'] > 0]['Rendimento_Diario_%'].sum()
retorno_perdas = dias_operados[dias_operados['Rendimento_Diario_%'] < 0]['Rendimento_Diario_%'].sum()
retorno_geral = retorno_ganhos + retorno_perdas

# --- 6. EXIBIÇÃO NA TELA (Item 8) ---
st.subheader("Resultados do Modelo (Ano de Teste: 2025)")

# Criando colunas para organizar os números de forma elegante
col1, col2, col3, col4 = st.columns(4)
col1.metric("Acurácia", f"{acuracia:.1f}%")
col2.metric("Precisão", f"{precisao:.1f}%")
col3.metric("F1-Score", f"{f1:.1f}%")
col4.metric("Especificidade", f"{especificidade:.1f}%")

st.markdown("---")

st.subheader("Simulação Financeira Day Trade")
col_fin1, col_fin2, col_fin3 = st.columns(3)
col_fin1.metric("Retorno de Ganhos", f"{retorno_ganhos:.2f}%")
col_fin2.metric("Retorno de Perdas", f"{retorno_perdas:.2f}%")
col_fin3.metric("Saldo Geral", f"{retorno_geral:.2f}%")

st.markdown("---")

# Gráficos exigidos no Item 7
st.subheader("Visualização dos Dados")
col_graf1, col_graf2 = st.columns(2)

with col_graf1:
    st.write("**Série Temporal de Fechamento (2023 - 2025)**")
    fig1, ax1 = plt.subplots(figsize=(6, 4))
    ax1.plot(dados.index, dados['Close'], color='#C154C1', linewidth=1)
    ax1.set_ylabel('Preço (R$)')
    ax1.grid(True)
    st.pyplot(fig1)

with col_graf2:
    st.write("**Distribuição Percentual do Alvo (Base Total)**")
    percentual = dados['Alvo'].value_counts(normalize=True) * 100
    fig2, ax2 = plt.subplots(figsize=(6, 4))
    barras = ax2.bar(['Baixa (0)', 'Alta (1)'], percentual, color=['#FF69B4', '#DA70D6'])
    ax2.set_ylabel('Percentual (%)')
    ax2.set_ylim(0, 100)
    for barra in barras:
        altura = barra.get_height()
        ax2.text(barra.get_x() + barra.get_width()/2., altura + 1, f'{altura:.1f}%', ha='center', va='bottom')
    st.pyplot(fig2)