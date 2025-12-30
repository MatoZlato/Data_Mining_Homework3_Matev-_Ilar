import streamlit as st
import pandas as pd
import os
import numpy as np
from transformers import pipeline
import matplotlib.pyplot as plt
from wordcloud import WordCloud

# 1. Nastavitev strani in nalaganje modela
st.set_page_config(page_title="Reputation Monitor PRO", layout="wide")
st.title("📊 Brand Reputation Monitor 2023")

@st.cache_resource
def load_sentiment_model():
    # Uporaba zahtevanega modela: distilbert-base-uncased-finetuned-sst-2-english
    return pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english")

sentiment_analyzer = load_sentiment_model()

# Preverjanje obstoja podatkov
if os.path.exists('podatki_2023.csv'):
    df = pd.read_csv('podatki_2023.csv')
    
    # --- SIMULACIJA MESECEV ZA 45 VRSTIC ---
    np.random.seed(42)
    df['Mesec_Num'] = np.random.randint(0, 12, size=len(df))
    meseci_imena = ["Jan", "Feb", "Mar", "Apr", "Maj", "Jun", "Jul", "Avg", "Sep", "Okt", "Nov", "Dec"]
    df['Mesec_Ime'] = df['Mesec_Num'].apply(lambda x: meseci_imena[x])

    # --- STRANSKI MENI ---
    st.sidebar.header("Navigacija")
    izbira = st.sidebar.radio("Izberite razdelek:", ["Products", "Testimonials", "Reviews"])
    
    mapping = {"Products": "product", "Testimonials": "testimonial", "Reviews": "review"}
    df_tip = df[df['Tip'] == mapping[izbira]].copy()

    if izbira == "Reviews":
        st.header("🔍 Analiza mnenj po mesecih")
        
        # 1. Slider za izbiro meseca
        izbran_mesec = st.select_slider("Izberite mesec za leto 2023:", options=meseci_imena)
        
        # 2. Filtriranje glede na slider
        df_filtriran = df_tip[df_tip['Mesec_Ime'] == izbran_mesec].copy()
        
        if not df_filtriran.empty:
            # 3. Sentiment Analysis (Hugging Face)
            with st.spinner('Analiziram sentiment in besedilo...'):
                results = sentiment_analyzer(df_filtriran['Komentar'].tolist())
                df_filtriran['Sentiment'] = [res['label'] for res in results]
                df_filtriran['Confidence'] = [res['score'] for res in results]

            # --- VIZUALIZACIJA (Dva stolpca) ---
            col1, col2 = st.columns(2)
            
            with col1:
                # Grafikon sentimenta
                st.subheader("Porazdelitev sentimenta")
                st.bar_chart(df_filtriran['Sentiment'].value_counts())
                
                # WORD CLOUD (Tvoj dodatek)
                st.subheader("☁️ Najpogostejše besede")
                besedilo = " ".join(k for k in df_filtriran.Komentar)
                wc = WordCloud(width=600, height=300, background_color='white').generate(besedilo)
                fig, ax = plt.subplots()
                ax.imshow(wc, interpolation='bilinear')
                ax.axis("off")
                st.pyplot(fig)

            with col2:
                # Tabela in metrike
                st.subheader("Podatki za mesec " + izbran_mesec)
                st.dataframe(df_filtriran[['Komentar', 'Sentiment', 'Confidence']], use_container_width=True)
                
                avg_conf = df_filtriran['Confidence'].mean()
                st.metric("Povprečna zanesljivost modela", f"{avg_conf:.2%}")
        else:
            st.warning(f"Za mesec {izbran_mesec} ni podatkov. Poskusite drug mesec.")
            
    else:
        # Prikaz za Products/Testimonials
        st.header(f"📍 Razdelek: {izbira}")
        st.write(f"Število zapisov: **{len(df_tip)}**")
        st.dataframe(df_tip[['Komentar', 'Datum']], use_container_width=True)

else:
    st.error("Manjka datoteka 'podatki_2023.csv' na GitHubu!")
