import streamlit as st
import pandas as pd
import os
import numpy as np
from transformers import pipeline
import matplotlib.pyplot as plt

# 1. Nastavitev in nalaganje modela
st.set_page_config(page_title="Reputation Monitor PRO", layout="wide")
st.title("📊 Brand Reputation Monitor 2023")

@st.cache_resource
def load_sentiment_model():
    # Uporaba zahtevanega modela
    return pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english")

sentiment_analyzer = load_sentiment_model()

if os.path.exists('podatki_2023.csv'):
    df = pd.read_csv('podatki_2023.csv')
    
    # --- SIMULACIJA DATUMOV ZA NALOGO ---
    # Da bo slider deloval na tvojih 45 vrsticah, jim dodelimo različne mesece
    np.random.seed(42)
    df['Mesec_Num'] = np.random.randint(0, 12, size=len(df))
    meseci_imena = ["Jan", "Feb", "Mar", "Apr", "Maj", "Jun", "Jul", "Avg", "Sep", "Okt", "Nov", "Dec"]
    df['Mesec_Ime'] = df['Mesec_Num'].apply(lambda x: meseci_imena[x])

    # --- STRANSKI MENI ---
    st.sidebar.header("Navigacija")
    izbira = st.sidebar.radio("Izberite razdelek:", ["Products", "Testimonials", "Reviews"])
    
    mapping = {"Products": "product", "Testimonials": "testimonial", "Reviews": "review"}
    df_tip = df[df['Tip'] == mapping[izbira]].copy()

    # --- REVIEWS: KLJUČNI DEL NALOGE ---
    if izbira == "Reviews":
        st.header("🔍 Analiza mnenj po mesecih")
        
        # 1. Slider za izbiro meseca
        izbran_mesec = st.select_slider(
            "Izberite mesec za filtriranje mnenj v letu 2023:",
            options=meseci_imena
        )
        
        # 2. Filtriranje podatkov glede na slider
        df_filtriran = df_tip[df_tip['Mesec_Ime'] == izbran_mesec].copy()
        
        st.write(f"Prikazujem **{len(df_filtriran)}** mnenj za mesec **{izbran_mesec} 2023**.")

        if not df_filtriran.empty:
            # 3. Sentiment Analysis (Transformer)
            with st.spinner('Analiziram sentiment...'):
                results = sentiment_analyzer(df_filtriran['Komentar'].tolist())
                df_filtriran['Sentiment'] = [res['label'] for res in results]
                df_filtriran['Confidence'] = [res['score'] for res in results]

            # 4. Vizualizacija (Bar Chart in Confidence)
            col1, col2 = st.columns([1, 1])
            
            with col1:
                st.subheader("Porazdelitev mnenj")
                sentiment_counts = df_filtriran['Sentiment'].value_counts()
                st.bar_chart(sentiment_counts)
                
                # Metrika za Confidence Score
                avg_conf = df_filtriran['Confidence'].mean()
                st.metric("Povprečna zanesljivost (Confidence)", f"{avg_conf:.2%}")

            with col2:
                st.subheader("Podrobna tabela")
                st.dataframe(df_filtriran[['Komentar', 'Sentiment', 'Confidence']], use_container_width=True)
        else:
            st.warning(f"Za mesec {izbran_mesec} ni zajetih mnenj. Poskusite drug mesec.")

    else:
        # Prikaz za Products in Testimonials
        st.header(f"📍 Razdelek: {izbira}")
        st.write(f"Skupno število zajetih vrstic: **{len(df_tip)}**")
        st.dataframe(df_tip[['Komentar', 'Datum']], use_container_width=True)

else:
    st.error("Manjka datoteka 'podatki_2023.csv'!")