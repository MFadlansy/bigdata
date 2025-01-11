import subprocess
import sys
import os

# Fungsi untuk menginstal dependensi dari requirements.txt jika belum terinstal
def install_requirements():
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
        print("Semua dependensi berhasil diinstal.")
    except subprocess.CalledProcessError as e:
        print(f"Terjadi kesalahan saat menginstal dependensi: {e}")
        sys.exit(1)

# Cek dan install dependensi
install_requirements()

# Import library yang diperlukan
import requests as req
from bs4 import BeautifulSoup as bs
import csv
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch.nn.functional as F
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import streamlit as st
import pandas as pd
import plotly.express as px

# Headers untuk HTTP request
hades = {
    'user-agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/106.0.0.0 Safari/537.36'
}

# Load tokenizer dan model dari Hugging Face
tokenizer = AutoTokenizer.from_pretrained("w11wo/indonesian-roberta-base-sentiment-classifier")
model = AutoModelForSequenceClassification.from_pretrained("w11wo/indonesian-roberta-base-sentiment-classifier")

# Fungsi untuk scraping dan analisis sentimen
def scrape_detik(query, hal):
    headlines = []
    with open('politik_simple.csv', 'a', newline='', encoding='utf-8') as file:
        wr = csv.writer(file, delimiter=',')
        for page in range(1, hal + 1):
            url = f'https://www.detik.com/search/searchnews?query={query}&page={page}&result_type=latest'
            ge = req.get(url, headers=hades).text
            try:
                sop = bs(ge, 'lxml')  # Gunakan 'html.parser' jika 'lxml' tidak tersedia
            except Exception as e:
                print(f"Error saat parsing HTML: {e}")
                sop = bs(ge, 'html.parser')

            li = sop.find('div', class_='list-content')
            if not li:
                continue

            lin = li.find_all('article')
            for x in lin:
                link_tag = x.find('a')
                if link_tag:
                    link = link_tag['href']
                    date_span = x.find('div', class_='media__date').find('span')
                    date = date_span.text.replace('WIB', '').replace('detikNews',
                                                                     '').strip() if date_span else 'No Date Found'
                    headline = link_tag.get('dtr-ttl', 'No Headline Found').strip()

                    headlines.append([headline, date, link])
                    wr.writerow([headline, date, link])
    return headlines

def predict_sentiment(headlines):
    sentiments = []
    all_important_words = {'positif': [], 'negatif': []}

    for headline, _, _ in headlines:
        inputs = tokenizer(headline, return_tensors="pt", truncation=True, padding=True, max_length=512)
        logits = model(**inputs).logits
        probabilities = F.softmax(logits, dim=1)

        pos_prob = probabilities[0][0].item()
        neg_prob = probabilities[0][2].item()
        sentiment = 'positif' if pos_prob > neg_prob else 'negatif'
        sentiments.append(sentiment)

        all_important_words[sentiment].append(headline)

    return sentiments, all_important_words

def display_wordcloud(all_important_words):
    st.subheader("Wordcloud")
    col1, col2 = st.columns(2)

    for sentiment, col in zip(['positif', 'negatif'], [col1, col2]):
        text = ' '.join(all_important_words[sentiment])
        wc = WordCloud(width=300, height=300, background_color='white').generate(text)

        col.image(wc.to_array(), caption=f"Wordcloud - {sentiment.capitalize()}", use_container_width=True)

# Fungsi untuk menyimpan hasil ke file CSV
def save_to_csv(data, filename):
    df = pd.DataFrame(data, columns=['Headline', 'Tanggal', 'Link', 'Sentimen'])
    df.to_csv(filename, index=False, encoding='utf-8')
    st.success(f"Hasil telah disimpan ke {filename}")

# Fungsi untuk menampilkan pie chart
def display_pie_chart(sentiments):
    # Memastikan sentimen memiliki nilai 'Positif' dan 'Negatif'
    sentiment_counts = pd.DataFrame({'Sentimen': sentiments}).value_counts().reset_index()
    sentiment_counts.columns = ['Sentimen', 'Jumlah']

    # Menentukan warna untuk sentimen positif dan negatif
    color_map = {'Positif': 'green', 'Negatif': 'red'}

    # Membuat pie chart
    fig = px.pie(sentiment_counts, names='Sentimen', values='Jumlah', title='Distribusi Sentimen')

    # Menyesuaikan warna dengan menggunakan update_traces
    fig.update_traces(marker=dict(colors=[color_map.get(sentiment, 'gray') for sentiment in sentiment_counts['Sentimen']]))

    # Menampilkan pie chart
    st.plotly_chart(fig)

# Streamlit UI
st.title("Sentiment Analysis Scraping")

query = st.text_input("Masukkan Kalimat", "")
hal = st.number_input("Jumlah Halaman", min_value=1, step=1, value=1)

if st.button("Mulai Scraping"):
    with st.spinner("Sedang melakukan scraping..."):
        headlines = scrape_detik(query, hal)
        sentiments, all_important_words = predict_sentiment(headlines)

        pos_count = sentiments.count('positif')
        neg_count = sentiments.count('negatif')

        st.success("Scraping dan analisis selesai!")
        st.write(f"Total Artikel: {len(headlines)} | Positif: {pos_count} | Negatif: {neg_count}")

        # Tampilkan hasil artikel
        st.subheader("Artikel Positif")
        st.write([headline[0] for headline, sentiment in zip(headlines, sentiments) if sentiment == 'positif'])

        st.subheader("Artikel Negatif")
        st.write([headline[0] for headline, sentiment in zip(headlines, sentiments) if sentiment == 'negatif'])

        # Tampilkan WordCloud
        display_wordcloud(all_important_words)

        # Tampilkan Pie Chart
        display_pie_chart(sentiments)

        # Simpan ke file CSV
        if st.button("Simpan Hasil ke CSV"):
            save_to_csv([(headline[0], headline[1], headline[2], sentiment) for headline, sentiment in zip(headlines, sentiments)], 'hasil_scraping.csv')
