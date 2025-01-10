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
import gdown
import zipfile

# Headers untuk HTTP request
hades = {
    'user-agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/106.0.0.0 Safari/537.36'
}

# Unduh model dari Google Drive jika belum ada
model_url = 'https://drive.google.com/uc?id=1J0DYlDE_7JDZeITOJ_gX7NKF5UfFgDd1'
model_path = './indonesian-roberta-base-sentiment-classifier'

if not os.path.exists(model_path):
    os.makedirs(model_path)
    gdown.download(model_url, output=model_path + '/model.zip', quiet=False)

    # Ekstrak model
    with zipfile.ZipFile(model_path + '/model.zip', 'r') as zip_ref:
        zip_ref.extractall(model_path)
    os.remove(model_path + '/model.zip')  # Hapus file zip setelah diekstrak

# Memuat tokenizer dan model dari folder lokal
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForSequenceClassification.from_pretrained(model_path)


# Fungsi untuk scraping dan analisis sentimen
def scrape_detik(query, hal):
    headlines = []
    with open('politik_simple.csv', 'a', newline='', encoding='utf-8') as file:
        wr = csv.writer(file, delimiter=',')
        for page in range(1, hal + 1):
            url = f'https://www.detik.com/search/searchnews?query={query}&page={page}&result_type=latest'
            ge = req.get(url, headers=hades).text
            sop = bs(ge, 'lxml')
            li = sop.find('div', class_='list-content')
            lin = li.find_all('article')

            for x in lin:
                link_tag = x.find('a')
                if link_tag:
                    link = link_tag['href']
                    date_span = x.find('div', class_='media__date').find('span')
                    date = date_span.text.replace('WIB', '').replace('detikNews',
                                                                     '').strip() if date_span else 'No Date Found'
                    headline = link_tag.get('dtr-ttl', 'No Headline Found').strip()

                    headlines.append(headline)
                    wr.writerow([headline, date, link])
    return headlines


def predict_sentiment(headlines):
    sentiments = []
    all_important_words = {'positif': [], 'negatif': []}

    for headline in headlines:
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

        col.image(wc.to_array(), caption=f"Wordcloud - {sentiment.capitalize()}", use_column_width=True)


# Streamlit UI
st.title("Sentiment Analysis Scraping")

query = st.text_input("Masukkan Query", "")
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
        st.write([headline for headline, sentiment in zip(headlines, sentiments) if sentiment == 'positif'])

        st.subheader("Artikel Negatif")
        st.write([headline for headline, sentiment in zip(headlines, sentiments) if sentiment == 'negatif'])

        # Tampilkan WordCloud
        display_wordcloud(all_important_words)
