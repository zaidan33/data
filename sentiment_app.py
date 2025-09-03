import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import re
import string
from datetime import datetime, timedelta
import requests
from bs4 import BeautifulSoup
import nltk
from textblob import TextBlob
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.pipeline import Pipeline
import joblib
import warnings
warnings.filterwarnings('ignore')

# Set page config
st.set_page_config(
    page_title="Analisis Sentimen Demo Indonesia 25-30 Agustus 2025",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Download NLTK data
@st.cache_resource
def download_nltk_data():
    try:
        nltk.data.find('tokenizers/punkt')
        nltk.data.find('corpora/stopwords')
    except LookupError:
        nltk.download('punkt')
        nltk.download('stopwords')
        nltk.download('vader_lexicon')

download_nltk_data()

class IndonesianSentimentAnalyzer:
    def __init__(self):
        # Indonesian stopwords
        self.indonesian_stopwords = set([
            'yang', 'di', 'ke', 'dari', 'dengan', 'untuk', 'pada', 'adalah', 'ini', 'itu',
            'dan', 'atau', 'tidak', 'bukan', 'juga', 'akan', 'telah', 'sudah', 'sedang',
            'dapat', 'bisa', 'harus', 'masih', 'sangat', 'lebih', 'paling', 'saya', 'kamu',
            'kami', 'mereka', 'dia', 'nya', 'mu', 'ku', 'anda', 'kita', 'kalian'
        ])
        
        # Indonesian positive and negative words (simplified lexicon)
        self.positive_words = set([
            'bagus', 'baik', 'hebat', 'luar biasa', 'mantap', 'keren', 'sukses', 'berhasil',
            'positif', 'mendukung', 'setuju', 'senang', 'gembira', 'bangga', 'puas', 'cinta',
            'suka', 'amazing', 'good', 'great', 'excellent', 'wonderful', 'fantastic', 'love',
            'like', 'happy', 'glad', 'proud', 'best', 'perfect', 'awesome', 'brilliant'
        ])
        
        self.negative_words = set([
            'buruk', 'jelek', 'tidak baik', 'gagal', 'kecewa', 'sedih', 'marah', 'kesal',
            'benci', 'tidak suka', 'menentang', 'tidak setuju', 'negatif', 'korup', 'bobrok',
            'bad', 'terrible', 'awful', 'horrible', 'hate', 'angry', 'sad', 'disappointed',
            'worst', 'fail', 'stupid', 'idiot', 'corrupt', 'evil', 'wrong'
        ])
        
        self.models = {}
        self.vectorizer = None
    
    def preprocess_text(self, text):
        """Preprocess Indonesian text"""
        if pd.isna(text) or text == '':
            return ''
        
        # Convert to lowercase
        text = text.lower()
        
        # Remove URLs
        text = re.sub(r'http\S+|www\S+|https\S+', '', text)
        
        # Remove mentions and hashtags
        text = re.sub(r'@\w+|#\w+', '', text)
        
        # Remove special characters and numbers
        text = re.sub(r'[^a-zA-Z\s]', '', text)
        
        # Remove extra whitespace
        text = ' '.join(text.split())
        
        # Remove stopwords
        words = text.split()
        words = [word for word in words if word not in self.indonesian_stopwords and len(word) > 2]
        
        return ' '.join(words)
    
    def lexicon_based_sentiment(self, text):
        """Simple lexicon-based sentiment analysis"""
        words = text.lower().split()
        positive_score = sum(1 for word in words if word in self.positive_words)
        negative_score = sum(1 for word in words if word in self.negative_words)
        
        if positive_score > negative_score:
            return 'Positif'
        elif negative_score > positive_score:
            return 'Negatif'
        else:
            return 'Netral'
    
    def textblob_sentiment(self, text):
        """TextBlob sentiment analysis"""
        try:
            blob = TextBlob(text)
            polarity = blob.sentiment.polarity
            
            if polarity > 0.1:
                return 'Positif'
            elif polarity < -0.1:
                return 'Negatif'
            else:
                return 'Netral'
        except:
            return 'Netral'
    
    def train_ml_models(self, df, text_column, label_column):
        """Train multiple ML models"""
        # Prepare data
        X = df[text_column].fillna('').apply(self.preprocess_text)
        y = df[label_column]
        
        # Remove empty texts
        mask = X.str.len() > 0
        X = X[mask]
        y = y[mask]
        
        if len(X) == 0:
            st.error("Tidak ada data teks yang valid untuk training")
            return
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Vectorization
        self.vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1, 2))
        X_train_vec = self.vectorizer.fit_transform(X_train)
        X_test_vec = self.vectorizer.transform(X_test)
        
        # Train models
        models = {
            'Naive Bayes': MultinomialNB(),
            'SVM': SVC(kernel='linear', probability=True),
            'Logistic Regression': LogisticRegression(max_iter=1000),
            'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42)
        }
        
        results = {}
        
        for name, model in models.items():
            try:
                model.fit(X_train_vec, y_train)
                y_pred = model.predict(X_test_vec)
                accuracy = accuracy_score(y_test, y_pred)
                
                self.models[name] = model
                results[name] = {
                    'accuracy': accuracy,
                    'predictions': y_pred,
                    'actual': y_test
                }
                
                st.write(f"**{name}**")
                st.write(f"Accuracy: {accuracy:.3f}")
                
                # Classification report
                report = classification_report(y_test, y_pred, output_dict=True)
                report_df = pd.DataFrame(report).transpose()
                st.dataframe(report_df.round(3))
                
            except Exception as e:
                st.error(f"Error training {name}: {str(e)}")
        
        return results
    
    def predict_sentiment(self, text, method='lexicon'):
        """Predict sentiment using specified method"""
        processed_text = self.preprocess_text(text)
        
        if method == 'lexicon':
            return self.lexicon_based_sentiment(processed_text)
        elif method == 'textblob':
            return self.textblob_sentiment(processed_text)
        elif method in self.models and self.vectorizer:
            try:
                text_vec = self.vectorizer.transform([processed_text])
                prediction = self.models[method].predict(text_vec)[0]
                return prediction
            except:
                return 'Netral'
        else:
            return 'Netral'

def scrape_news_sample():
    """Generate sample news data (simulated scraping)"""
    # Sample data that simulates scraped news about demos
    sample_data = [
        {
            'tanggal': '2025-08-25',
            'sumber': 'DetikNews',
            'judul': 'Mahasiswa Gelar Demo Damai Menuntut Transparansi Pemerintah',
            'teks': 'Ratusan mahasiswa dari berbagai universitas menggelar demonstrasi damai di depan gedung DPR. Mereka menuntut transparansi dalam kebijakan pemerintah dan pemberantasan korupsi.',
            'kategori': 'Politik'
        },
        {
            'tanggal': '2025-08-26',
            'sumber': 'Kompas.com',
            'judul': 'Polisi Amankan Situasi Demo, Tidak Ada Bentrokan',
            'teks': 'Kepolisian berhasil mengamankan situasi demonstrasi dengan baik. Tidak terjadi bentrokan dan acara berjalan tertib sesuai prosedur.',
            'kategori': 'Keamanan'
        },
        {
            'tanggal': '2025-08-27',
            'sumber': 'CNN Indonesia',
            'judul': 'Pemerintah Tanggapi Positif Aspirasi Demonstran',
            'teks': 'Pemerintah menyambut baik aspirasi yang disampaikan demonstran dan berjanji untuk menindaklanjuti tuntutan yang konstruktif.',
            'kategori': 'Politik'
        },
        {
            'tanggal': '2025-08-28',
            'sumber': 'Tempo.co',
            'judul': 'Demo Berlanjut, Massa Semakin Berkurang',
            'teks': 'Demonstrasi hari ketiga diikuti massa yang semakin berkurang. Beberapa tuntutan sudah mendapat respon positif dari pemerintah.',
            'kategori': 'Politik'
        },
        {
            'tanggal': '2025-08-29',
            'sumber': 'Antara News',
            'judul': 'Evaluasi Demo: Berjalan Damai dan Demokratis',
            'teks': 'Secara keseluruhan, rangkaian demonstrasi berjalan dengan damai dan menunjukkan kematangan demokrasi di Indonesia.',
            'kategori': 'Politik'
        },
        {
            'tanggal': '2025-08-30',
            'sumber': 'Tribun News',
            'judul': 'Mahasiswa Apresiasi Dialog dengan Pemerintah',
            'teks': 'Perwakilan mahasiswa mengapresiasi kesempatan dialog yang diberikan pemerintah dan berharap komitmen akan ditepati.',
            'kategori': 'Politik'
        }
    ]
    
    # Add social media simulation
    social_media_data = [
        {
            'tanggal': '2025-08-25',
            'sumber': 'Twitter',
            'judul': 'Tweet tentang demo',
            'teks': 'Demo mahasiswa hari ini sangat tertib dan damai. Semoga aspirasi mereka didengar pemerintah #DemoMahasiswa',
            'kategori': 'Media Sosial'
        },
        {
            'tanggal': '2025-08-26',
            'sumber': 'Facebook',
            'judul': 'Post Facebook',
            'teks': 'Bangga dengan generasi muda yang bisa menyampaikan aspirasi dengan cara yang beradab dan tidak anarkis',
            'kategori': 'Media Sosial'
        },
        {
            'tanggal': '2025-08-27',
            'sumber': 'Instagram',
            'judul': 'Story Instagram',
            'teks': 'Demo ini menunjukkan bahwa demokrasi di Indonesia sudah matang. Tidak ada kerusuhan, semua berjalan tertib',
            'kategori': 'Media Sosial'
        },
        {
            'tanggal': '2025-08-28',
            'sumber': 'Twitter',
            'judul': 'Tweet negatif',
            'teks': 'Demo tidak ada gunanya, buang-buang waktu saja. Pemerintah tidak akan peduli dengan tuntutan mahasiswa',
            'kategori': 'Media Sosial'
        },
        {
            'tanggal': '2025-08-29',
            'sumber': 'TikTok',
            'judul': 'Video TikTok',
            'teks': 'Salut sama mahasiswa yang demo dengan cara yang santun. Ini contoh demokrasi yang baik',
            'kategori': 'Media Sosial'
        }
    ]
    
    return pd.DataFrame(sample_data + social_media_data)

def create_manual_labels():
    """Create manual sentiment labels for training"""
    labels = ['Positif', 'Positif', 'Positif', 'Netral', 'Positif', 'Positif', 
             'Positif', 'Positif', 'Positif', 'Negatif', 'Positif']
    return labels

def main():
    st.title("📊 Analisis Sentimen Demo Indonesia")
    st.subheader("25-30 Agustus 2025")
    
    # Initialize analyzer
    if 'analyzer' not in st.session_state:
        st.session_state.analyzer = IndonesianSentimentAnalyzer()
    
    # Sidebar
    st.sidebar.title("🔧 Pengaturan")
    
    # Navigation
    page = st.sidebar.selectbox(
        "Pilih Halaman:",
        ["🏠 Beranda", "📥 Data Collection", "🔄 Preprocessing", "🤖 Model Training", 
         "📈 Analisis Sentimen", "📊 Visualisasi", "📋 Laporan"]
    )
    
    if page == "🏠 Beranda":
        st.markdown("""
        ## Selamat Datang di Aplikasi Analisis Sentimen Demo Indonesia
        
        Aplikasi ini dirancang untuk menganalisis sentimen publik terkait demonstrasi yang terjadi di Indonesia 
        pada tanggal 25-30 Agustus 2025.
        
        ### 🎯 Fitur Utama:
        - **Data Collection**: Simulasi pengumpulan data dari berbagai sumber
        - **Preprocessing**: Pembersihan dan normalisasi teks bahasa Indonesia
        - **Model Training**: Pelatihan berbagai model machine learning
        - **Sentiment Analysis**: Analisis sentimen dengan multiple methods
        - **Visualisasi**: Dashboard interaktif untuk eksplorasi data
        - **Laporan**: Generate laporan komprehensif
        
        ### 📊 Metodologi:
        1. **Rule-based**: Menggunakan lexicon bahasa Indonesia
        2. **TextBlob**: Library NLP populer
        3. **Machine Learning**: Naive Bayes, SVM, Logistic Regression, Random Forest
        
        ### 🚀 Cara Penggunaan:
        1. Mulai dengan **Data Collection** untuk mendapatkan data
        2. Lakukan **Preprocessing** untuk membersihkan data
        3. **Training Model** untuk membangun classifier
        4. Jalankan **Analisis Sentimen** pada data
        5. Eksplorasi hasil melalui **Visualisasi**
        6. Generate **Laporan** final
        """)
        
        # Statistics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Sources", "6+")
        with col2:
            st.metric("Analysis Methods", "4")
        with col3:
            st.metric("ML Models", "4")
        with col4:
            st.metric("Date Range", "6 Hari")
    
    elif page == "📥 Data Collection":
        st.header("📥 Pengumpulan Data")
        
        st.markdown("""
        ### Sumber Data:
        - 🗞️ Media Online (DetikNews, Kompas, CNN Indonesia, dll)
        - 📱 Media Sosial (Twitter, Facebook, Instagram, TikTok)
        - 💬 Forum Diskusi (Reddit, Kaskus)
        - 📺 YouTube Comments
        """)
        
        if st.button("🔄 Simulasi Scraping Data", type="primary"):
            with st.spinner("Mengumpulkan data..."):
                # Simulate data collection
                df = scrape_news_sample()
                st.session_state.raw_data = df
                
                st.success(f"✅ Berhasil mengumpulkan {len(df)} artikel/post")
                
                # Display sample data
                st.subheader("📄 Sample Data")
                st.dataframe(df)
                
                # Data statistics
                st.subheader("📊 Statistik Data")
                col1, col2 = st.columns(2)
                
                with col1:
                    source_counts = df['sumber'].value_counts()
                    fig_source = px.bar(
                        x=source_counts.index, 
                        y=source_counts.values,
                        title="Distribusi Sumber Data",
                        labels={'x': 'Sumber', 'y': 'Jumlah'}
                    )
                    st.plotly_chart(fig_source, use_container_width=True)
                
                with col2:
                    category_counts = df['kategori'].value_counts()
                    fig_cat = px.pie(
                        values=category_counts.values, 
                        names=category_counts.index,
                        title="Distribusi Kategori"
                    )
                    st.plotly_chart(fig_cat, use_container_width=True)
    
    elif page == "🔄 Preprocessing":
        st.header("🔄 Preprocessing Data")
        
        if 'raw_data' not in st.session_state:
            st.warning("⚠️ Silakan lakukan Data Collection terlebih dahulu!")
            return
        
        df = st.session_state.raw_data.copy()
        
        st.subheader("📝 Teks Asli vs Hasil Preprocessing")
        
        # Show preprocessing example
        sample_idx = st.selectbox("Pilih Sample:", range(len(df)))
        original_text = df.iloc[sample_idx]['teks']
        processed_text = st.session_state.analyzer.preprocess_text(original_text)
        
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**Teks Asli:**")
            st.text_area("", original_text, height=100, disabled=True)
        
        with col2:
            st.markdown("**Setelah Preprocessing:**")
            st.text_area("", processed_text, height=100, disabled=True)
        
        if st.button("🔄 Proses Semua Data", type="primary"):
            with st.spinner("Memproses data..."):
                df['teks_processed'] = df['teks'].apply(st.session_state.analyzer.preprocess_text)
                st.session_state.processed_data = df
                
                st.success("✅ Preprocessing selesai!")
                
                # Show statistics
                st.subheader("📊 Statistik Preprocessing")
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    avg_original = df['teks'].str.len().mean()
                    st.metric("Rata-rata Panjang Asli", f"{avg_original:.0f} karakter")
                
                with col2:
                    avg_processed = df['teks_processed'].str.len().mean()
                    st.metric("Rata-rata Panjang Processed", f"{avg_processed:.0f} karakter")
                
                with col3:
                    reduction = ((avg_original - avg_processed) / avg_original) * 100
                    st.metric("Pengurangan", f"{reduction:.1f}%")
                
                st.dataframe(df[['judul', 'teks', 'teks_processed']])
    
    elif page == "🤖 Model Training":
        st.header("🤖 Pelatihan Model")
        
        if 'processed_data' not in st.session_state:
            st.warning("⚠️ Silakan lakukan Preprocessing terlebih dahulu!")
            return
        
        df = st.session_state.processed_data.copy()
        
        st.markdown("""
        ### 🎯 Manual Labeling
        Untuk melatih model supervised learning, kita perlu label sentimen manual.
        Dalam aplikasi nyata, ini akan dilakukan oleh annotator manusia.
        """)
        
        # Add manual labels (in real scenario, this would be done by human annotators)
        if st.button("📝 Buat Label Manual", type="primary"):
            labels = create_manual_labels()
            df['sentiment'] = labels[:len(df)]
            st.session_state.labeled_data = df
            
            st.success("✅ Label manual berhasil dibuat!")
            
            # Show label distribution
            sentiment_counts = df['sentiment'].value_counts()
            fig = px.bar(
                x=sentiment_counts.index,
                y=sentiment_counts.values,
                title="Distribusi Label Sentimen",
                color=sentiment_counts.index,
                color_discrete_map={
                    'Positif': 'green',
                    'Negatif': 'red', 
                    'Netral': 'gray'
                }
            )
            st.plotly_chart(fig, use_container_width=True)
            
            st.dataframe(df[['judul', 'teks_processed', 'sentiment']])
        
        # Train models
        if 'labeled_data' in st.session_state:
            st.subheader("🔧 Training Models")
            
            if st.button("🚀 Train All Models", type="primary"):
                with st.spinner("Training models..."):
                    results = st.session_state.analyzer.train_ml_models(
                        st.session_state.labeled_data, 
                        'teks_processed', 
                        'sentiment'
                    )
                    st.session_state.training_results = results
                    
                    st.success("✅ Semua model berhasil dilatih!")
    
    elif page == "📈 Analisis Sentimen":
        st.header("📈 Analisis Sentimen")
        
        if 'processed_data' not in st.session_state:
            st.warning("⚠️ Silakan lakukan Preprocessing terlebih dahulu!")
            return
        
        df = st.session_state.processed_data.copy()
        
        # Choose analysis method
        method = st.selectbox(
            "Pilih Metode Analisis:",
            ["lexicon", "textblob", "Naive Bayes", "SVM", "Logistic Regression", "Random Forest"]
        )
        
        if st.button("🔍 Analisis Semua Data", type="primary"):
            with st.spinner(f"Menganalisis sentimen dengan {method}..."):
                df[f'sentiment_{method}'] = df['teks_processed'].apply(
                    lambda x: st.session_state.analyzer.predict_sentiment(x, method)
                )
                
                st.session_state.sentiment_results = df
                
                # Show results
                sentiment_counts = df[f'sentiment_{method}'].value_counts()
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.subheader("📊 Distribusi Sentimen")
                    fig = px.pie(
                        values=sentiment_counts.values,
                        names=sentiment_counts.index,
                        title=f"Hasil Analisis - {method}",
                        color_discrete_map={
                            'Positif': 'green',
                            'Negatif': 'red',
                            'Netral': 'gray'
                        }
                    )
                    st.plotly_chart(fig, use_container_width=True)
                
                with col2:
                    st.subheader("📈 Statistik")
                    total = len(df)
                    for sentiment, count in sentiment_counts.items():
                        percentage = (count/total)*100
                        st.metric(sentiment, f"{count} ({percentage:.1f}%)")
        
        # Individual text analysis
        st.subheader("🔍 Test Individual Text")
        test_text = st.text_area("Masukkan teks untuk dianalisis:")
        
        if test_text and st.button("Analisis"):
            result = st.session_state.analyzer.predict_sentiment(test_text, method)
            
            # Display result with color
            color = 'green' if result == 'Positif' else 'red' if result == 'Negatif' else 'gray'
            st.markdown(f"**Hasil:** <span style='color:{color}'>{result}</span>", 
                       unsafe_allow_html=True)
            
            # Show processed text
            processed = st.session_state.analyzer.preprocess_text(test_text)
            st.text_area("Teks setelah preprocessing:", processed, disabled=True)
    
    elif page == "📊 Visualisasi":
        st.header("📊 Visualisasi Data")
        
        if 'sentiment_results' not in st.session_state:
            st.warning("⚠️ Silakan lakukan Analisis Sentimen terlebih dahulu!")
            return
        
        df = st.session_state.sentiment_results.copy()
        
        # Get sentiment column (find the latest one)
        sentiment_cols = [col for col in df.columns if col.startswith('sentiment_')]
        if not sentiment_cols:
            st.error("Tidak ada hasil analisis sentimen yang ditemukan!")
            return
        
        sentiment_col = sentiment_cols[-1]  # Use the latest analysis
        
        st.subheader("📈 Dashboard Sentimen")
        
        # Timeline analysis
        df['tanggal'] = pd.to_datetime(df['tanggal'])
        
        # Sentiment over time
        daily_sentiment = df.groupby(['tanggal', sentiment_col]).size().unstack(fill_value=0)
        
        fig_timeline = go.Figure()
        
        colors = {'Positif': 'green', 'Negatif': 'red', 'Netral': 'gray'}
        
        for sentiment in daily_sentiment.columns:
            fig_timeline.add_trace(go.Scatter(
                x=daily_sentiment.index,
                y=daily_sentiment[sentiment],
                mode='lines+markers',
                name=sentiment,
                line=dict(color=colors.get(sentiment, 'blue'))
            ))
        
        fig_timeline.update_layout(
            title="Tren Sentimen Harian",
            xaxis_title="Tanggal",
            yaxis_title="Jumlah Post",
            hovermode='x unified'
        )
        
        st.plotly_chart(fig_timeline, use_container_width=True)
        
        # Sentiment by source
        col1, col2 = st.columns(2)
        
        with col1:
            source_sentiment = pd.crosstab(df['sumber'], df[sentiment_col])
            
            fig_source = px.bar(
                source_sentiment.reset_index(),
                x='sumber',
                y=['Positif', 'Negatif', 'Netral'],
                title="Sentimen per Sumber",
                color_discrete_map=colors
            )
            st.plotly_chart(fig_source, use_container_width=True)
        
        with col2:
            category_sentiment = pd.crosstab(df['kategori'], df[sentiment_col])
            
            fig_cat = px.bar(
                category_sentiment.reset_index(),
                x='kategori', 
                y=['Positif', 'Negatif', 'Netral'],
                title="Sentimen per Kategori",
                color_discrete_map=colors
            )
            st.plotly_chart(fig_cat, use_container_width=True)
        
        # Word cloud simulation (top positive and negative words)
        st.subheader("☁️ Kata-kata Dominan")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**💚 Kata Positif Dominan**")
            positive_texts = df[df[sentiment_col] == 'Positif']['teks_processed'].str.cat(sep=' ')
            positive_words = positive_texts.split()
            
            if positive_words:
                from collections import Counter
                top_positive = Counter(positive_words).most_common(10)
                pos_df = pd.DataFrame(top_positive, columns=['Kata', 'Frekuensi'])
                st.dataframe(pos_df)
        
        with col2:
            st.markdown("**❤️ Kata Negatif Dominan**")
            negative_texts = df[df[sentiment_col] == 'Negatif']['teks_processed'].str.cat(sep=' ')
            negative_words = negative_texts.split()
            
            if negative_words:
                from collections import Counter
                top_negative = Counter(negative_words).most_common(10)
                neg_df = pd.DataFrame(top_negative, columns=['Kata', 'Frekuensi'])
                st.dataframe(neg_df)
        
        # Detailed data view
        st.subheader("📋 Data Detail")
        
        # Filter options
        col1, col2, col3 = st.columns(3)
        
        with col1:
            source_filter = st.selectbox("Filter Sumber:", ['Semua'] + list(df['sumber'].unique()))
        
        with col2:
            sentiment_filter = st.selectbox("Filter Sentimen:", ['Semua'] + list(df[sentiment_col].unique()))
        
        with col3:
            date_range = st.date_input("Filter Tanggal:", [df['tanggal'].min(), df['tanggal'].max()])
        
        # Apply filters
        filtered_df = df.copy()
        
        if source_filter != 'Semua':
            filtered_df = filtered_df[filtered_df['sumber'] == source_filter]
        
        if sentiment_filter != 'Semua':
            filtered_df = filtered_df[filtered_df[sentiment_col] == sentiment_filter]
        
        if len(date_range) == 2:
            start_date, end_date = date_range
            filtered_df = filtered_df[
                (filtered_df['tanggal'].dt.date >= start_date) & 
                (filtered_df['tanggal'].dt.date <= end_date)
            ]
        
        st.dataframe(
            filtered_df[['tanggal', 'sumber', 'judul', 'teks', sentiment_col]],
            use_container_width=True
        )
    
    elif page == "📋 Laporan":
        st.header("📋 Laporan Analisis Sentimen")
        
        if 'sentiment_results' not in st.session_state:
            st.warning("⚠️ Silakan lakukan Analisis Sentimen terlebih dahulu!")
            return
        
        df = st.session_state.sentiment_results.copy()
        sentiment_cols = [col for col in df.columns if col.startswith('sentiment_')]
        
        if not sentiment_cols:
            st.error("Tidak ada hasil analisis sentimen yang ditemukan!")
            return
        
        sentiment_col = sentiment_cols[-1]
        method_used = sentiment_col.replace('sentiment_', '')
        
        # Generate comprehensive report
        st.subheader("📊 Executive Summary")
        
        total_data = len(df)
        sentiment_counts = df[sentiment_col].value_counts()
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Data", total_data)
        
        with col2:
            positif_pct = (sentiment_counts.get('Positif', 0) / total_data) * 100
            st.metric("Sentimen Positif", f"{positif_pct:.1f}%")
        
        with col3:
            negatif_pct = (sentiment_counts.get('Negatif', 0) / total_data) * 100
            st.metric("Sentimen Negatif", f"{negatif_pct:.1f}%")
        
        with col4:
            netral_pct = (sentiment_counts.get('Netral', 0) / total_data) * 100
            st.metric("Sentimen Netral", f"{netral_pct:.1f}%")
        
        # Key insights
        st.subheader("🔍 Temuan Utama")
        
        insights = []
        
        # Dominant sentiment
        dominant_sentiment = sentiment_counts.index[0]
        dominant_pct = (sentiment_counts.iloc[0] / total_data) * 100
        insights.append(f"• **Sentimen dominan**: {dominant_sentiment} ({dominant_pct:.1f}%)")
        
        # Sentiment trend
        df['tanggal'] = pd.to_datetime(df['tanggal'])
        first_day = df[df['tanggal'] == df['tanggal'].min()]
        last_day = df[df['tanggal'] == df['tanggal'].max()]
        
        first_positive = (first_day[sentiment_col] == 'Positif').sum()
        last_positive = (last_day[sentiment_col] == 'Positif').sum()
        
        if last_positive > first_positive:
            trend = "meningkat"
        elif last_positive < first_positive:
            trend = "menurun"
        else:
            trend = "stabil"
        
        insights.append(f"• **Tren sentimen positif**: {trend} dari awal hingga akhir periode")
        
        # Source analysis
        source_sentiment = pd.crosstab(df['sumber'], df[sentiment_col], normalize='index')
        most_positive_source = source_sentiment['Positif'].idxmax()
        most_positive_pct = source_sentiment['Positif'].max() * 100
        
        insights.append(f"• **Sumber paling positif**: {most_positive_source} ({most_positive_pct:.1f}% positif)")
        
        # Display insights
        for insight in insights:
            st.markdown(insight)
        
        # Detailed analysis by category
        st.subheader("📈 Analisis per Kategori")
        
        category_analysis = df.groupby('kategori')[sentiment_col].value_counts(normalize=True).unstack(fill_value=0) * 100
        st.dataframe(category_analysis.round(1))
        
        # Timeline analysis
        st.subheader("📅 Analisis Timeline")
        
        daily_stats = df.groupby('tanggal').agg({
            sentiment_col: ['count', lambda x: (x == 'Positif').sum(), lambda x: (x == 'Negatif').sum()]
        }).round(2)
        
        daily_stats.columns = ['Total Post', 'Positif', 'Negatif']
        daily_stats['Positif %'] = (daily_stats['Positif'] / daily_stats['Total Post'] * 100).round(1)
        daily_stats['Negatif %'] = (daily_stats['Negatif'] / daily_stats['Total Post'] * 100).round(1)
        
        st.dataframe(daily_stats)
        
        # Methodology
        st.subheader("🔬 Metodologi")
        
        st.markdown(f"""
        **Metode yang Digunakan**: {method_used}
        
        **Preprocessing Steps**:
        - Konversi ke lowercase
        - Penghapusan URL, mention, hashtag
        - Penghapusan karakter khusus dan angka
        - Penghapusan stopwords bahasa Indonesia
        - Normalisasi spasi
        
        **Analisis Sentimen**:
        - Lexicon-based: Menggunakan dictionary kata positif/negatif bahasa Indonesia
        - TextBlob: Library NLP dengan support bahasa Indonesia
        - Machine Learning: Model yang dilatih dengan data berlabel manual
        
        **Klasifikasi**:
        - **Positif**: Sentimen mendukung, menyetujui, atau menunjukkan apresiasi
        - **Negatif**: Sentimen menolak, mengkritik, atau menunjukkan ketidakpuasan
        - **Netral**: Sentimen objektif atau tidak menunjukkan polaritas yang jelas
        """)
        
        # Limitations
        st.subheader("⚠️ Keterbatasan")
        
        st.markdown("""
        - Data yang digunakan adalah simulasi, bukan hasil scraping real-time
        - Label manual dibuat untuk tujuan demonstrasi
        - Model belum dioptimalkan dengan hyperparameter tuning
        - Belum menangani sarkasme dan ironi secara optimal
        - Context-specific sentiment belum sepenuhnya terakomodasi
        """)
        
        # Recommendations
        st.subheader("💡 Rekomendasi")
        
        recommendations = []
        
        if positif_pct > 60:
            recommendations.append("• Sentimen positif yang tinggi menunjukkan dukungan publik terhadap cara penyampaian aspirasi")
        
        if negatif_pct > 30:
            recommendations.append("• Perlu dialog lebih intensif untuk menangani sentimen negatif")
        
        if netral_pct > 40:
            recommendations.append("• Banyak publik yang masih netral, perlu edukasi dan komunikasi yang lebih baik")
        
        recommendations.extend([
            "• Tingkatkan transparansi komunikasi pemerintah",
            "• Manfaatkan media sosial untuk engagement yang lebih baik",
            "• Monitor sentimen secara real-time untuk respon yang cepat"
        ])
        
        for rec in recommendations:
            st.markdown(rec)
        
        # Download report
        st.subheader("💾 Download Laporan")
        
        # Create downloadable CSV
        report_data = df[['tanggal', 'sumber', 'kategori', 'judul', 'teks', sentiment_col]].copy()
        csv = report_data.to_csv(index=False)
        
        st.download_button(
            label="📊 Download Data CSV",
            data=csv,
            file_name=f"sentiment_analysis_demo_indonesia_{datetime.now().strftime('%Y%m%d')}.csv",
            mime="text/csv"
        )
        
        # Create summary report
        summary_report = f"""
LAPORAN ANALISIS SENTIMEN
Demo Indonesia 25-30 Agustus 2025

=== RINGKASAN EKSEKUTIF ===
Total Data Dianalisis: {total_data}
Metode Analisis: {method_used}
Periode: 25-30 Agustus 2025

=== DISTRIBUSI SENTIMEN ===
Positif: {sentiment_counts.get('Positif', 0)} ({positif_pct:.1f}%)
Negatif: {sentiment_counts.get('Negatif', 0)} ({negatif_pct:.1f}%)
Netral: {sentiment_counts.get('Netral', 0)} ({netral_pct:.1f}%)

=== TEMUAN UTAMA ===
{chr(10).join(insights)}

=== REKOMENDASI ===
{chr(10).join(recommendations)}

Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
        """
        
        st.download_button(
            label="📄 Download Summary Report",
            data=summary_report,
            file_name=f"sentiment_summary_{datetime.now().strftime('%Y%m%d')}.txt",
            mime="text/plain"
        )

if __name__ == "__main__":
    main()