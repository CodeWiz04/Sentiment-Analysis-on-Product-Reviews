import streamlit as st
import pandas as pd
import string
from textblob import TextBlob
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import nltk
import numpy as np

# ----------------- SETUP -----------------
nltk.download('stopwords', quiet=True)

# Stopwords and stemmer
stop_words = set(stopwords.words('english'))
ps = PorterStemmer()

# ----------------- PREPROCESSING FUNCTIONS -----------------
def remove_punctuation(text):
    return text.translate(str.maketrans('', '', string.punctuation))

def remove_stopwords(text):
    return ' '.join([word for word in text.split() if word not in stop_words])

def stem_text(text):
    return ' '.join([ps.stem(word) for word in text.split()])

def preprocess_text(text):
    text = text.lower()
    text = remove_punctuation(text)
    text = remove_stopwords(text)
    text = stem_text(text)
    return text

# FIXED: Use actual ratings instead of TextBlob sentiment
def get_sentiment_from_rating(rating):
    """Convert rating to binary sentiment (1-3: negative, 4-5: positive)"""
    return 1 if rating >= 4 else 0

# ----------------- LOAD & TRAIN MODEL -----------------
@st.cache_data
def train_model():
    # Load data
    try:
        df = pd.read_csv("Reviews.csv").head(50000)
    except FileNotFoundError:
        st.error("Reviews.csv file not found. Please ensure the file is in the correct directory.")
        return None, None, None
    
    # Check if required columns exist
    if 'Text' not in df.columns or 'Score' not in df.columns:
        st.error("Required columns 'Text' and 'Score' not found in the CSV file.")
        return None, None, None
    
    # Remove rows with missing text or scores
    df = df.dropna(subset=['Text', 'Score'])
    
    # FIXED: Use actual ratings for sentiment labels
    df['Sentiment'] = df['Score'].apply(get_sentiment_from_rating)
    
    # Check class distribution
    sentiment_counts = df['Sentiment'].value_counts()
    st.sidebar.write("Training Data Distribution:")
    st.sidebar.write(f"Positive: {sentiment_counts.get(1, 0)}")
    st.sidebar.write(f"Negative: {sentiment_counts.get(0, 0)}")
    
    # Preprocess text
    df['Text'] = df['Text'].apply(lambda x: preprocess_text(str(x)))
    
    # Remove empty texts after preprocessing
    df = df[df['Text'].str.len() > 0]
    
    # TF-IDF with better parameters
    tfidf = TfidfVectorizer(
        max_features=5000,
        min_df=2,  # Ignore terms that appear in less than 2 documents
        max_df=0.95,  # Ignore terms that appear in more than 95% of documents
        ngram_range=(1, 2)  # Use both unigrams and bigrams
    )
    
    X = tfidf.fit_transform(df['Text']).toarray()
    y = df['Sentiment']

    # Train/Test split with stratification
    x_train, x_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # FIXED: Logistic Regression with better parameters
    lr = LogisticRegression(
        class_weight='balanced',
        max_iter=2000,
        C=1.0,  # Regularization parameter
        solver='liblinear',
        random_state=42
    )
    lr.fit(x_train, y_train)

    # Evaluate model
    y_pred = lr.predict(x_test)
    acc = accuracy_score(y_test, y_pred)
    
    # Print classification report for debugging
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    
    return lr, tfidf, acc

# Load model with error handling
try:
    model, tfidf, accuracy = train_model()
    if model is None:
        st.stop()
except Exception as e:
    st.error(f"Error loading model: {str(e)}")
    st.stop()

st.set_page_config(
    page_title="Amazon Review Sentiment Analyzer", 
    page_icon="🛒", 
    layout="centered",
    initial_sidebar_state="expanded"  # Show sidebar for debugging info
)

# Your existing CSS (unchanged)
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;600;700&display=swap');

body {
    background: linear-gradient(135deg, #0f2027, #203a43, #2c5364);
    background-attachment: fixed;
    font-family: 'Poppins', sans-serif;
}

.stApp {
    padding-top: 0 !important;
}

.main-container {
    background: rgba(255,255,255,0.05);
    backdrop-filter: blur(10px);
    border-radius: 15px;
    padding: 1.5rem;
    max-width: 900px;
    margin: auto;
    box-shadow: 0 4px 15px rgba(0,0,0,0.3);
}

.header {
    text-align: center;
    padding: 1rem 0;
}

.main-title {
    font-size: 2rem;
    font-weight: 700;
    color: white;
    margin-bottom: 0.4rem;
}

.sub-text {
    font-size: 1rem;
    color: rgba(255,255,255,0.8);
    margin-bottom: 1.5rem;
}

.stTextInput>div>div>input, 
.stTextArea>div>div>textarea {
    border-radius: 10px;
    padding: 0.6rem;
    font-size: 0.95rem;
    background: rgba(15, 23, 42, 0.7);
    color: white;
}

.stButton>button {
    background: linear-gradient(135deg, #ff6b6b, #ff8e53);
    color: white;
    border: none;
    border-radius: 10px;
    padding: 0.6rem;
    font-size: 1rem;
    font-weight: 600;
    width: 100%;
}

.result-card {
    border-radius: 12px;
    padding: 1rem;
    text-align: center;
    margin-top: 1rem;
    font-size: 1rem;
    background: rgba(255, 255, 255, 0.08);
}

.confidence-bar {
    height: 18px;
    border-radius: 9px;
    margin: 0.8rem auto;
    overflow: hidden;
    background: rgba(15, 23, 42, 0.7);
    max-width: 300px;
}

.confidence-fill {
    height: 100%;
}

.footer {
    text-align: center;
    font-size: 0.8rem;
    padding-top: 1rem;
    color: rgba(255, 255, 255, 0.6);
}

.negative {
    background: rgba(255, 107, 107, 0.2);
    border-left: 4px solid #ff6b6b;
}

.positive {
    background: rgba(72, 187, 120, 0.2);
    border-left: 4px solid #48bb78;
}
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="main-container">', unsafe_allow_html=True)

# Header
st.markdown("""
<div class="header">
    <h1 class="main-title">✨ Amazon Review Sentiment Analyzer</h1>
    <p class="sub-text">Instantly detect the tone of product reviews</p>
</div>
""", unsafe_allow_html=True)

# Display model accuracy
if accuracy:
    st.sidebar.metric("Model Accuracy", f"{accuracy:.2%}")

# Form
with st.form("review_form", clear_on_submit=False):
    col1, col2 = st.columns(2)
    with col1:
        product_id = st.text_input("📦 Product ID", placeholder="B07H8QMZSC")
    with col2:
        product_name = st.text_input("🛒 Product Name", placeholder="Wireless Headphones")
    review_text = st.text_area("📝 Your Review", placeholder="Enter your review here...", height=100)
    submitted = st.form_submit_button("Analyze")

# Results
if submitted:
    if not product_id or not product_name or not review_text:
        st.error("⚠ Please fill in all fields.")
    else:
        # Preprocess the input review
        processed = preprocess_text(review_text)
        
        # Check if processed text is empty
        if not processed or len(processed.strip()) == 0:
            st.warning("⚠️ The review text became empty after preprocessing. Please try a different review.")
        else:
            # Transform and predict
            input_vector = tfidf.transform([processed]).toarray()
            
            # Make prediction
            prediction = model.predict(input_vector)[0]
            probabilities = model.predict_proba(input_vector)[0]
            
            # Get confidence for the predicted class
            confidence = probabilities[prediction] * 100
            
            # Show debugging info in sidebar
            st.sidebar.write("**Debugging Info:**")
            st.sidebar.write(f"Processed text: {processed[:100]}...")
            st.sidebar.write(f"Prediction: {prediction}")
            st.sidebar.write(f"Probabilities: Negative={probabilities[0]:.3f}, Positive={probabilities[1]:.3f}")

            # Show results
            if prediction == 1:
                st.markdown(f"""
                    <div class="result-card positive">
                        🌟 <b>Positive Review Detected!</b><br>
                        Product: <b>{product_name}</b> (ID: {product_id})<br>
                        Confidence: {confidence:.2f}%
                    </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                    <div class="result-card negative">
                        ⚠️ <b>Negative Review Detected</b><br>
                        Product: <b>{product_name}</b> (ID: {product_id})<br>
                        Confidence: {confidence:.2f}%
                    </div>
                """, unsafe_allow_html=True)

# Footer
st.markdown('<div class="footer">Amazon Review Sentiment Analyzer © 2025</div>', unsafe_allow_html=True)
st.markdown('</div>', unsafe_allow_html=True)