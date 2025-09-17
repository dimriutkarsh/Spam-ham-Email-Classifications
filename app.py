import streamlit as st
import pickle
import string
from nltk.corpus import stopwords
import nltk
from nltk.stem.porter import PorterStemmer

ps = PorterStemmer()


def transform_text(text):
    text = text.lower()
    text = nltk.word_tokenize(text)

    y = []
    for i in text:
        if i.isalnum():
            y.append(i)

    text = y[:]
    y.clear()

    for i in text:
        if i not in stopwords.words('english') and i not in string.punctuation:
            y.append(i)

    text = y[:]
    y.clear()

    for i in text:
        y.append(ps.stem(i))

    return " ".join(y)

tfidf = pickle.load(open('vectorizer.pkl','rb'))
model = pickle.load(open('model.pkl','rb'))

# ---------- UI Enhancements ----------
st.set_page_config(page_title="Spam Classifier", page_icon="📧", layout="centered")

st.markdown(
    """
    <style>
    .main {
        background: linear-gradient(to right, #ece9e6, #ffffff);
    }
    .title {
        text-align: center;
        font-size: 40px !important;
        font-weight: bold;
        color: #2E86C1;
    }
    .stTextArea textarea {
        border-radius: 12px;
        border: 2px solid #2E86C1;
    }
    .stButton>button {
        background-color: #2E86C1;
        color: white;
        font-size: 18px;
        border-radius: 12px;
        padding: 8px 20px;
        transition: 0.3s;
    }
    .stButton>button:hover {
        background-color: #1B4F72;
    }
    </style>
    """,
    unsafe_allow_html=True
)

st.markdown('<p class="title">📧 Email / SMS Spam Classifier</p>', unsafe_allow_html=True)

st.write("### 🔍 Enter a message below and let AI check if it’s **Spam or Not Spam**.")

# ---------- Input Section ----------
input_sms = st.text_area("✉️ Type your message here:", height=150)

if st.button('🚀 Predict'):
    if input_sms.strip() == "":
        st.warning("⚠️ Please enter a message before predicting.")
    else:
        # 1. preprocess
        transformed_sms = transform_text(input_sms)
        # 2. vectorize
        vector_input = tfidf.transform([transformed_sms])
        # 3. predict
        result = model.predict(vector_input)[0]
        # 4. Display
        if result == 1:
            st.success("🛑 This message looks like **Spam**!")
        else:
            st.success("✅ This message looks **Safe (Not Spam)**.")
