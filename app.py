import streamlit as st
import joblib
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

st.set_page_config(page_title="Anti-Spam AI", page_icon="🛡️", layout="wide")

try:
    stop_words = set(stopwords.words('english'))
except LookupError:
    nltk.download('stopwords', quiet=True)
    stop_words = set(stopwords.words('english'))

stemmer = PorterStemmer()

def clean_text(text):
    text = re.sub('[^a-zA-Z0-9]', ' ', text)
    text = text.lower()
    words = text.split()
    cleaned_words = [stemmer.stem(word) for word in words if word not in stop_words]
    return ' '.join(cleaned_words)

@st.cache_resource
def load_models():
    vec = joblib.load('tfidf_vectorizer.pkl')
    nb = joblib.load('nb_model.pkl')
    lr = joblib.load('lr_model.pkl')
    svm = joblib.load('svm_model.pkl')
    return vec, nb, lr, svm

vectorizer, nb_model, lr_model, svm_model = load_models()

st.title(" Інтелектуальна система класифікації спаму")
st.markdown("Введіть будь-яке повідомлення англійською мовою, щоб протестувати сукупність моделей машинного навчання.")

user_input = st.text_area("Текст повідомлення:", height=150, placeholder="URGENT! You have won a 1 week FREE membership...")

if st.button(" Аналізувати повідомлення", type="primary"):
    if not user_input.strip():
        st.warning("Будь ласка, введіть текст для аналізу.")
    else:
        cleaned_text = clean_text(user_input)
        vec_input = vectorizer.transform([cleaned_text])
        
        def display_result(col, title, model, vec_in):
            pred = model.predict(vec_in)[0]
            prob = model.predict_proba(vec_in)[0]
            
            with col:
                st.subheader(title)
                if pred == 1:
                    st.error(f" СПАМ")
                    st.progress(prob[1], text=f"Впевненість алгоритму: {prob[1]*100:.1f}%")
                else:
                    st.success(f" ЗВИЧАЙНЕ SMS")
                    st.progress(prob[0], text=f"Впевненість алгоритму: {prob[0]*100:.1f}%")

        st.markdown("###  Результати класифікації")
        col1, col2, col3 = st.columns(3)
        display_result(col1, "Наївний Байєс", nb_model, vec_input)
        display_result(col2, "Логістична Регресія (L1)", lr_model, vec_input)
        display_result(col3, "SVM (Опорні вектори)", svm_model, vec_input)
        
        st.markdown("---")
        st.markdown("###  Внутрішній аналіз")
        with st.expander("Розгорнути технічні деталі обробки NLP", expanded=True):
            st.markdown("**1. Оригінальний текст:**")
            st.info(user_input)
            st.markdown("**2. Результат NLP-конвеєра (Стемінг + видалення стоп-слів):**")
            st.warning(cleaned_text if cleaned_text else "[Текст повністю складався зі стоп-слів або спецсимволів]")
            st.caption("Саме цей очищений текст перетворюється на TF-IDF вектор і передається у моделі. Як бачите, закінчення слів відкинуті (Stemming), а спецсимволи видалені.")