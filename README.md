# Інтелектуальна система класифікації спаму (SMS Spam Classifier)

Програмний комплекс для автоматичної бінарної класифікації текстових повідомлень та виявлення спаму/фішингу з використанням методів машинного навчання та обробки природної мови (NLP). 

Розроблено в межах курсової роботи з дисципліни «Алгоритмізація та програмування».

## 🛠 Технологічний стек
* **Мова програмування:** Python 3.10+
* **Machine Learning:** scikit-learn (LinearSVC, LogisticRegression, MultinomialNB)
* **NLP:** NLTK (PorterStemmer, Stopwords), TF-IDF Vectorizer
* **Обробка даних:** pandas, numpy
* **Вебфреймворк:** Streamlit

## 📊 Результати дослідження
У процесі розробки протестовано три алгоритми бінарної класифікації на наборі даних *SMS Spam Collection Dataset*. Найкращу ефективність на високорозмірних розріджених текстових даних продемонстрував метод опорних векторів з лінійним ядром (**LinearSVC**), налаштований з апаратним балансуванням ваг класів (`class_weight='balanced'`):
* **Accuracy (Загальна точність):** 99.01%
* **Precision (Влучність):** 95.71%
* **Recall (Повнота):** 96.40%
* **F1-score:** 0.9606

## 📁 Структура репозиторію
* `app.py` — головний файл вебдодатка Streamlit.
* `coursework2.ipynb` — скрипт розвідувального аналізу (EDA), NLP-конвеєра та навчання моделей.
* `spam.csv` — вихідний набір даних.

## 🚀 Як запустити проєкт локально
### Клонуйте репозиторій:

Bash
git clone [https://github.com/HolovenkoK/spam-classification-system.git](https://github.com/HolovenkoK/spam-classification-system.git)
## Перейдіть до папки проєкту:

Bash
cd spam-classification-system
## Встановіть необхідні бібліотеки:

Bash
pip install streamlit scikit-learn pandas numpy nltk joblib
## Запустіть вебдодаток:

Bash
streamlit run app.py

