import string
import nltk
import numpy as np
import streamlit as st
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Download necessary NLTK data files
nltk.download("punkt", quiet=True)
nltk.download("punkt_tab", quiet=True)
nltk.download("wordnet", quiet=True)
nltk.download("stopwords", quiet=True)

# ----------------------------------------------------
# 1. FAQ Knowledge Base (Questions & Answers)
# ----------------------------------------------------
FAQ_DATA = [
    {
        "question": "What is your return policy?",
        "answer": "You can return any item within 30 days of purchase with the original receipt.",
    },
    {
        "question": "How long does shipping take?",
        "answer": "Standard shipping typically takes 3–5 business days.",
    },
    {
        "question": "Do you offer international shipping?",
        "answer": "Yes, we ship to over 50 countries worldwide with calculated shipping fees.",
    },
    {
        "question": "How can I track my order?",
        "answer": "Once your order is shipped, a tracking number and link are sent to your email.",
    },
    {
        "question": "What payment methods do you accept?",
        "answer": "We accept Visa, MasterCard, PayPal, Apple Pay, and UPI.",
    },
    {
        "question": "How can I contact customer support?",
        "answer": "Reach our support team 24/7 at support@example.com or call +1-800-123-4567.",
    },
]

# ----------------------------------------------------
# 2. NLP Preprocessing (Tokenization, Cleaning, Lemmatization)
# ----------------------------------------------------
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words("english"))


def preprocess(text: str) -> str:
    # Lowercase & remove punctuation
    text = (
        text.lower()
        .translate(str.maketrans("", "", string.punctuation))
        .strip()
    )

    # Tokenize
    tokens = word_tokenize(text)

    # Remove stopwords & apply lemmatization
    cleaned_tokens = [
        lemmatizer.lemmatize(token)
        for token in tokens
        if token not in stop_words
    ]

    return " ".join(cleaned_tokens)


# Preprocess FAQ questions
faq_questions = [item["question"] for item in FAQ_DATA]
processed_faq_questions = [preprocess(q) for q in faq_questions]

# Initialize TF-IDF Vectorizer
vectorizer = TfidfVectorizer()
tfidf_matrix = vectorizer.fit_transform(processed_faq_questions)


# ----------------------------------------------------
# 3. Intent / Similarity Matching Function
# ----------------------------------------------------
def get_best_faq_response(user_query: str, threshold: float = 0.2) -> str:
    clean_query = preprocess(user_query)

    # Edge case: empty query after cleaning
    if not clean_query.strip():
        return "Please ask a complete question so I can help you."

    # Transform user query using fitted TF-IDF
    query_vector = vectorizer.transform([clean_query])

    # Compute cosine similarity against all FAQ questions
    similarities = cosine_similarity(query_vector, tfidf_matrix).flatten()
    best_idx = np.argmax(similarities)
    best_score = similarities[best_idx]

    # Return answer if confidence is above threshold
    if best_score >= threshold:
        return FAQ_DATA[best_idx]["answer"]
    else:
        return "I'm sorry, I couldn't find an answer to that. Please contact support@example.com for further assistance."


# ----------------------------------------------------
# 4. Streamlit Chat Interface
# ----------------------------------------------------
st.set_page_config(page_title="FAQ Chatbot", page_icon="🤖", layout="centered")

st.title("🤖 Customer Support FAQ Chatbot")
st.caption(
    "Ask any question about shipping, payments, returns, or order tracking."
)

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "Hi! How can I help you today?"}
    ]

# Display previous conversation
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])

# User input field
if user_prompt := st.chat_input("Type your question here..."):
    # Display user query
    st.session_state.messages.append({"role": "user", "content": user_prompt})
    with st.chat_message("user"):
        st.write(user_prompt)

    # Get FAQ response
    bot_reply = get_best_faq_response(user_prompt)

    # Display bot response
    st.session_state.messages.append(
        {"role": "assistant", "content": bot_reply}
    )
    with st.chat_message("assistant"):
        st.write(bot_reply)
