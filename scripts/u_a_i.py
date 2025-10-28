import streamlit as st
from sentence_transformers import SentenceTransformer

@st.cache_resource
def load_text_embedding_model():
    return SentenceTransformer("NeuML/pubmedbert-base-embeddings")

def embed_text(text):
    model = load_text_embedding_model()
    embedding = model.encode([text])[0]
    return embedding.tolist()
