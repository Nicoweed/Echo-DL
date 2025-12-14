import streamlit as st
import torch
from echo import load_model, load_tokenizer, predict_intent

model = load_model()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

tokenizer = load_tokenizer()

st.write("Hello world")

text = st.text_input("Votre transcript de réunion", key="user_input")
if st.button("Prédire l'émotion", key="predict_button"):
    if text == "":
        st.write("Le texte est vide !")
    else:
        intent, confidence = predict_intent(text, model, tokenizer, device)
        st.write(f"Intention: {intent:.2%}")
        st.write(f"Confiance: {confidence:.2%}")

# You can access the value at any point with:
# st.session_state.name