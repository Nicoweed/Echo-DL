import gradio as gr
import torch
from echo import load_model, load_tokenizer, predict_intent

model = load_model()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

tokenizer = load_tokenizer()

def predict_emotion(transcript: str):
    if not transcript:
        return "Texte vide !", "Texte vide !"
    intent, confidence = predict_intent(transcript, model, tokenizer, device)
    return intent, f"{confidence:.2%}"

demo = gr.Interface(
    fn=predict_emotion,
    inputs=[gr.Textbox(label="Transcript de réunion")],
    outputs=[gr.Textbox(label="Intention"), gr.Textbox(label="Indice de confiance de la prédiction")],
    title="Émotion de réunion",
    description="Postez ici le transcript de votre réunion, pour prédire l'émotion globale qui en ressort.",
    api_name="echo",
    flagging_mode="never"
)

demo.launch(share=True)