from transformers import CamembertTokenizer, CamembertForSequenceClassification, CamembertConfig
import torch

_model = None #keep model in cache to avoid multiple loads

def load_model():
    global _model
    if _model is None:
        _model = CamembertForSequenceClassification.from_pretrained(
            './balanced_model/',
            use_safetensors=True
        )
    return _model

def load_tokenizer():
    return CamembertTokenizer.from_pretrained('./balanced_model/')

# TODO: useful?
def get_config():
    return CamembertConfig.from_pretrained("./balanced_model/config.json")

def predict_intent(text, model, tokenizer, device):
    model.eval()

    id2label = model.config.id2label
    
    encoding = tokenizer(
        text,
        add_special_tokens=True,
        max_length=128,
        padding='max_length',
        truncation=True,
        return_tensors='pt'
    )
    
    input_ids = encoding['input_ids'].to(device)
    attention_mask = encoding['attention_mask'].to(device)
    
    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        probs = torch.nn.functional.softmax(outputs.logits, dim=1)
        pred = torch.argmax(probs, dim=1)
        confidence = probs[0][pred].item()
    
    #get id2label
    return id2label[pred.item()], confidence