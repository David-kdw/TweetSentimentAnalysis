import gradio as gr
import torch
import torch.nn as nn
from transformers import BertModel, AutoTokenizer


class CustomBert(nn.Module):
    def __init__(self, model_name_or_path="bert-base-uncased", n_classes=5):
        super(CustomBert, self).__init__()
        self.bert_pretrained = BertModel.from_pretrained(model_name_or_path)
        self.classifier = nn.Linear(self.bert_pretrained.config.hidden_size, n_classes)

    def forward(self, input_ids, attention_mask):
        x = self.bert_pretrained(input_ids=input_ids, attention_mask=attention_mask)
        x = self.classifier(x.pooler_output)
        return x


# Charger le modèle et définir le dispositif
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = CustomBert()
model.load_state_dict(torch.load("/content/drive/MyDrive/my_custom_bert.pth", map_location=device))
model.to(device)
model.eval()  # Mettre le modèle en mode évaluation

# Liste des classes selon les labels de ton dataset
classes = ["Extremely Negative", "Negative", "Neutral", "Positive", "Extremely Positive"]

def classifier_fn(text: str):
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    inputs = tokenizer(
        text,
        padding="max_length",
        max_length=250,
        truncation=True,
        return_tensors="pt"
    )
    inputs = {key: value.to(device) for key, value in inputs.items()}  # Assurer que les entrées sont sur le bon dispositif
    output = model(input_ids=inputs["input_ids"], attention_mask=inputs["attention_mask"])
    _, pred = output.max(1)

    return classes[pred.item()]


demo = gr.Interface(
    fn=classifier_fn,
    inputs=["text"],
    outputs=["text"],
)

demo.launch()
