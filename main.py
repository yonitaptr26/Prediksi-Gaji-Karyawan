from fastapi import FastAPI
from pydantic import BaseModel
import torch
import torch.nn as nn
import numpy as np
from transformers import BertTokenizer, BertModel
import joblib

# Inisialisasi aplikasi FastAPI
app = FastAPI()

# Memeriksa apakah GPU tersedia, jika tidak gunakan CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Definisi arsitektur model BERT + LSTM
class BertLSTMModel(nn.Module):
    def __init__(self, bert_model_name='bert-base-uncased', lstm_hidden_size=128, lstm_layers=1):
        super(BertLSTMModel, self).__init__()
        self.bert = BertModel.from_pretrained(bert_model_name)  # Model Bert
        self.lstm = nn.LSTM(
            input_size=self.bert.config.hidden_size,
            hidden_size=lstm_hidden_size,
            num_layers=lstm_layers,
            batch_first=True,
            bidirectional=True
        )
        # Linear layer untuk menghasilkan output prediksi nilai tunggal
        self.fc = nn.Linear(lstm_hidden_size * 2, 1)

    def forward(self, input_ids, attention_mask):
        bert_output = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        cls_embeddings = bert_output.last_hidden_state[:, 0, :]
        lstm_output, _ = self.lstm(cls_embeddings.unsqueeze(1))
        output = self.fc(lstm_output[:, -1, :])
        return output
    
# Memuat scaler untuk denormalisasi hasil prediksi
try:
    scaler = joblib.load("scaler_sw.pkl")
    print("Scaler berhasil dimuat!")
except Exception as e:
    print(f"Error memuat scaler: {e}")

# Memuat tokenizer
try:
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    print("Tokenizer berhasil dimuat!")
except Exception as e:
    print(f"Error memuat tokenizer: {e}")


# Memuat model BERT + LSTM dari file
try:
    model = BertLSTMModel()  # Instansiasi model
    state_dict = torch.load('bert_lstm_model_full_baru.pth', map_location=device)
    model.load_state_dict(state_dict)
    model.to(device)  # Memindahkan model ke device
    model.eval()  # Atur model ke mode evaluasi
    print("Model berhasil dimuat!")
except Exception as e:
    print(f"Error memuat model: {e}")

def predict_new_data(model, text_input, tokenizer, scaler, max_len=128):
    encoding = tokenizer(text_input, truncation=True, max_length=max_len, padding='max_length', return_tensors='pt')
    input_ids = encoding['input_ids'].to(device)
    attention_mask = encoding['attention_mask'].to(device)

    model.to(device)

    with torch.no_grad():
        output = model(input_ids, attention_mask)
        pred = output.squeeze().cpu().numpy()
        print(f"Output model (sebelum denormalisasi): {pred}")

    # Pastikan prediksi masuk akal
    if np.any(pred < 0) or np.any(pred > 1):
        print("Prediksi di luar rentang yang diharapkan!")

    # Denormalisasi hasil prediksi
    try:
        pred_denorm = scaler.inverse_transform(np.array(pred).reshape(-1, 1)).squeeze()
        print(f"Output model (setelah denormalisasi): {pred_denorm}")
    except Exception as e:
        print(f"Error saat denormalisasi: {e}")
        pred_denorm = pred  # Kembalikan hasil prediksi mentah jika error

    return pred_denorm


# Schema masukan untuk API menggunakan Pydantic
class PredictionRequest(BaseModel):
    text: str  # Masukan berupa teks

# Endpoint API untuk melakukan prediksi
@app.post("/predict")
def predict(request: PredictionRequest):
    try:
        text_input = request.text
        pred = predict_new_data(model, text_input, tokenizer, scaler)
        return {"predicted_value": round(float(pred), 2)}
    except Exception as e:
        return {"error": str(e)}
