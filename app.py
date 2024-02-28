import gradio as gr
import torch
import torch.nn as nn
import torch.nn.functional as F
from tensorflow.keras.preprocessing.text import tokenizer_from_json
from tensorflow.keras.preprocessing.sequence import pad_sequences
import json
 

# 加載Tokenizer配置
with open('tokenizer_config.json') as f:
    tokenizer_data = f.read()
tokenizer = tokenizer_from_json(tokenizer_data)

class TextLSTM(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, num_layers, num_classes):
        super(TextLSTM, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, num_classes)
        
    def forward(self, x):
        x = self.embedding(x)
        x, _ = self.lstm(x)
        x = x[:, -1, :]
        x = self.fc(x)
        return x

hidden_dim = 128
num_layers = 2
vocab_size = 10000
embed_dim = 100
num_classes = 6
model = TextLSTM(vocab_size, embed_dim, hidden_dim, num_layers, num_classes)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.load_state_dict(torch.load("lstm.pt", map_location=device))
model.to(device)
model.eval()

max_seq_length = 171  

def prodata(sentence):
    test_tokenized = tokenizer.texts_to_sequences([sentence])
    test_padded = pad_sequences(test_tokenized, maxlen=max_seq_length)
    test_padded_tensor = torch.tensor(test_padded, dtype=torch.long).to(device)
    return test_padded_tensor

def greet(text):
    print(text)
    ptext = prodata(text)
    with torch.no_grad():  
        prediction = model(ptext)
        predicted_class = torch.argmax(prediction, dim=1) 
        result = predicted_class.item() 
        if result == 0:
            res = "Sadness"
        elif result == 1:
            res = "Joy"
        elif result == 2:
            res = "Love"
        elif result == 3:
            res = "Anger"
        elif result == 4:
            res = "Fear"
        elif result == 5:
            res = "Surprise"    
        print(predicted_class)
    return res,f"{res}.png"

demo = gr.Interface(
    fn=greet,
    inputs=["text"],
    outputs=["text","image"]
)

demo.launch()
