import torch
from transformers import GPT2Tokenizer
from models.models.mini import *
# Model va tokenizerni yuklash
def load_model_and_tokenizer(model_class, model_path, tokenizer_path, vocab_size, n_positions, n_embd, n_layer, n_head):
    # Tokenizerni yuklash
    tokenizer = GPT2Tokenizer.from_pretrained(tokenizer_path)
    
    # Modelni yaratish va saqlangan holatini yuklash
    model = model_class(vocab_size, n_positions, n_embd, n_layer, n_head)
    model.load_state_dict(torch.load(model_path))
    model.eval()
    
    return model, tokenizer

# Model parametrlarini sozlash
vocab_size = 50258  # Tokenizerga mos keladigan lug'at hajmi
n_positions = 1024  # Maksimal ketma-ketlik uzunligi
n_embd = 1536  # Embedding o‘lchami
n_layer = 24  # Transformer qatlamlari soni
n_head = 24  # E'tibor boshlarining soni

# Modelni yuklash va tokenizerni qayta o‘rnatish
model_class = DRC  # Modelni yaratish uchun sinf
model, tokenizer = load_model_and_tokenizer(model_class, '/Users/abdulhodiy/Desktop/mylms/gpt2_tokenizer/drc_model.pth', '/Users/abdulhodiy/Desktop/mylms/gpt2_tokenizer', vocab_size, n_positions, n_embd, n_layer, n_head)

# Chatbot funksiyasi
def chat(input_text):
    model.eval()
    with torch.no_grad():
        input_ids = tokenizer.encode(input_text, return_tensors='pt', padding=True, truncation=True, max_length=50)
        output = model(input_ids, max_length=50)
        predicted_token_ids = torch.argmax(output[0, -1, ], dim=-1)
        predicted_text = tokenizer.decode(predicted_token_ids)
        return predicted_text

# Chatbot uchun oddiy muloqot loopi
print("Chatbot tayyor! Savollaringizni yozing, 'exit' yoki 'quit' deb yozing.")
while True:
    user_input = input("Siz: ")
    if user_input.lower() in ["exit", "quit", "q"]:
        print("Chatbot: Xayr!")
        break
    response = chat(user_input)
    print(f"Chatbot: {response}")
