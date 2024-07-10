import json
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import GPT2Tokenizer, GPT2LMHeadModel, AdamW

# Ma'lumotlar tayyorlash
class DialogDataset(Dataset):
    def __init__(self, file_path, tokenizer, max_length=1024):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.dialogs = []
        
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                dialog = json.loads(line)
                self.dialogs.append(dialog['text'])
    
    def __len__(self):
        return len(self.dialogs)
    
    def __getitem__(self, idx):
        dialog = self.dialogs[idx]
        encodings = self.tokenizer(dialog, return_tensors='pt', truncation=True, padding='max_length', max_length=self.max_length)
        input_ids = encodings['input_ids'].squeeze()
        attention_mask = encodings['attention_mask'].squeeze()
        
        return input_ids, attention_mask

# Modelni yuklash
def load_model(model_path, model_name='gpt2'):
    tokenizer = GPT2Tokenizer.from_pretrained(model_name)
    model = GPT2LMHeadModel.from_pretrained(model_name)
    model.load_state_dict(torch.load(model_path))
    return model, tokenizer

# Modelni o'qitish
def train(model, dataloader, optimizer, num_epochs=1):
    model.train()
    for epoch in range(num_epochs):
        for batch in dataloader:
            input_ids, attention_mask = batch
            outputs = model(input_ids, attention_mask=attention_mask, labels=input_ids)
            loss = outputs.loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            print(f"Epoch: {epoch}, Loss: {loss.item()}")

# Modelni sinash
def generate_response(model, tokenizer, prompt, max_length=50):
    input_ids = tokenizer.encode(prompt, return_tensors='pt')
    with torch.no_grad():
        output = model.generate(input_ids, max_length=max_length, num_return_sequences=1)
    response = tokenizer.decode(output[0], skip_special_tokens=True)
    return response

# Asosiy dastur
if __name__ == "__main__":
    # Ma'lumotlar yo'li
    file_path = 'dialogs.jsonl'  # Bu yerda ma'lumotlaringiz fayli yo'lini ko'rsating

    # Model va tokenizerni yuklash
    model_path = 'gpt2_177mb_model.pth'  # Bu yerda o'zingiz saqlagan model fayli yo'lini ko'rsating
    model_name = 'gpt2'
    model, tokenizer = load_model(model_path, model_name=model_name)

    # Dataset va DataLoader
    dataset = DialogDataset(file_path, tokenizer)
    dataloader = DataLoader(dataset, batch_size=2, shuffle=True)

    # Optimizer va yo'qotish funksiyasi
    optimizer = AdamW(model.parameters(), lr=5e-5)

    # Modelni o'qitish
    num_epochs = 3
    train(model, dataloader, optimizer, num_epochs=num_epochs)

    # Modelni qayta saqlash
    new_model_path = 'gpt2_dialog_model.pth'
    torch.save(model.state_dict(), new_model_path)
    print(f'Model saved at {new_model_path}')

    # Sinash
    model.load_state_dict(torch.load(new_model_path))
    model.eval()
    
    prompt = "Salom, bugun nima qilamiz?"
    response = generate_response(model, tokenizer, prompt)
    print(f"Prompt: {prompt}")
    print(f"Response: {response}")
