import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from transformers import GPT2Tokenizer, AdamW

class CustomDataset(Dataset):
    def __init__(self, input_texts, output_texts, tokenizer, max_length):
        self.input_texts = input_texts
        self.output_texts = output_texts
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.input_texts)
    
    def __getitem__(self, idx):
        input_text = self.input_texts[idx]
        output_text = self.output_texts[idx]
        
        input_ids = self.tokenizer.encode(input_text, max_length=self.max_length, truncation=True, padding='max_length', return_tensors='pt').squeeze()
        output_ids = self.tokenizer.encode(output_text, max_length=self.max_length, truncation=True, padding='max_length', return_tensors='pt').squeeze()
        
        return {
            'input_ids': input_ids,
            'labels': output_ids
        }

class DRC(nn.Module):
    def __init__(self, vocab_size, n_positions, n_embd, n_layer, n_head):
        super(DRC, self).__init__()
        
        self.tok_emb = nn.Embedding(vocab_size, n_embd)
        self.pos_emb = nn.Embedding(n_positions, n_embd)
        
        self.blocks = nn.ModuleList([
            nn.TransformerEncoderLayer(d_model=n_embd, nhead=n_head) for _ in range(n_layer)
        ])
        
        self.ln_f = nn.LayerNorm(n_embd)
        self.head = nn.Linear(n_embd, vocab_size, bias=False)
        
        self.n_positions = n_positions

    def forward(self, input_ids, max_length):
        batch_size, seq_length = input_ids.size()
        
        # Token and position embeddings
        pos_ids = torch.arange(0, seq_length, dtype=torch.long, device=input_ids.device)
        pos_ids = pos_ids.unsqueeze(0).expand(batch_size, seq_length)
        
        tok_emb = self.tok_emb(input_ids)
        pos_emb = self.pos_emb(pos_ids)
        x = tok_emb + pos_emb
        
        # Transformer blocks
        for block in self.blocks:
            x = block(x)
        
        x = self.ln_f(x)
        logits = self.head(x)
        
        return logits[:, :max_length, :]

# Model parametrlarini sozlash
vocab_size = 50258  # Tokenizerga mos keladigan lug'at hajmi
n_positions = 1024  # Maksimal ketma-ketlik uzunligi
n_embd = 1536  # Embedding o‘lchami
n_layer = 24  # Transformer qatlamlari soni
n_head = 24  # E'tibor boshlarining soni

# Modelni yaratish
model = DRC(vocab_size, n_positions, n_embd, n_layer, n_head)

# Tokenizerni yuklash va yangi maxsus token qo‘shish
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
tokenizer.add_special_tokens({'pad_token': '[PAD]'})

# Modelni tokenlarni saqlash uchun qayta o‘rganish
model.tok_emb = nn.Embedding(len(tokenizer), n_embd)

# O‘qitish uchun ma'lumotlarni tayyorlash
input_texts = ["Hello, how are you?", "What is your name?", "Tell me a joke."]
output_texts = ["I'm fine, thank you!", "I'm a chatbot!", "Why don't scientists trust atoms? Because they make up everything!"]
max_length = 500  # Maksimal ketma-ketlik uzunligi

# Dataset va dataloader yaratish
train_dataset = CustomDataset(input_texts, output_texts, tokenizer, max_length)
train_loader = DataLoader(train_dataset, batch_size=2, shuffle=True)

# O‘qitish uchun parametrlarni sozlash
epochs = 5  # Yangi o‘qitish uchun epoxalar soni
learning_rate = 10e-4  # Yangi o‘qitish uchun o‘rganish tezligi
criterion = nn.CrossEntropyLoss(ignore_index=tokenizer.pad_token_id)
optimizer = AdamW(model.parameters(), lr=learning_rate)

# Modelni o‘qitish
def train_model(model, train_loader, epochs, criterion, optimizer, max_length):
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for batch in train_loader:
            input_ids = batch['input_ids']
            labels = batch['labels']
            
            optimizer.zero_grad()
            outputs = model(input_ids, max_length)
            loss = criterion(outputs.view(-1, len(tokenizer)), labels.view(-1))
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        print(f'Epoch {epoch + 1}, Loss: {total_loss / len(train_loader)}')

# O‘qitish jarayonini boshlash
train_model(model, train_loader, epochs, criterion, optimizer, max_length)

# Modelni saqlash
torch.save(model.state_dict(), 'gpt2_tokenizer/drc_model.pth')
tokenizer.save_pretrained('gpt2_tokenizer/')