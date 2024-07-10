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

# Model parameters
vocab_size = 50258  # Updated to include the new pad token
n_positions = 1024  # Maximum sequence length
n_embd = 768  # Embedding size
n_layer = 12  # Number of transformer layers
n_head = 12  # Number of attention heads

# Initialize model
model = DRC(vocab_size, n_positions, n_embd, n_layer, n_head)

# Load the GPT-2 tokenizer and add a padding token
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
tokenizer.add_special_tokens({'pad_token': '[PAD]'})

# Resize the model's token embeddings to accommodate the new pad token
model.tok_emb = nn.Embedding(len(tokenizer), n_embd)

# Prepare your data
input_texts = ["Hello, how are you?", "What is your name?", "Tell me a joke."]
output_texts = ["I'm fine, thank you!", "I'm a chatbot!", "Why don't scientists trust atoms? Because they make up everything!"]

# Create datasets and data loaders
max_length = 50  # Maximum length of sequences
train_dataset = CustomDataset(input_texts, output_texts, tokenizer, max_length)
train_loader = DataLoader(train_dataset, batch_size=2, shuffle=True)

# Loss function and optimizer
criterion = nn.CrossEntropyLoss(ignore_index=tokenizer.pad_token_id)
optimizer = AdamW(model.parameters(), lr=1e-4)

# Training loop
epochs = 3
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

# Chatbot function
def chat(input_text):
    model.eval()
    with torch.no_grad():
        input_ids = tokenizer.encode(input_text, return_tensors='pt', padding=True, max_length=max_length, truncation=True)
        output = model(input_ids, max_length)
        predicted_token_id = torch.argmax(output[0, -1, :]).item()
        predicted_token = tokenizer.decode(predicted_token_id)
        return predicted_token

# Simple chat loop
while True:
    user_input = input("You: ")
    if user_input.lower() in ["exit", "quit", "q"]:
        break
    response = chat(user_input)
    print(f"Bot: {response}")
# Model va tokenizerni saqlash
def save_model_and_tokenizer(model, tokenizer, model_path, tokenizer_path):
    # Modelni saqlash
    torch.save(model.state_dict(), model_path)
    
    # Tokenizerni saqlash
    tokenizer.save_pretrained(tokenizer_path)

# Saqlash uchun yo'llar
model_path = 'drc_model.pth'
tokenizer_path = 'gpt2_tokenizer/'

# Model va tokenizerni saqlash
save_model_and_tokenizer(model, tokenizer, model_path, tokenizer_path)
