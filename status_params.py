import json
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer, AutoConfig

# Load and preprocess dataset
dataset = load_dataset('json', data_files={'train': '/Users/abdulhodiy/Desktop/mylms/ai/data/data.json'})

# Initialize the tokenizer
tokenizer = AutoTokenizer.from_pretrained('microsoft/DialoGPT-medium')

# Add a padding token if it doesn't exist
if tokenizer.pad_token is None:
    tokenizer.add_special_tokens({'pad_token': '[PAD]'})

# Create a configuration for the model with 2x the parameters
config = AutoConfig.from_pretrained('microsoft/DialoGPT-medium')
config.n_layer = config.n_layer * 2       # 2x the number of layers
config.n_embd = config.n_embd * 2         # 2x the hidden size
config.n_head = config.n_head * 2         # 2x the number of attention heads

# Initialize the model with the custom configuration
model = AutoModelForCausalLM.from_config(config)

# Resize token embeddings to account for new special tokens
model.resize_token_embeddings(len(tokenizer))

# Tokenize the dataset
def tokenize_function(examples):
    # Texts to tokenize
    texts = examples['text'][0]  # Access the first (and only) list within 'text'
    # Tokenize each text separately and return the result
    return tokenizer(texts, padding='max_length', truncation=True, max_length=512)

# Apply the tokenize function to the dataset
tokenized_datasets = dataset.map(tokenize_function, batched=True, remove_columns=["text"])

# Set up training arguments
training_args = TrainingArguments(
    output_dir='./results',
    num_train_epochs=3,
    per_device_train_batch_size=2,
    per_device_eval_batch_size=2,
    warmup_steps=500,
    weight_decay=0.01,
    logging_dir='./logs',
    logging_steps=10,
    save_steps=10_000,
    save_total_limit=2,
)

# Initialize the Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets['train'],
    eval_dataset=tokenized_datasets['train']
)

# Train the model
trainer.train()

# Save the model and tokenizer
model.save_pretrained('/Users/abdulhodiy/Desktop/mylms/ai/models/gpts/t_device-001')
tokenizer.save_pretrained('/Users/abdulhodiy/Desktop/mylms/ai/models/gpts/t_device-001/tokenizer')
