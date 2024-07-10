from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments
from datasets import load_dataset, Dataset

# Model and tokenizer loading
model_name = "microsoft/DialoGPT-small"
model = AutoModelForCausalLM.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Add a padding token if it doesn't exist
if tokenizer.pad_token is None:
    tokenizer.add_special_tokens({'pad_token': '[PAD]'})

# Load the dataset
dataset = load_dataset('json', data_files='/Users/abdulhodiy/Desktop/mylms/ai/data/DragonX.json')

# Inspect the structure of the dataset
print(dataset)

# Define a function to process the dataset
def extract_data(examples):
    instructions = []
    inputs = []
    outputs = []
    print(examples)  # Debug: print the structure of examples
    for item in examples:
        
        instructions.append(item['instruction'])
        inputs.append(item['input'])
        outputs.append(item['output'])
    return {'instruction': instructions, 'input': inputs, 'output': outputs}

# Apply the extraction
dataset = dataset.map(lambda examples: extract_data(examples['data']), batched=True, remove_columns=["data"])

# Check the dataset structure after extraction
print(dataset)

# Flatten the dataset and convert it into a Dataset object
flat_data = {
    'instruction': sum(dataset['train']['instruction'], []),
    'input': sum(dataset['train']['input'], []),
    'output': sum(dataset['train']['output'], [])
}

flat_dataset = Dataset.from_dict(flat_data)

# Split dataset into train and test
split_dataset = flat_dataset.train_test_split(test_size=0.2)

# Tokenization function
def tokenize_function(examples):
    combined_texts = [instruction + " " + input_text + tokenizer.eos_token for instruction, input_text in zip(examples['instruction'], examples['input'])]
    return tokenizer((combined_texts), padding='max_length', truncation=True, max_length=512)

# Tokenize the dataset
tokenized_datasets = split_dataset.map(tokenize_function, batched=True, remove_columns=["instruction", "input", "output"])

# Training arguments
training_args = TrainingArguments(
    output_dir="./results",
    evaluation_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=2,
    per_device_eval_batch_size=2,
    num_train_epochs=3,
    weight_decay=0.01,
)

# Create the Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets['train'],
    eval_dataset=tokenized_datasets['test']
)

# Train the model
trainer.train()

# Save the model and tokenizer
model.save_pretrained("/Users/abdulhodiy/Desktop/mylms/models/dev-001")
tokenizer.save_pretrained("/Users/abdulhodiy/Desktop/mylms/models/dev-001")
