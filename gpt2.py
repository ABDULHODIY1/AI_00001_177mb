from transformers import GPT2Tokenizer, TFGPT2Model
tokenizer = GPT2Tokenizer.from_pretrained('/Users/abdulhodiy/Desktop/mylms/gpt2_tokenizer')
model = TFGPT2Model.from_pretrained('/Users/abdulhodiy/Desktop/mylms/gpt2_tokenizer')
text = "hello"
encoded_input = tokenizer(text, return_tensors='tf')
output = model(encoded_input)
