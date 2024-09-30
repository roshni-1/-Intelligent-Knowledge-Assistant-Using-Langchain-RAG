from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

# Loading the model and tokenizer
model_name = "gpt2"  
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

# Defining the pipeline
pipe = pipeline("text-generation", model=model, tokenizer=tokenizer)

# Setting pad_token to eos_token to avoid the warning
tokenizer.pad_token = tokenizer.eos_token

# Setting up prompt
prompt = "What is the capital of Italy? Respond with just the name of the city:"

# Generating response with minimal tokens
response = pipe(prompt, max_new_tokens=1, num_return_sequences=1, do_sample=False)

# Extracting the answer from response
generated_text = response[0]['generated_text']

# extraction to get just the city name
answer = generated_text.split(":")[-1].strip()  

# Output 
print(answer)
