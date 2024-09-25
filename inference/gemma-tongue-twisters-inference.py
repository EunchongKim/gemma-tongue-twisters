from transformers import AutoTokenizer, AutoModelForCausalLM

model_name = "english-tt-fine-tuned-gemma-2-2b"
model = AutoModelForCausalLM.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

input_text = "Generate tongue twisters about key words: a cup of coffee"
input_ids = tokenizer.encode(input_text, return_tensors='pt')

output = model.generate(
    input_ids,
    max_length=100,
    num_return_sequences=1,
    no_repeat_ngram_size=2,
    repetition_penalty=2.0,
    top_k=50,
    top_p=0.95,
    temperature=0.7,
    do_sample=True,
    eos_token_id=tokenizer.eos_token_id,
)

generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
print(generated_text)
