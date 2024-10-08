---
license: gemma
language:
- en
base_model:
- google/gemma-2-2b-it
pipeline_tag: text-generation
---

# **Model Demo**

You can try the **Tongue Twister Generator** in the [Hugging Face Space - Tongue Twister Generator with TTS](https://huggingface.co/spaces/ec-kim/gemma2-tongue-twister-generator).

<br>

# **Model Card: gemma2-tongue-twister-generator**

## **Model Details**

- **Model Name**: gemma2-tongue-twister-generator
- **Model Type**: Causal Language Model (Fine-tuned with LoRA)
- **Architecture**: Gemma 2 2B (based on Transformer architecture)
- **Base Model**: `google/gemma-2-2b-it`
- **Training Method**: Fine-tuned using Low-Rank Adaptation (LoRA)
- **Languages**: English
- **License**: Gemma License

## **Model Description**

The **gemma2-tongue-twister-generator** is a fine-tuned version of the **Gemma 2 2B** model. It is specifically trained to generate creative and challenging tongue twisters based on given keywords or prompts. The model can create intricate wordplays with rhythmic patterns, making it ideal for educational and entertainment purposes, such as helping individuals practice speech articulation and fluency.

### **Use Cases**

- **Tongue Twister Generation**: Generate custom tongue twisters based on a set of provided keywords.
- **Speech Practice**: Enhance speaking and articulation skills by practicing with challenging wordplay.
- **Entertainment**: Generate engaging and fun tongue twisters for social games or language challenges.

## **Training Procedure**

- **Dataset**: The model was fine-tuned on a dataset of 1,900 English tongue twisters, where each example contains a prompt (keywords or questions) and an answer (a tongue twister). The dataset is based on examples found in the paper "TongueTwister Games: A New Benchmark and Dataset for Word Play Generation" published at ACL 2023.
  - **Example Prompt**: `"Generate tongue twisters about key words: sea seashell"`
  - **Example Answer**: `"She sells seashells by the seashore."`
  
- **Training Method**: The model was fine-tuned using **LoRA (Low-Rank Adaptation)** without quantization to maintain high precision and model quality.
- **Parameters**:
  - **Learning Rate**: 3e-5
  - **Epochs**: 3
  - **Batch Size**: 8
  - **Gradient Accumulation**: 4 steps
  - **Scheduler**: Cosine Learning Rate Scheduler
  - **Warmup Steps**: 100
  - **Evaluation**: Early stopping was based on validation performance.

## **Model Performance**

- **Metrics**: 
  - **Human Evaluation**: Qualitative evaluation indicates that the model generates coherent and challenging tongue twisters based on keywords.
  - **No Extra Keywords**: The model does not generate keywords not present in the prompt, ensuring output relevance to the given input.

### **Limitations**

- **Rhythmic Patterns**: The model may not perfectly generate complex syllable stress or rhythmic complexity in all tongue twisters, which might affect the challenge level or entertainment factor.
- **Language**: This model is designed to work only with English tongue twisters.

### **Potential Biases**

The model was trained on tongue twisters, which are specific to a subset of the English language and may not capture cultural or linguistic diversity. The dataset could contain patterns reflective of Western language games, which might affect the diversity of the generated text.

## Inference Environment

The following versions were used during the inference of the model:

- Python: 3.10.12
- transformers: 4.44.2
- torch: 2.4.1+cu121
- peft: 0.13.0

### Installation

You can install these dependencies using the following command:

```bash
pip install transformers peft torch
```

## **How to Use the Model**

You can use this model for inference using the Hugging Face `transformers` library:

```python
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

# Load the model and tokenizer
model_name = "ec-kim/gemma2-tongue-twister-generator"
base_model = AutoModelForCausalLM.from_pretrained("google/gemma-2-2b-it")
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = PeftModel.from_pretrained(base_model, model_name)

# Generate tongue twister
question = "Generate tongue twisters about key words: cinnamon synonym"
inputs = tokenizer(question, return_tensors="pt")

outputs = model.generate(
    inputs.input_ids,
    max_length=100,
    num_return_sequences=1,
    no_repeat_ngram_size=2,
    repetition_penalty=2.0,
    top_k=50,
    top_p=0.95,
    temperature=0.7,
    eos_token_id=tokenizer.eos_token_id,
    do_sample=True
)

response = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(response)
```

## **Sample Output**

**Prompt**: "Generate tongue twisters about key words: weeping winnie"  
**Output**: "Generate tongue twisters about key words: weeping winnie wore wimpier wings when she wanted to weep."

## **Intended Use**

- **Primary Use Case**: Generating tongue twisters for entertainment, speech practice, or educational purposes.
- **Use with Caution**: Not intended for generating factual or coherent long-form text. The model excels in generating short, playful text but may not produce factual information.

## **Future Work**

- Extend the dataset with more diverse and creative tongue twisters.
- Explore multi-lingual training for generating tongue twisters in other languages.
- Fine-tune on datasets with specific constraints like syllable stress and rhythmic complexity.

## **Contributions**

This model was fine-tuned by using the Hugging Face `transformers` and `peft` libraries. The training dataset is based on the paper ["TongueTwister Games: A New Benchmark and Dataset for Word Play Generation"](https://aclanthology.org/2023.acl-short.51.pdf) from ACL 2023. For any issues, please contact [here](https://github.com/EunchongKim)
