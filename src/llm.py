from transformers import (
    AutoModelForCausalLM, 
    AutoTokenizer, 
    BitsAndBytesConfig,
    AutoProcessor
)

device = 'cuda'

quant_config = BitsAndBytesConfig(
    load_in_4bit=True
)

tokenizer = AutoTokenizer.from_pretrained('microsoft/Phi-3-mini-4k-instruct')
model = AutoModelForCausalLM.from_pretrained(
    'microsoft/Phi-3-mini-4k-instruct',
    quantization_config=quant_config,
    device_map=device
)
processor = AutoProcessor.from_pretrained('microsoft/Phi-3-mini-4k-instruct')

CONTEXT_LENGTH = 3800 # This uses around 9.9GB of GPU memory when highest context length is reached.

def generate_next_tokens(user_input, context, history):
    print('History: ', history)
    print('*' * 50)

    chat = [
        {'role': 'user', 'content': 'Hi'},
        {'role': 'assistant', 'content': 'Hello.'},
        {'role': 'user', 'content': user_input + '\n' + 'Answer the above question based on the following context:\n' + context},
    ]

    template = tokenizer.apply_chat_template(
        chat, 
        tokenize=False, 
        add_generation_prompt=True
    )

    print(template)

    if len(history) <= 1:
        prompt = '<s>' + template
    else:
        prompt = '<s>'
        for history_list in history:
            prompt += f"<|user|>\n{history_list[0]}<|end|>\n<|assistant|>\n{history_list[1]}<|end|>\n"
        prompt += f"<|user|>\n{user_input}<|end|>\n<|assistant|>\n"

    print('Prompt: ', prompt)
    print('*' * 50)

    inputs = tokenizer(prompt, return_tensors='pt').to(device)
    input_ids, attention_mask = inputs.input_ids, inputs.attention_mask

    # A way to manage context length + memory for best results.
    print('Global context length till now: ', input_ids.shape[1])
    if input_ids.shape[1] > CONTEXT_LENGTH:
        input_ids = input_ids[:, -CONTEXT_LENGTH:]
        attention_mask = attention_mask[:, -CONTEXT_LENGTH:]

    print('-' * 100)

    generate_kwargs = {
        "max_new_tokens": 1024
    }

    outputs = model.generate(**inputs, **generate_kwargs)

    answer = tokenizer.batch_decode(outputs[:, inputs['input_ids'].shape[1]:], skip_special_tokens=True)

    print(answer[0])

if __name__ == '__main__':
    generate_next_tokens('Who are you and what can you do?', context='', history='')