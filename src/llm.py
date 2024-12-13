"""
This function in this script is only called as a part of LLM generation 
with a given context.
"""

CONTEXT_LENGTH = 3800 # This uses around 9.9GB of GPU memory when highest context length is reached.

YELLOW = "\033[93m"
RESET = "\033[0m"

history = ''

def generate_next_tokens(
    user_input, context, model, tokenizer, streamer, device
):
    global history

    # print('History: ', history)
    print('*' * 50)

    user_input += '\n' + 'Answer the above question based on the following context:\n' + context

    chat = [
        {'role': 'user', 'content': 'Hi'},
        {'role': 'assistant', 'content': 'Hello.'},
        {'role': 'user', 'content': user_input},
    ]

    template = tokenizer.apply_chat_template(
        chat, 
        tokenize=False, 
        add_generation_prompt=True
    )

    # print(template)

    prompt =  '<s>' + history + user_input + '<|end|>\n<|assistant|>\n' if len(history) > 1 else '<s>' + template

    # print('Prompt: ', prompt)
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
        "max_new_tokens": 1024, 
        "streamer": streamer
    }

    outputs = model.generate(**inputs, **generate_kwargs)

    answer = tokenizer.batch_decode(outputs[:, inputs['input_ids'].shape[1]:], skip_special_tokens=True)[0]

    if len(history) > 1:
        history += f"{user_input}<|end|>\n<|assistant|>\n{answer}<|end|>\n<|user|>\n"
    else:
        history = f"{template}{answer}<|end|>\n<|user|>\n"

    # print(f"\n{YELLOW}{answer}{RESET}")

if __name__ == '__main__':
    from transformers import (
        AutoModelForCausalLM, 
        AutoTokenizer, 
        BitsAndBytesConfig
    )

    import torch

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    quant_config = BitsAndBytesConfig(
        load_in_4bit=True
    )

    tokenizer = AutoTokenizer.from_pretrained(
        'microsoft/Phi-3-mini-4k-instruct', trust_remote_code=True
    )
    model = AutoModelForCausalLM.from_pretrained(
        'microsoft/Phi-3-mini-4k-instruct',
        quantization_config=quant_config,
        device_map=device,
        trust_remote_code=True
    )
    generate_next_tokens(
        'Who are you and what can you do?', 
        context='',
        # context='You are a chess player.', 
        model=model,
        tokenizer=tokenizer,
        device=device
    )