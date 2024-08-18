import gradio as gr
import json
import os
import threading

from transformers import (
    AutoModelForCausalLM, 
    AutoTokenizer, 
    BitsAndBytesConfig,
    TextIteratorStreamer
)
from search import load_documents, main, load_embedding_model
from create_embeddings import load_and_preprocess_text_files

device = 'cuda'

quant_config = BitsAndBytesConfig(
    load_in_4bit=True
)

model_id = 'microsoft/Phi-3-mini-128k-instruct'
# model_id = 'microsoft/Phi-3-small-128k-instruct'

tokenizer = AutoTokenizer.from_pretrained(
    model_id, trust_remote_code=True
)
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    quantization_config=quant_config,
    device_map=device,
    trust_remote_code=True
)

embedding_model = load_embedding_model('all-MiniLM-L6-v2')

streamer = TextIteratorStreamer(
    tokenizer, skip_prompt=True, skip_special_tokens=True
)

CONTEXT_LENGTH = 3800 # This uses around 9.9GB of GPU memory when highest context length is reached.

documents = None
results = []

def generate_next_tokens(user_input, history):
    global documents
    global results

    # print(f"User Input: ", user_input)
    # print('History: ', history)
    print('*' * 50)

    # The way we are managing uploaded file and history here:
    # When the user first uploads the file, the entire content gets
    # loaded into the prompt for that particular chat turn.
    # When the next turn comes, right now, we are using the `history`
    # list from Gradio to load the history again, however, that only contains
    # the file path. So, we cannot exactly get the content of the file in the
    # next turn. However, the model may remember the context from its own
    # reply and user's query. This approach saves a lot of memory as well.
    if len(user_input['files']) != 0:
        results = load_and_preprocess_text_files(
            results,
            user_input['files'][0],
            add_file_content=True,
            chunk_size=128,
            overlap=16,
            model=embedding_model
        )
        embedded_docs = [result for result in results]
        # Save documents with embeddings to a JSON file.
        with open(os.path.join('..', 'data', 'temp.json'), 'w') as f:
            json.dump(embedded_docs, f)
        
        documents = load_documents(os.path.join('..', 'data', 'temp.json'))
    
    final_input = ''
    user_text = user_input['text']

    # Get the context.
    context_list = main(
        documents, 
        user_text, 
        embedding_model,
        extract_content=True,
        topk=3
    )
    context = '\n\n'.join(context_list)
    final_input += user_text + '\n' + 'Answer the above question based on the following context:\n' + context

    chat = [
        {'role': 'user', 'content': 'Hi'},
        {'role': 'assistant', 'content': 'Hello.'},
        {'role': 'user', 'content': final_input},
    ]

    template = tokenizer.apply_chat_template(
        chat, 
        tokenize=False, 
        add_generation_prompt=True
    )

    # Loading from Gradio's `history` list. If a file was uploaded in the 
    # previous turn, only the file path remains in the history and not the 
    # content. Good for saving memory (context) but bad for detailed querying.
    if len(history) == 0:
        prompt = '<s>' + template
    else:
        prompt = '<s>'
        for history_list in history:
            prompt += f"<|user|>\n{history_list[0]}<|end|>\n<|assistant|>\n{history_list[1]}<|end|>\n"
        prompt += f"<|user|>\n{final_input}<|end|>\n<|assistant|>\n"

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

    generate_kwargs = dict(
        {'input_ids': input_ids.to(device), 'attention_mask': attention_mask.to(device)},
        streamer=streamer,
        max_new_tokens=1024,
    )

    thread = threading.Thread(
        target=model.generate, 
        kwargs=generate_kwargs
    )

    thread.start()

    outputs = []
    for new_token in streamer:
        outputs.append(new_token)
        final_output = ''.join(outputs)

        yield final_output


input_text = gr.Textbox(lines=5, label='Prompt')
output_text = gr.Textbox(label='Generated Text')

iface = gr.ChatInterface(
    fn=generate_next_tokens, 
    multimodal=True,
    title='File Chat'
)

iface.launch()
