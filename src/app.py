import gradio as gr
import json
import os
import threading
import argparse
import cv2

from transformers import (
    AutoModelForCausalLM, 
    AutoTokenizer, 
    BitsAndBytesConfig,
    TextIteratorStreamer,
    AutoProcessor
)
from search import load_documents, load_embedding_model
from search import main as search_main
from create_embeddings import load_and_preprocess_text_files
from PIL import Image

parser = argparse.ArgumentParser()
parser.add_argument(
    '--share',
    help='create a shareable gradio link',
    action='store_true'
)
parser.add_argument(
    '--json',
    help='optional json file path if you have already embedded hundreds of files, \
          helps because you do not need to upload the json file to the ui, \
          if json file path is provided, it takes precedence compared to \
          uploading any other file',
    default=None
)
args = parser.parse_args()

device = 'cuda'

model_id = None
model = None
tokenizer = None
streamer = None
processor = None

def load_llm(chat_model_id):
    global model
    global tokenizer
    global streamer
    global processor

    gr.Info(f"Loading model: {chat_model_id}")

    quant_config = BitsAndBytesConfig(
        load_in_4bit=True
    )

    processor = AutoProcessor.from_pretrained(
        chat_model_id, 
        trust_remote_code=True, 
        num_crops=4
    ) 
    tokenizer = AutoTokenizer.from_pretrained(
        chat_model_id, trust_remote_code=True
    )
    
    model = AutoModelForCausalLM.from_pretrained(
        chat_model_id,
        quantization_config=quant_config,
        device_map=device,
        trust_remote_code=True,
        _attn_implementation='eager'
    )

    streamer = TextIteratorStreamer(
        tokenizer, skip_prompt=True, skip_special_tokens=True
    )

def load_and_preprocess_images(image_path):
    image = Image.open(image_path)
    return image

def load_and_process_videos(file_path, images, placeholder, counter):
    cap = cv2.VideoCapture(file_path)
    length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    for i in range(length):
        counter += 1
        cap.set(cv2.CAP_PROP_POS_FRAMES, i)
        ret, frame = cap.read()
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        images.append(Image.fromarray(image))
        placeholder += f"<|image_{counter}|>\n"
    return images, placeholder, counter

embedding_model = load_embedding_model('all-MiniLM-L6-v2')


CONTEXT_LENGTH = 3800 # This uses around 9.9GB of GPU memory when highest context length is reached.
GLOBAL_IMAGE_LIST = []

documents = None
results = []

def generate_next_tokens(user_input, history, chat_model_id):
    global documents
    global results
    global model_id

    # If a new PDF file is uploaded, create embeddings, store in `temp.json`
    # and load the embedding file.
    images = []
    placeholder = ''

    # If a JSON file path is passed in the arguments.
    if args.json is not None:
        print('Loading JSON')
        documents = load_documents(os.path.join(args.json))
    # Else load whatever file is uploaded.
    else:
        if len(user_input['files']) != 0:
            for file_path in user_input['files']:
                counter = 0
                if file_path.endswith('.mp4'):
                    GLOBAL_IMAGE_LIST.append(file_path)
                    images, placeholder, counter = load_and_process_videos(
                        file_path, images, placeholder, counter
                    )
                elif file_path.endswith('.jpg') or \
                    file_path.endswith('.png') or \
                    file_path.endswith('.jpeg'):
                    counter += 1
                    GLOBAL_IMAGE_LIST.append(file_path)
                    image = load_and_preprocess_images(
                        file_path
                    )
                    images.append(image)
                    placeholder += f"<|image_{counter}|>\n"
                elif file_path.endswith('.pdf') or \
                    file_path.endswith('.txt'):
                    results = load_and_preprocess_text_files(
                        results,
                        file_path,
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
                elif file_path.endswith('.json'): # Load an indexed file directly.
                    documents = load_documents(file_path)
        
    if chat_model_id == 'microsoft/Phi-3.5-vision-instruct' and len(images) == 0:
        counter = 0
        for i, file_path in enumerate(GLOBAL_IMAGE_LIST):
            if file_path.endswith('.mp4'):
                images, placeholder, counter = load_and_process_videos(
                    file_path, images, placeholder, counter
                )
            else:
                counter += 1
                image = load_and_preprocess_images(
                    file_path
                )
                images.append(image)
                placeholder += f"<|image_{counter}|>\n"

    if chat_model_id == 'microsoft/Phi-3.5-vision-instruct' and len(images) == 0:
        gr.Warning(
            'Please upload an image to use the Vision model. '
            'Or select one of the text models from the advanced '
            'dropdown to chat with PDFs and other text files.',
            duration=20
        )
    if chat_model_id != 'microsoft/Phi-3.5-vision-instruct' and len(images) != 0:
        gr.Warning(
            'You are using a text model. '
            'Please select a Vision model from the advanced '
            'dropdown to chat with images.',
            duration=20
        )

    if chat_model_id != model_id:
        load_llm(chat_model_id)
        model_id = chat_model_id

    # print(f"User Input: ", user_input)
    # print('History: ', history)
    print('*' * 50)

    
    final_input = ''
    user_text = user_input['text']


    if len(images) != 0:
        final_input += placeholder+user_text
        chat = [
            {'role': 'user', 'content': placeholder+user_text},
        ]
        template = processor.tokenizer.apply_chat_template(
            chat,
            tokenize=False,
            add_generation_prompt=True
        )
    else:
        # Get the context.
        if documents == None: # Bypass if no document is uploaded.
            context_list = []
        else:
            context_list = search_main(
                documents, 
                user_text, 
                embedding_model,
                extract_content=True,
                topk=3
            )
        context = '\n\n'.join(context_list)
        final_input += user_text + '\n' + 'Answer the above question based on the following context. If the context is empty, then just chat normally:\n' + context
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
    if len(history) == 0 and len(images) != 0:
        prompt = '<s>' + template
    else:
        prompt = '<s>'
        for history_list in history:
            prompt += f"<|user|>\n{history_list[0]}<|end|>\n<|assistant|>\n{history_list[1]}<|end|>\n"
        prompt += f"<|user|>\n{final_input}<|end|>\n<|assistant|>\n"

    print('Prompt: ', prompt)
    print('*' * 50)

    if len(images) != 0:
        inputs = processor(prompt, images, return_tensors='pt').to(device)
        generate_kwargs = dict(
            **inputs,
            eos_token_id=processor.tokenizer.eos_token_id, 
            streamer=streamer,
            max_new_tokens=1024,
        )   
    else:
        inputs = tokenizer(prompt, return_tensors='pt').to(device)
        input_ids, attention_mask = inputs.input_ids, inputs.attention_mask
        generate_kwargs = dict(
            {'input_ids': input_ids.to(device), 'attention_mask': attention_mask.to(device)},
            streamer=streamer,
            max_new_tokens=1024,
        )   

        # A way to manage context length + memory for best results.
        print('Global context length till now: ', input_ids.shape[1])
        if input_ids.shape[1] > CONTEXT_LENGTH:
            input_ids = input_ids[:, -CONTEXT_LENGTH:]
            attention_mask = attention_mask[:, -CONTEXT_LENGTH:]

    print('-' * 100)

    if len(images) != 0:
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

    else:
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


def main():
    iface = gr.ChatInterface(
        fn=generate_next_tokens, 
        multimodal=True,
        title='Image, PDF, and Text Chat with Phi Models',
        additional_inputs=[
            gr.Dropdown(
                choices=[
                    'microsoft/Phi-3.5-mini-instruct',
                    'microsoft/Phi-3-small-8k-instruct',
                    'microsoft/Phi-3-medium-4k-instruct',
                    'microsoft/Phi-3-small-128k-instruct',
                    'microsoft/Phi-3-medium-128k-instruct',
                    'microsoft/Phi-3.5-vision-instruct'
                ],
                label='Select Model',
                value='microsoft/Phi-3.5-mini-instruct'
            )
        ],
        theme=gr.themes.Soft(primary_hue='orange', secondary_hue='gray')
    )
    
    iface.launch(share=args.share)

if __name__ == '__main__':
    main()