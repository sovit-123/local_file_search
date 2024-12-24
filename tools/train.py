"""
Training a Sentence Transformer model.
"""

import os

from sentence_transformers import SentenceTransformer, losses, SentenceTransformerTrainer
from sentence_transformers.losses import ContrastiveTensionDataLoader

def load_text_files(folder_path):
    file_names = os.listdir(folder_path)

    valid_ratio = 0.0
    valid_samples = int(len(file_names)*valid_ratio)
    train_samples = int(len(file_names)-valid_samples)

    print(f"Train samples: {train_samples}, valid samples: {valid_samples}")

    train_texts = []
    valid_texts = []
    for i, file_name in enumerate(file_names):
        if i < train_samples:
            if file_name.endswith('.txt'):
                with open(os.path.join(folder_path, file_name), 'r') as f:
                    train_texts.append(f.read())
        else:
            if file_name.endswith('.txt'):
                with open(os.path.join(folder_path, file_name), 'r') as f:
                    valid_texts.append(f.read())

    return train_texts, valid_texts

folder_path = '../data/paper_files'
train_texts, valid_texts = load_text_files(folder_path)

model_name = 'all-MiniLM-L6-v2'

model = SentenceTransformer(model_name)

model.max_seq_length = 512

print(model)

train_dataloader = ContrastiveTensionDataLoader(
    train_texts, 
    batch_size=16, 
    pos_neg_ratio=4
)
valid_dataloader = ContrastiveTensionDataLoader(
    valid_texts, 
    batch_size=16, 
    pos_neg_ratio=4
)

loss = losses.ContrastiveTensionLoss(model=model)

history = model.fit(
    [(train_dataloader, loss)],
    epochs=50,
    checkpoint_save_total_limit=2,
    checkpoint_path='outputs'
)

# trainer = SentenceTransformerTrainer(
#     model=model,
#     train_dataset=train_dataloader,
#     eval_dataset=valid_dataloader,
#     loss=loss
# )

# history = trainer.train()

print(history)