import os
import torch
from transformers import AutoTokenizer, AutoModel

task_names = ['EMP', 'mouse', 'promcore', 'prom300', 'splice', 'tf', 'virus']

# Use os.path.join for cross-platform compatibility
description_path = os.path.join(".", "dataset", "description_V2")
descriptions = []
for task in task_names:
    task_file_path = os.path.join(description_path, f"{task}.txt")
    with open(task_file_path, 'r') as f:
        descriptions.append(f.read().replace('\n', ''))
        f.close()

tokenizer = AutoTokenizer.from_pretrained('intfloat/multilingual-e5-large-instruct')
model = AutoModel.from_pretrained('intfloat/multilingual-e5-large-instruct')

# Tokenize the input texts
batch_dict = tokenizer(descriptions, max_length=512, padding=True, truncation=True, return_tensors='pt')

outputs = model(**batch_dict)
embeddings = outputs.last_hidden_state

# Create embedding dictionary
embeddings_named = {name:  # name of the dataset
                    {'embedding_seq': embeddings[i],   # embedding seq with paddings
                     'attention_mask': batch_dict['attention_mask'][i],  # attention_mask for this embedding seq 
                     'pooler_output': outputs.pooler_output[i]}  # pooler_output for this embedding seq (CLS embedding)  
                    for i, name in enumerate(task_names)}

# Use os.path.join for base folder
base_folder = os.path.join(".", "data", "GUE", "processed")

for task in task_names:
    embedding_file_path = os.path.join(base_folder, task, "embedding_seq.pth")
    attention_mask_file_path = os.path.join(base_folder, task, "attention_mask.pth")
    pooler_output_file_path = os.path.join(base_folder, task, "pooler_output.pth")
    torch.save(embeddings_named[task]['embedding_seq'], embedding_file_path)
    torch.save(embeddings_named[task]['attention_mask'], attention_mask_file_path)
    torch.save(embeddings_named[task]['pooler_output'], pooler_output_file_path)

print('finish! embeddings are saved in the data/GUE/processed folder')