import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel

def get_embedding(text: str) -> torch.Tensor:
    # Initialize tokenizer and model (you should do this once and reuse)
    tokenizer = AutoTokenizer.from_pretrained('intfloat/multilingual-e5-large-instruct')
    model = AutoModel.from_pretrained('intfloat/multilingual-e5-large-instruct')
    
    # Tokenize the text
    batch_dict = tokenizer(text, max_length=512, padding=True, truncation=True, return_tensors='pt')
    
    # Get model outputs
    outputs = model(**batch_dict)
    
    # Average pooling
    attention_mask = batch_dict['attention_mask']
    last_hidden = outputs.last_hidden_state.masked_fill(~attention_mask[..., None].bool(), 0.0)
    embeddings = last_hidden.sum(dim=1) / attention_mask.sum(dim=1)[..., None]
    
    # Normalize embeddings
    embeddings = F.normalize(embeddings, p=2, dim=1)
    
    return embeddings  # Return the first (and only) embedding

# Example usage
text = "Hello, world!"
embedding = get_embedding(text)
print(embedding.shape)  # torch.Size([1, 1024])

# Example usage
text2 = ["how are you", "你好"]
embedding2 = get_embedding(text2)
print(embedding2.shape)   # torch.Size([2, 1024])

# 计算余弦相似度
cosine_similarity = F.cosine_similarity(embedding2[0], embedding2[1], dim=0)
print(cosine_similarity)  # 输出两个文本之间的余弦相似度

