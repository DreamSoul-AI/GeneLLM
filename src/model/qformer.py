from transformers import BertConfig, BertModel
import torch
import torch.nn as nn

class QFormer(nn.Module):
    def __init__(self,
                 bert_model_name='bert-base-uncased',
                 num_query_tokens=32,
                 hidden_size=768,
                 encoder_width=768,
                 cross_attention_freq=2):
        super().__init__()

        # Modify BERT config to support cross-attention
        config = BertConfig.from_pretrained(bert_model_name)
        config.add_cross_attention = True
        config.cross_attention_freq = cross_attention_freq
        config.encoder_width = encoder_width

        self.bert = BertModel(config)

        self.query_tokens = nn.Parameter(torch.randn(1, num_query_tokens, hidden_size))

    def forward(self, encoder_hidden_states, encoder_attention_mask=None):
        batch_size = encoder_hidden_states.size(0)

        # Expand query tokens
        query_tokens = self.query_tokens.expand(batch_size, -1, -1)

        # Attention mask defaults to all-visible if not provided
        if encoder_attention_mask is None:
            encoder_attention_mask = torch.ones(
                encoder_hidden_states.size()[:-1], dtype=torch.long, device=encoder_hidden_states.device
            )

        # Run query tokens through BERT with cross-attention to encoder_hidden_states
        outputs = self.bert(
            inputs_embeds=query_tokens,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            return_dict=True,
        )

        return outputs.last_hidden_state  # [B, Q, H]
