import pytorch_lightning as pl
from torch import nn
from transformers import BertModel
from transformers.models import bert


class BertMNLIFinetuner(pl.LightningModule):
    def __init__(self):
        super().__init__()

        self.bert = BertModel.from_pretrained("bert-base-cased", output_attentions=True)
        self.W = nn.Linear(bert.config.hidden_size, 3)
        self.num_classes = 3

    def forward(self, input_ids, attention_mask, token_type_ids):
        h, _, attn = self.bert(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)

        h_cls = h[:, 0]
        logits = self.W(h_cls)
        return logits, attn
