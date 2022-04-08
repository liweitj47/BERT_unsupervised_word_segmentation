import torch
import torch.nn as nn
from torch.nn import CrossEntropyLoss
from transformers import BertModel, BertPreTrainedModel
from transformers.modeling_bert import BertOnlyMLMHead

class SegmentBERT(BertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels
        config.attention_probs_dropout_prob = 0.0
        config.hidden_dropout_prob = 0.0

        self.bert = BertModel(config)
        self.cls = BertOnlyMLMHead(config)
        self.seq_classifier = nn.Linear(config.hidden_size, config.num_labels)

        self.init_weights()
    
    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        masked_lm_labels=None,
        mode=0
    ):
        '''
        mode == 0: not run sequence labeling
        mode == 1: run sequence labeling
        '''
        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
        ) # sequence_output, pooled_output, (hidden_states), (attentions)

        sequence_output = outputs[0]
        outputs = outputs[:1]

        if masked_lm_labels is not None:
            prediction_scores = self.cls(sequence_output)
            loss_fct = CrossEntropyLoss()  # -100 index = padding token
            # Only keep active parts of the loss
            if attention_mask is not None:
                active_loss = attention_mask.view(-1) == 1
                active_logits = prediction_scores.view(-1, self.num_labels)
                active_labels = torch.where(
                    active_loss, masked_lm_labels.view(-1), torch.tensor(loss_fct.ignore_index).type_as(masked_lm_labels)
                )
                masked_lm_loss = loss_fct(active_logits, active_labels)
            else:
                masked_lm_loss = loss_fct(prediction_scores.view(-1, self.config.vocab_size), masked_lm_labels.view(-1))
            del prediction_scores
            outputs = (masked_lm_loss,) + outputs

        if mode == 1:
            logits = self.seq_classifier(sequence_output)
            outputs = (logits,) + outputs

        return outputs  # (logits,) (masked_lm_loss,) sequence_output