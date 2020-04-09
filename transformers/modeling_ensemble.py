import logging
import math
import os
from collections import OrderedDict
import torch
from torch import nn
from torch.nn import CrossEntropyLoss, MSELoss

from .activations import gelu, gelu_new, swish
from .configuration_bert import BertConfig
from .file_utils import add_start_docstrings, add_start_docstrings_to_callable
from .modeling_utils import PreTrainedModel, prune_linear_layer


logger = logging.getLogger(__name__)
from .modeling_bert import BertPreTrainedModel, BertModel
from .modeling_roberta import RobertaModel




class EnsembleForOffensiveClassification(BertPreTrainedModel):
    def __init__(self, bert_config, roberta_config):
        super().__init__(bert_config, roberta_config)
        # self.num_labels = bert_config.num_labels
        self.num_labels = 2
        self.ftcbert = BertModel(bert_config)
        self.ptftcbert = BertModel(bert_config)
        self.ptftrbert = BertModel(bert_config)
        self.ftcroberta = RobertaModel(roberta_config)
        self.ptftcroberta = RobertaModel(roberta_config)
        self.ptftrroberta = RobertaModel(roberta_config)
        self.dropout = nn.Dropout(bert_config.hidden_dropout_prob)
        self.classfication_classifier = nn.Linear(6*bert_config.hidden_size, self.config.num_labels)
        # self.regression_classifier = nn.Linear(bert_config.hidden_size, 1)
        self.init_weights()

    def forward(
        self,
        bert_input_ids=None,
        roberta_input_ids=None,
        classification_labels=None,
        bert_attention_mask=None,
        roberta_attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        # regression_labels=None
    ):
        pooled_output = torch.cat((self.ftcbert(
            bert_input_ids,
            attention_mask=bert_attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
        )[1],self.ptftcbert(
            bert_input_ids,
            attention_mask=bert_attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
        )[1],self.ptftrbert(
            bert_input_ids,
            attention_mask=bert_attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
        )[1],self.ftcroberta(
            roberta_input_ids,
            attention_mask=roberta_attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
        )[1],self.ptftcroberta(
            roberta_input_ids,
            attention_mask=roberta_attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
        )[1],self.ptftrroberta(
            roberta_input_ids,
            attention_mask=roberta_attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
        )[1]), -1)

        pooled_output = self.dropout(pooled_output)
        classification_logits = self.classfication_classifier(pooled_output)
        outputs = (classification_logits, )   # add hidden states and attention if they are here

        if classification_labels is not None:
            classification_loss_fct = CrossEntropyLoss()
            classification_loss = classification_loss_fct(classification_logits.view(-1, self.num_labels), classification_labels.view(-1))
            loss = classification_loss
            outputs = (loss,) + outputs

        return outputs  # (loss), logits, (hidden_states), (attentions)

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, *model_args, **kwargs):
        bert_config = kwargs.pop("bert_config", None)
        roberta_config = kwargs.pop("roberta_config", None)
        model = cls(bert_config,roberta_config)
        model.cpu()
        def rename_state_dict_prefix(old_state_dict, prefix, new_prefix):
            new_state_dict = OrderedDict()
            for key in old_state_dict.keys():
                if prefix in key:
                    new_key = key.replace(prefix, new_prefix)
                else:
                    new_key = key
                new_state_dict[new_key] = old_state_dict[key]
            return new_state_dict
        import os.path as os
        ftcbert_state_dict = torch.load(os.join(pretrained_model_name_or_path,"ftcbert.bin"))
        ptftcbert_state_dict = torch.load(os.join(pretrained_model_name_or_path, "ptftcbert.bin"))
        ptftrbert_state_dict = torch.load(os.join(pretrained_model_name_or_path, "ptftrbert.bin"))
        ftcroberta_state_dict = torch.load(os.join(pretrained_model_name_or_path, "ftcroberta.bin"))
        ptftcroberta_state_dict = torch.load(os.join(pretrained_model_name_or_path, "ptftcroberta.bin"))
        ptftrroberta_state_dict = torch.load(os.join(pretrained_model_name_or_path, "ptftrroberta.bin"))

        ftcbert_state_dict = rename_state_dict_prefix(ftcbert_state_dict, "bert", "ftcbert")
        ptftcbert_state_dict = rename_state_dict_prefix(ptftcbert_state_dict, "bert", "ptftcbert")
        ptftrbert_state_dict = rename_state_dict_prefix(ptftrbert_state_dict, "bert", "ptftrbert")
        ftcroberta_state_dict = rename_state_dict_prefix(ftcroberta_state_dict, "roberta", "ftcroberta")
        ptftcroberta_state_dict = rename_state_dict_prefix(ptftcroberta_state_dict, "roberta", "ptftcroberta")
        ptftrroberta_state_dict = rename_state_dict_prefix(ptftrroberta_state_dict, "roberta", "ptftrroberta")

        state = model.state_dict()
        state.update(ftcbert_state_dict)
        state.update(ptftcbert_state_dict)
        state.update(ptftrbert_state_dict)
        state.update(ftcroberta_state_dict)
        state.update(ptftcroberta_state_dict)
        state.update(ptftrroberta_state_dict)
        model.load_state_dict(state, strict=False)
        model.tie_weights()
        model.eval()
        print("load model successful!")
        return model

