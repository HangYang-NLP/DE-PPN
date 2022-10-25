import torch.nn as nn
import torch
from transformers.models.bert.modeling_bert import BertIntermediate, BertOutput, BertAttention, BertSelfAttention, BertConfig, BertEncoder
from models.set_criterion import SetCriterion
from DEE import transformer

BertLayerNorm = torch.nn.LayerNorm


class SetPred4DEE(nn.Module):
    def __init__(self, config, event_type2role_index_list, return_intermediate = True):
        super(SetPred4DEE, self).__init__()
        self.config = config
        event_type_classes = config.event_type_classes
        self.num_generated_sets = config.num_generated_sets
        num_set_layers =  config.num_set_decoder_layers
        num_role_layers =  config.num_role_decoder_layers
        self.cost_weight = config.cost_weight
        self.event_cls = nn.Linear(config.hidden_size, event_type_classes)
        self.query_embed = nn.Embedding(self.num_generated_sets, config.hidden_size)
        self.event_type2role_index_list = event_type2role_index_list
        self.role_index_list = [role_index for role_index_list in event_type2role_index_list for role_index in role_index_list]
        self.event_type2role_index_list.append(None)
        self.role_index_num = len(set(self.role_index_list))
        self.role_embed = nn.Embedding(self.role_index_num, config.hidden_size)
        self.role_embed4None = nn.Embedding(1, config.hidden_size)

        if config.use_event_type_enc:
            self.event_type_embed = nn.Embedding(5, config.hidden_size)

        self.return_intermediate = return_intermediate
        self.dropout = nn.Dropout(config.hidden_dropout)
        self.LayerNorm = BertLayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        if num_set_layers > 0:
            self.set_layers = nn.ModuleList([DecoderLayer(config) for _ in range(num_set_layers)])
        else:
            self.set_layers = False

        if num_role_layers > 0:
            self.role_layers = nn.ModuleList([DecoderLayer(config) for _ in range(num_role_layers)])
        else:
            self.role_layers = False

        self.metric_1 = nn.Linear(config.hidden_size, config.hidden_size)
        self.metric_2 = nn.Linear(config.hidden_size, config.hidden_size)
        self.metric_3 = nn.Linear(config.hidden_size, config.hidden_size)
        self.metric_4 = nn.Linear(config.hidden_size, 1, bias=False)
        self.event_type_weight = config.event_type_weight
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cuda')
        self.criterion = SetCriterion(config, event_type_classes, self.event_type_weight, self.cost_weight)
        self.cross_entropy = nn.CrossEntropyLoss(reduction="sum")

        if self.config.use_sent_span_encoder:
            self.span_sent_encodr = transformer.make_transformer_encoder(config.num_tf_layers, config.hidden_size, ff_size=config.ff_size, dropout=config.dropout)

        if self.config.use_role_decoder:
            self.event2role_decoder = transformer.make_transformer_decoder(
                config.num_event2role_decoder_layer, config.hidden_size, ff_size=config.ff_size, dropout=config.dropout
            )

    def forward(self, doc_sent_context, batch_span_context, doc_span_info, event_type_pred = None, train_flag = True):
        """
        :param doc_sent_context:   num_sents, hidden_size
        :param batch_span_context:   num_candidate_args, hidden_size
        :param doc_span_info:       event_infor of a doc
        :return:
        """
        role4None_embed = self.role_embed4None.weight.unsqueeze(0)
        batch_span_context = batch_span_context.unsqueeze(0)
        batch_span_context = torch.cat((batch_span_context, role4None_embed), 1)
        num_pred_entities = batch_span_context.size()[1]
        doc_sent_context = doc_sent_context.unsqueeze(0)
        if self.config.use_sent_span_encoder:
            doc_span_sent_context = torch.cat((batch_span_context, doc_sent_context), 1)
            doc_span_sent_context = self.span_sent_encodr(doc_span_sent_context, None)
        else:
            doc_span_sent_context = torch.cat((batch_span_context, doc_sent_context), 1)

        bsz = doc_span_sent_context.size()[0]
        hidden_states = self.query_embed.weight.unsqueeze(0).repeat(bsz, 1, 1)
        hidden_states = self.dropout(self.LayerNorm(hidden_states))

        if self.config.use_event_type_enc:
            event_type_tensor = torch.tensor(event_type_pred, dtype=torch.long, requires_grad=False).to(self.device)
            event_type_embed = self.event_type_embed(event_type_tensor)
            hidden_states = hidden_states + event_type_embed

        all_hidden_states = ()
        if self.set_layers:
            for i, layer_module in enumerate(self.set_layers):
                if self.return_intermediate:
                    all_hidden_states = all_hidden_states + (hidden_states,)
                layer_outputs = layer_module(
                    # hidden_states, batch_span_context
                    hidden_states, doc_sent_context
                    # hidden_states, doc_span_sent_context
                )
                hidden_states = layer_outputs[0]
        else:
            hidden_states = hidden_states

        ### event type classification (no-None or None)
        pred_doc_event_logps = self.event_cls(hidden_states).squeeze(0)
        if train_flag:
            event_type_idxs_list = doc_span_info.pred_event_type_idxs_list[event_type_pred][:self.num_generated_sets]
            event_arg_idxs_objs_list = doc_span_info.pred_event_arg_idxs_objs_list[event_type_pred][:self.num_generated_sets]

        event_index2role_list = [self.event_type2role_index_list[event_type_pred]]
        event_index2role_index_tensor = torch.tensor(event_index2role_list, dtype=torch.long, requires_grad=False).to(self.device)
        num_roles = len(event_index2role_list[0])
        event_role_embed = self.role_embed(event_index2role_index_tensor)
        event_role_embed = self.dropout(self.LayerNorm(event_role_embed))
        all_hidden_states = ()

        if self.role_layers:
            for i, layer_module in enumerate(self.role_layers):
                if self.return_intermediate:
                    all_hidden_states = all_hidden_states + (event_role_embed,)
                layer_outputs = layer_module(
                    # event_role_embed, doc_sent_context
                    event_role_embed, batch_span_context
                    # event_role_embed, doc_span_sent_context
                )
                event_role_embed = layer_outputs[0]
        else:
            event_role_embed = event_role_embed
        event_role_hidden_states = event_role_embed
        if self.config.use_role_decoder:
            pred_role_enc = torch.repeat_interleave(event_role_hidden_states.unsqueeze(1), repeats=self.num_generated_sets, dim=1)
            pred_set_role_enc = self.event2role_decoder(pred_role_enc.squeeze(0), hidden_states.unsqueeze(2).squeeze(0), None, None)
            pred_set_role_tensor = self.metric_1(pred_set_role_enc.unsqueeze(2)) + self.metric_2(doc_span_sent_context).unsqueeze(1)
        else:
            pred_set_tensor = self.metric_1(hidden_states).unsqueeze(2) + self.metric_2(doc_span_sent_context).unsqueeze(1)
            pred_set_role_tensor = pred_set_tensor.unsqueeze(2) + self.metric_3(event_role_hidden_states).unsqueeze(1).unsqueeze(3)

        pred_role_logits = self.metric_4(torch.tanh(pred_set_role_tensor)).squeeze()
        pred_role_logits = pred_role_logits.view(self.num_generated_sets, num_roles, -1) # [num_sets, num_roles, num_entities]
        pred_role_logits = pred_role_logits[:,:,:num_pred_entities]
        outputs = {'pred_doc_event_logps': pred_doc_event_logps,'pred_role_logits': pred_role_logits}

        if train_flag:
            targets = {'doc_event_label': event_type_idxs_list, 'role_label': event_arg_idxs_objs_list}
            loss = self.criterion(outputs, targets)
            return loss, outputs
        else:
            return outputs


class DecoderLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.attention = BertAttention(config)
        self.crossattention = BertAttention(config)
        self.intermediate = BertIntermediate(config)
        self.output = BertOutput(config)

    def forward(
        self,
        hidden_states,
        encoder_hidden_states,
        encoder_attention_mask = None
    ):
        self_attention_outputs = self.attention(hidden_states)
        attention_output = self_attention_outputs[0]
        outputs = self_attention_outputs[1:]  # add self attentions if we output attention weights

        encoder_batch_size, encoder_sequence_length, _ = encoder_hidden_states.size()
        encoder_hidden_shape = (encoder_batch_size, encoder_sequence_length)

        cross_attention_outputs = self.crossattention(
            hidden_states=attention_output, encoder_hidden_states=encoder_hidden_states
        )
        attention_output = cross_attention_outputs[0]
        outputs = outputs + cross_attention_outputs[1:]  # add cross attentions if we output attention weights

        intermediate_output = self.intermediate(attention_output)
        layer_output = self.output(intermediate_output, attention_output)
        outputs = (layer_output,) + outputs
        return outputs
