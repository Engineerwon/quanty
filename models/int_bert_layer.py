import math
import torch
import transformers
from torch import nn
from typing import Optional, Tuple, List
from quantize.int_linear import QuantLinear
from quantize.int_matmul import QuantMatMul
import torch.nn.functional as F
from quantize.omni_norm import OmniLayerNorm
from collections import OrderedDict
import pdb
from models.models_utils import truncate_number
from models.transformation import *
from transformers.pytorch_utils import apply_chunking_to_forward, find_pruneable_heads_and_indices, prune_linear_layer

from transformers.models.bert.configuration_bert import BertConfig




class QuantBertSelfAttention_2to12(nn.Module):
    """Multi-headed attention from 'Attention Is All You Need' paper"""

    def __init__(
        self,
        org_module: nn.Module, 
        former_final_layernorm: nn.LayerNorm,
        hidden_size: int,
        num_attention_heads: int,
        position_embedding_type=None,
        is_decoder: bool = False,
        bias: bool = True,
        args=None,
        disable_act_quant=False,
    ):
        super().__init__()

        self.hidden_size = hidden_size
        if self.hidden_size % num_attention_heads != 0:
            raise ValueError(
                f"The hidden size ({self.hidden_size}) is not a multiple of the number of attention "
                f"heads ({num_attention_heads})"
            )
        
        self.num_attention_heads = num_attention_heads
        self.attention_head_size = int(self.hidden_size / num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size
        self.is_decoder=False




        self.former_final_layernorm = OmniLayerNorm(former_final_layernorm) 

        
        self.key = QuantLinear(
            org_module.key,
            args.k_quant_params,
            args.act_quant_params,
        )
        self.value = QuantLinear(
            org_module.value,
            args.v_quant_params,
            args.act_quant_params,
        )
        self.query = QuantLinear(
            org_module.query,
            args.q_quant_params,
            args.act_quant_params,
        )
        self.qkt_matmul = QuantMatMul(
            args.q_quant_params, args.k_quant_params, matmul_func=torch.matmul
        )
        self.pv_matmul = QuantMatMul(
            args.p_quant_params, args.v_quant_params, matmul_func=torch.matmul
        )

        
        """이 부분이 문제가 될수도. 직접 임베딩 타입을 지정해주어야되나?"""
        # self.position_embedding_type = position_embedding_type or getattr(
        #     config, "position_embedding_type", "absolute"
        # )
        # if self.position_embedding_type == "relative_key" or self.position_embedding_type == "relative_key_query":
        #     self.max_position_embeddings = config.max_position_embeddings
        #     self.distance_embedding = nn.Embedding(2 * config.max_position_embeddings - 1, self.attention_head_size)
        
        self.position_embedding_type = position_embedding_type



        self.use_weight_quant = False
        self.use_act_quant = False

    def transpose_for_scores(self, x: torch.Tensor) -> torch.Tensor:
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward( #일단 FloatTensor를 전부 Tenosor로 바꿈
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.Tensor] = None,
        past_key_value: Optional[Tuple[Tuple[torch.Tensor]]] = None,
        output_attentions: Optional[bool] = False,
    ) -> Tuple[torch.Tensor]:
        
        hidden_states = self.former_final_layernorm(hidden_states) #이래도 되나?


        mixed_query_layer = self.query(hidden_states)
        

        # If this is instantiated as a cross-attention module, the keys
        # and values come from an encoder; the attention mask needs to be
        # such that the encoder's padding tokens are not attended to.
        is_cross_attention = encoder_hidden_states is not None

        if is_cross_attention and past_key_value is not None:
            # reuse k,v, cross_attentions
            key_layer = past_key_value[0]
            value_layer = past_key_value[1]
            attention_mask = encoder_attention_mask
        elif is_cross_attention:
            key_layer = self.transpose_for_scores(self.key(encoder_hidden_states))
            value_layer = self.transpose_for_scores(self.value(encoder_hidden_states))
            attention_mask = encoder_attention_mask
        elif past_key_value is not None:
            key_layer = self.transpose_for_scores(self.key(hidden_states))
            value_layer = self.transpose_for_scores(self.value(hidden_states))
            key_layer = torch.cat([past_key_value[0], key_layer], dim=2)
            value_layer = torch.cat([past_key_value[1], value_layer], dim=2)
        else:# 이것만 쓸거다
            key_layer = self.key(hidden_states)
            key_layer = self.transpose_for_scores(key_layer)
            key_layer = self.qkt_matmul.quant_x2(key_layer)
            value_layer = self.transpose_for_scores(self.value(hidden_states))
            value_layer = self.pv_matmul.quant_x2(value_layer)

        query_layer = self.transpose_for_scores(mixed_query_layer)
        query_layer = self.qkt_matmul.quant_x1(query_layer)

        use_cache = past_key_value is not None
        if self.is_decoder: #안쓴다
            # if cross_attention save Tuple(torch.Tensor, torch.Tensor) of all cross attention key/value_states.
            # Further calls to cross_attention layer can then reuse all cross-attention
            # key/value_states (first "if" case)
            # if uni-directional self-attention (decoder) save Tuple(torch.Tensor, torch.Tensor) of
            # all previous decoder key/value_states. Further calls to uni-directional self-attention
            # can concat previous decoder key/value_states to current projected key/value_states (third "elif" case)
            # if encoder bi-directional self-attention `past_key_value` is always `None`
            past_key_value = (key_layer, value_layer)

        # Take the dot product between "query" and "key" to get the raw attention scores.
        attention_scores = self.qkt_matmul(query_layer, key_layer.transpose(-1, -2))

        if self.position_embedding_type == "relative_key" or self.position_embedding_type == "relative_key_query":
            query_length, key_length = query_layer.shape[2], key_layer.shape[2]
            if use_cache:
                position_ids_l = torch.tensor(key_length - 1, dtype=torch.long, device=hidden_states.device).view(
                    -1, 1
                )
            else:
                position_ids_l = torch.arange(query_length, dtype=torch.long, device=hidden_states.device).view(-1, 1)
            position_ids_r = torch.arange(key_length, dtype=torch.long, device=hidden_states.device).view(1, -1)
            distance = position_ids_l - position_ids_r

            positional_embedding = self.distance_embedding(distance + self.max_position_embeddings - 1)
            positional_embedding = positional_embedding.to(dtype=query_layer.dtype)  # fp16 compatibility

            if self.position_embedding_type == "relative_key":
                relative_position_scores = torch.einsum("bhld,lrd->bhlr", query_layer, positional_embedding)
                attention_scores = attention_scores + relative_position_scores
            elif self.position_embedding_type == "relative_key_query":
                relative_position_scores_query = torch.einsum("bhld,lrd->bhlr", query_layer, positional_embedding)
                relative_position_scores_key = torch.einsum("bhrd,lrd->bhlr", key_layer, positional_embedding)
                attention_scores = attention_scores + relative_position_scores_query + relative_position_scores_key

        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        if attention_mask is not None:
            # Apply the attention mask is (precomputed for all layers in BertModel forward() function)
            attention_scores = attention_scores + attention_mask

        # Normalize the attention scores to probabilities.
        attention_probs = nn.functional.softmax(attention_scores, dim=-1)
        attention_probs = self.pv_matmul.quant_x1(attention_probs)

        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.
        # attention_probs = self.dropout(attention_probs)

        # Mask heads if we want to
        if head_mask is not None:
            attention_probs = attention_probs * head_mask

        context_layer = self.pv_matmul(attention_probs, value_layer)

        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(new_context_layer_shape)

        outputs = (context_layer, attention_probs) if output_attentions else (context_layer,)

        if self.is_decoder:
            outputs = outputs + (past_key_value,)
        return outputs

    def set_quant_state(self, weight_quant: bool = False, act_quant: bool = False):
        # setting weight quantization here does not affect actual forward pass
        self.use_weight_quant = weight_quant
        self.use_act_quant = act_quant
        for m in self.modules():
            if isinstance(m, (QuantLinear, QuantMatMul)):
                m.set_quant_state(weight_quant, act_quant)

class QuantBertSelfAttention_1(nn.Module): 
    """Multi-headed attention from 'Attention Is All You Need' paper"""

    def __init__(
        self,
        org_module: nn.Module, 
        hidden_size: int,
        num_attention_heads: int,
        position_embedding_type=None,
        is_decoder: bool = False,
        bias: bool = True,
        args=None,
        disable_act_quant=False,
        config=None
    ):
        super().__init__()

        self.hidden_size = hidden_size
        if self.hidden_size % num_attention_heads != 0:
            raise ValueError(
                f"The hidden size ({self.hidden_size}) is not a multiple of the number of attention "
                f"heads ({num_attention_heads})"
            )
        
        self.num_attention_heads = num_attention_heads
        self.attention_head_size = int(self.hidden_size / num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size
        self.is_decoder=False

        
        self.key = QuantLinear(
            org_module.key,
            args.k_quant_params,
            args.act_quant_params,
        )
        self.value = QuantLinear(
            org_module.value,
            args.v_quant_params,
            args.act_quant_params,
        )
        self.query = QuantLinear(
            org_module.query,
            args.q_quant_params,
            args.act_quant_params,
        )
        self.qkt_matmul = QuantMatMul(
            args.q_quant_params, args.k_quant_params, matmul_func=torch.matmul
        )
        self.pv_matmul = QuantMatMul(
            args.p_quant_params, args.v_quant_params, matmul_func=torch.matmul
        )

        
        """이 부분이 문제가 될수도. 직접 임베딩 타입을 지정해주어야되나?"""
        self.position_embedding_type = position_embedding_type or getattr(
            config, "position_embedding_type", "absolute"
        )
        if self.position_embedding_type == "relative_key" or self.position_embedding_type == "relative_key_query":
            self.max_position_embeddings = config.max_position_embeddings
            self.distance_embedding = nn.Embedding(2 * config.max_position_embeddings - 1, self.attention_head_size)
        
        



        self.use_weight_quant = False
        self.use_act_quant = False

    def transpose_for_scores(self, x: torch.Tensor) -> torch.Tensor:
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward( #일단 FloatTensor를 전부 Tenosor로 바꿈
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.Tensor] = None,
        past_key_value: Optional[Tuple[Tuple[torch.Tensor]]] = None,
        output_attentions: Optional[bool] = False,
    ) -> Tuple[torch.Tensor]:
        mixed_query_layer = self.query(hidden_states)
        

        # If this is instantiated as a cross-attention module, the keys
        # and values come from an encoder; the attention mask needs to be
        # such that the encoder's padding tokens are not attended to.
        is_cross_attention = encoder_hidden_states is not None

        if is_cross_attention and past_key_value is not None:
            # reuse k,v, cross_attentions
            key_layer = past_key_value[0]
            value_layer = past_key_value[1]
            attention_mask = encoder_attention_mask
        elif is_cross_attention:
            key_layer = self.transpose_for_scores(self.key(encoder_hidden_states))
            value_layer = self.transpose_for_scores(self.value(encoder_hidden_states))
            attention_mask = encoder_attention_mask
        elif past_key_value is not None:
            key_layer = self.transpose_for_scores(self.key(hidden_states))
            value_layer = self.transpose_for_scores(self.value(hidden_states))
            key_layer = torch.cat([past_key_value[0], key_layer], dim=2)
            value_layer = torch.cat([past_key_value[1], value_layer], dim=2)
        else:# 이것만 쓸거다
            key_layer = self.key(hidden_states)
            key_layer = self.transpose_for_scores(key_layer)
            key_layer = self.qkt_matmul.quant_x2(key_layer)
            value_layer = self.transpose_for_scores(self.value(hidden_states))
            value_layer = self.pv_matmul.quant_x2(value_layer)

        query_layer = self.transpose_for_scores(mixed_query_layer)
        query_layer = self.qkt_matmul.quant_x1(query_layer)

        use_cache = past_key_value is not None
        if self.is_decoder: #안쓴다
            # if cross_attention save Tuple(torch.Tensor, torch.Tensor) of all cross attention key/value_states.
            # Further calls to cross_attention layer can then reuse all cross-attention
            # key/value_states (first "if" case)
            # if uni-directional self-attention (decoder) save Tuple(torch.Tensor, torch.Tensor) of
            # all previous decoder key/value_states. Further calls to uni-directional self-attention
            # can concat previous decoder key/value_states to current projected key/value_states (third "elif" case)
            # if encoder bi-directional self-attention `past_key_value` is always `None`
            past_key_value = (key_layer, value_layer)
        
        # Take the dot product between "query" and "key" to get the raw attention scores.
        attention_scores = self.qkt_matmul(query_layer, key_layer.transpose(-1, -2))

        if self.position_embedding_type == "relative_key" or self.position_embedding_type == "relative_key_query":
            query_length, key_length = query_layer.shape[2], key_layer.shape[2]
            if use_cache:
                position_ids_l = torch.tensor(key_length - 1, dtype=torch.long, device=hidden_states.device).view(
                    -1, 1
                )
            else:
                position_ids_l = torch.arange(query_length, dtype=torch.long, device=hidden_states.device).view(-1, 1)
            position_ids_r = torch.arange(key_length, dtype=torch.long, device=hidden_states.device).view(1, -1)
            distance = position_ids_l - position_ids_r

            positional_embedding = self.distance_embedding(distance + self.max_position_embeddings - 1)
            positional_embedding = positional_embedding.to(dtype=query_layer.dtype)  # fp16 compatibility

            if self.position_embedding_type == "relative_key":
                relative_position_scores = torch.einsum("bhld,lrd->bhlr", query_layer, positional_embedding)
                attention_scores = attention_scores + relative_position_scores
            elif self.position_embedding_type == "relative_key_query":
                relative_position_scores_query = torch.einsum("bhld,lrd->bhlr", query_layer, positional_embedding)
                relative_position_scores_key = torch.einsum("bhrd,lrd->bhlr", key_layer, positional_embedding)
                attention_scores = attention_scores + relative_position_scores_query + relative_position_scores_key

        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        if attention_mask is not None:
            # Apply the attention mask is (precomputed for all layers in BertModel forward() function)
            attention_scores = attention_scores + attention_mask

        # Normalize the attention scores to probabilities.
        attention_probs = nn.functional.softmax(attention_scores, dim=-1)
        attention_probs = self.pv_matmul.quant_x1(attention_probs)

        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.
        # attention_probs = self.dropout(attention_probs)

        # Mask heads if we want to
        if head_mask is not None:
            attention_probs = attention_probs * head_mask

        context_layer = self.pv_matmul(attention_probs, value_layer)

        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(new_context_layer_shape)

        outputs = (context_layer, attention_probs) if output_attentions else (context_layer,)

        if self.is_decoder:
            outputs = outputs + (past_key_value,)
        return outputs

    def set_quant_state(self, weight_quant: bool = False, act_quant: bool = False):
        # setting weight quantization here does not affect actual forward pass
        self.use_weight_quant = weight_quant
        self.use_act_quant = act_quant
        for m in self.modules():
            if isinstance(m, (QuantLinear, QuantMatMul)):
                m.set_quant_state(weight_quant, act_quant)

class QuantBertSelfOutput(nn.Module):
    def __init__( self,
        org_module: nn.Module, 
        # is_decoder: bool = False,
        bias: bool = True,
        args=None,
        disable_act_quant=False,
        ):
        super().__init__()

        #dense를 out_proj으로 이름 바꿈
        self.dense = QuantLinear(
            org_module.dense, weight_quant_params=args.weight_quant_params, act_quant_params=args.act_quant_params
        )

        self.LayerNorm = OmniLayerNorm(org_module.LayerNorm)
        # self.dropout = nn.Dropout(config.hidden_dropout_prob)

        self.use_weight_quant = False
        self.use_act_quant = False

    def forward(self, hidden_states: torch.Tensor, input_tensor: torch.Tensor) -> torch.Tensor:
        hidden_states = self.dense(hidden_states)
        # hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states
    
    def set_quant_state(self, weight_quant: bool = False, act_quant: bool = False):
        # setting weight quantization here does not affect actual forward pass
        self.use_weight_quant = weight_quant
        self.use_act_quant = act_quant
        for m in self.modules():
            if isinstance(m, (QuantLinear, QuantMatMul)):
                m.set_quant_state(weight_quant, act_quant)






class QuantBertAttention_2to12(nn.Module):
    def __init__( self, org_module: nn.Module, former_final_layernorm: nn.LayerNorm,
                 hidden_size: int, num_attention_heads: int, 
                 
                 is_decoder: bool = False,
                 args= None,
                 position_embedding_type=None,
                 disable_act_quant=False,
                 ):
        super().__init__()
        self.self = QuantBertSelfAttention_2to12(
            hidden_size=hidden_size,
            num_attention_heads=num_attention_heads,
            org_module= org_module.self,
            former_final_layernorm=former_final_layernorm,
            position_embedding_type=position_embedding_type,
            # is_decoder: bool = False,
            args=args,
            disable_act_quant=False,)

        self.output = QuantBertSelfOutput(
            org_module=org_module.output,
            args=args,
            disable_act_quant=False,)

        self.pruned_heads = set()

    """일단은 prune 헤드는 죽이자"""
    # def prune_heads(self, heads):
    #     if len(heads) == 0:
    #         return
    #     heads, index = find_pruneable_heads_and_indices(
    #         heads, self.self.num_attention_heads, self.self.attention_head_size, self.pruned_heads
    #     )

    #     # Prune linear layers
    #     self.self.query = prune_linear_layer(self.self.query, index)
    #     self.self.key = prune_linear_layer(self.self.key, index)
    #     self.self.value = prune_linear_layer(self.self.value, index)
    #     self.output.dense = prune_linear_layer(self.output.dense, index, dim=1)

    #     # Update hyper params and store pruned heads
    #     self.self.num_attention_heads = self.self.num_attention_heads - len(heads)
    #     self.self.all_head_size = self.self.attention_head_size * self.self.num_attention_heads
    #     self.pruned_heads = self.pruned_heads.union(heads)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.Tensor] = None,
        past_key_value: Optional[Tuple[Tuple[torch.Tensor]]] = None,
        output_attentions: Optional[bool] = False,
    ) -> Tuple[torch.Tensor]:
        self_outputs = self.self(
            hidden_states,
            attention_mask,
            head_mask,
            encoder_hidden_states,
            encoder_attention_mask,
            past_key_value,
            output_attentions,
        )
        attention_output = self.output(self_outputs[0], hidden_states)
        outputs = (attention_output,) + self_outputs[1:]  # add attentions if we output them
        return outputs


class QuantBertAttention_1(nn.Module):
    def __init__( self, org_module: nn.Module, 
                 hidden_size: int, num_attention_heads: int, 
                 is_decoder: bool = False,
                 args= None,
                 position_embedding_type=None,
                 disable_act_quant=False,
                 config=None
                 ):
        super().__init__()
        self.self = QuantBertSelfAttention_1(
            hidden_size=hidden_size,
            num_attention_heads=num_attention_heads,
            org_module= org_module.self,
            position_embedding_type=position_embedding_type,
            # is_decoder: bool = False,
            args=args,
            disable_act_quant=False,
            config=config)

        self.output = QuantBertSelfOutput(
            org_module=org_module.output,
            args=args,
            disable_act_quant=False,)

        self.pruned_heads = set()

    """일단은 prune 헤드는 죽이자"""
    # def prune_heads(self, heads):
    #     if len(heads) == 0:
    #         return
    #     heads, index = find_pruneable_heads_and_indices(
    #         heads, self.self.num_attention_heads, self.self.attention_head_size, self.pruned_heads
    #     )

    #     # Prune linear layers
    #     self.self.query = prune_linear_layer(self.self.query, index)
    #     self.self.key = prune_linear_layer(self.self.key, index)
    #     self.self.value = prune_linear_layer(self.self.value, index)
    #     self.output.dense = prune_linear_layer(self.output.dense, index, dim=1)

    #     # Update hyper params and store pruned heads
    #     self.self.num_attention_heads = self.self.num_attention_heads - len(heads)
    #     self.self.all_head_size = self.self.attention_head_size * self.self.num_attention_heads
    #     self.pruned_heads = self.pruned_heads.union(heads)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.Tensor] = None,
        past_key_value: Optional[Tuple[Tuple[torch.Tensor]]] = None,
        output_attentions: Optional[bool] = False,
    ) -> Tuple[torch.Tensor]:
        
        self_outputs = self.self(
            hidden_states,
            attention_mask,
            head_mask,
            encoder_hidden_states,
            encoder_attention_mask,
            past_key_value,
            output_attentions,
        )
        attention_output = self.output(self_outputs[0], hidden_states)
        outputs = (attention_output,) + self_outputs[1:]  # add attentions if we output them
        return outputs










class QuantBertIntermediate(nn.Module):
    def __init__(self,
        org_module: nn.Module, 
        # is_decoder: bool = False,
        bias: bool = True,
        args=None,
        disable_act_quant=False,):

        super().__init__()
        #dense를 fc1으로 이름 바꿈
        self.dense = QuantLinear(
            org_module.dense, weight_quant_params=args.weight_quant_params, act_quant_params=args.act_quant_params
        )
        # if isinstance(config.hidden_act, str):
        #     self.intermediate_act_fn = ACT2FN[config.hidden_act]
        # else:
        #     self.intermediate_act_fn = config.hidden_act

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = self.dense(hidden_states)
        hidden_states = F.gelu(hidden_states) #한번 gelu로 해봅시당
        return hidden_states




class QuantBertOutput_1to11(nn.Module):
    def __init__(self,
        org_module: nn.Module, 
        # is_decoder: bool = False,
        bias: bool = True,
        args=None,
        disable_act_quant=False,):
        
        super().__init__()
       
        self.dense = QuantLinear(
            org_module.dense, weight_quant_params=args.weight_quant_params, act_quant_params=args.act_quant_params
        )
       
        # self.LayerNorm = OmniLayerNorm(org_module.LayerNorm) 
        # self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states: torch.Tensor, input_tensor: torch.Tensor) -> torch.Tensor:
        hidden_states = self.dense(hidden_states)
        # hidden_states = self.dropout(hidden_states)
        # hidden_states = self.LayerNorm(hidden_states + input_tensor), 마지막 layernorm만 다음 블럭으로 넘기자.
        hidden_states = hidden_states + input_tensor
        return hidden_states

class QuantBertOutput_12(nn.Module):
    def __init__(self,
        org_module: nn.Module, 
        # is_decoder: bool = False,
        bias: bool = True,
        args=None,
        disable_act_quant=False,):
        
        super().__init__()
        #dense를 fc2로 이름 바꿈
        self.dense = QuantLinear(
            org_module.dense, weight_quant_params=args.weight_quant_params, act_quant_params=args.act_quant_params
        )
       
        self.LayerNorm = OmniLayerNorm(org_module.LayerNorm) 
        # self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states: torch.Tensor, input_tensor: torch.Tensor) -> torch.Tensor:
        hidden_states = self.dense(hidden_states)
        # hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states




class QuantBertLayer_2to11(nn.Module):
    def __init__(
        self,
        config: BertConfig,
        ori_layer,
        former_final_layernorm,
        args,
    ):
        super().__init__()
        self.chunk_size_feed_forward = config.chunk_size_feed_forward
        self.seq_len_dim = 1
        
        self.attention = QuantBertAttention_2to12(
            org_module=ori_layer.attention,
            former_final_layernorm=former_final_layernorm,
            hidden_size=config.hidden_size,
            num_attention_heads=config.num_attention_heads,
            is_decoder=False,
            args=args,
            position_embedding_type=None,
        )


        self.is_decoder = False
        # self.add_cross_attention = config.add_cross_attention
        # self.dropout = config.dropout
        self.intermediate = QuantBertIntermediate(org_module=ori_layer.intermediate, args=args)
        
        self.output = QuantBertOutput_1to11(org_module=ori_layer.output, args=args)
        
        self.type = ori_layer.intermediate.dense.weight.dtype

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.FloatTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        encoder_hidden_states: Optional[torch.FloatTensor] = None,
        encoder_attention_mask: Optional[torch.FloatTensor] = None,
        past_key_value: Optional[Tuple[Tuple[torch.FloatTensor]]] = None,
        output_attentions: Optional[bool] = False,
    ) -> Tuple[torch.Tensor] :
       

        # Self Attention
        self_attn_past_key_value = past_key_value[:2] if past_key_value is not None else None
        self_attention_outputs = self.attention(
            hidden_states,
            attention_mask,
            head_mask,
            output_attentions=output_attentions,
            past_key_value=self_attn_past_key_value,
        )
        attention_output = self_attention_outputs[0]

        # if decoder, the last output is tuple of self-attn cache
        if self.is_decoder:
            outputs = self_attention_outputs[1:-1]
            present_key_value = self_attention_outputs[-1]
        else:
            outputs = self_attention_outputs[1:]  # add self attentions if we output attention weights

        # 일단 cross attention부분은 다 죽여버림.
        # cross_attn_present_key_value = None
        # if self.is_decoder and encoder_hidden_states is not None:
        #     if not hasattr(self, "crossattention"):
        #         raise ValueError(
        #             f"If `encoder_hidden_states` are passed, {self} has to be instantiated with cross-attention layers"
        #             " by setting `config.add_cross_attention=True`"
        #         )

        #     # cross_attn cached key/values tuple is at positions 3,4 of past_key_value tuple
        #     cross_attn_past_key_value = past_key_value[-2:] if past_key_value is not None else None
        #     cross_attention_outputs = self.crossattention(
        #         attention_output,
        #         attention_mask,
        #         head_mask,
        #         encoder_hidden_states,
        #         encoder_attention_mask,
        #         cross_attn_past_key_value,
        #         output_attentions,
        #     )
        #     attention_output = cross_attention_outputs[0]
        #     outputs = outputs + cross_attention_outputs[1:-1]  # add cross attentions if we output attention weights

        #     # add cross-attn cache to positions 3,4 of present_key_value tuple
        #     cross_attn_present_key_value = cross_attention_outputs[-1]
        #     present_key_value = present_key_value + cross_attn_present_key_value
        layer_output = apply_chunking_to_forward(
            self.feed_forward_chunk, self.chunk_size_feed_forward, self.seq_len_dim, attention_output
        )
        outputs = (layer_output,) + outputs

        return outputs
    
    def feed_forward_chunk(self, attention_output):
        intermediate_output = self.intermediate(attention_output)
        layer_output = self.output(intermediate_output, attention_output)
        return layer_output

       


    def set_quant_state(self, weight_quant: bool = False, act_quant: bool = False):
        # setting weight quantization here does not affect actual forward pass
        self.use_weight_quant = weight_quant
        self.use_act_quant = act_quant
        names = []
        for name, m in self.named_modules():
            if isinstance(m, (QuantLinear, QuantMatMul)):
                names.append(name)
                m.set_quant_state(weight_quant, act_quant)

    @torch.no_grad()
    def smooth_and_quant_inplace(self):
        # return
        if self.let:
            for name, module in self.named_parameters():
                if "smooth_scale" in name:
                    module.data = truncate_number(module)
            smooth_ln_fcs_inplace(self.self_attn_layer_norm,[self.self_attn.q_proj, self.self_attn.k_proj, self.self_attn.v_proj],
                                    self.qkv_smooth_scale,self.qkv_smooth_shift)
            smooth_ln_fcs_inplace(self.final_layer_norm,[self.fc1],
                                    self.fc1_smooth_scale,self.fc1_smooth_shift)
            smooth_fc_fc_inplace(self.self_attn.v_proj,self.self_attn.out_proj,
                                self.out_smooth_scale, self.out_smooth_shift)
            smooth_q_k_inplace(self.self_attn.q_proj, self.self_attn.k_proj,
                                self.qkt_smooth_scale)
        for name, module in self.named_modules():
            if isinstance(module, QuantLinear):
                module.weight = module.weight_quantizer(module.weight)
                module.use_temporary_parameter=False
                

    def clear_temp_variable(self):
        for name, module in self.named_modules():
            if isinstance(module, QuantLinear):
                del module.temp_weight
                del module.temp_bias

    def smooth_and_quant_temporary(self):
        if self.let:
            with torch.no_grad():
                for name, module in self.named_parameters():
                    if "smooth_scale" in name:
                        module.data = truncate_number(module)
            smooth_ln_fcs_temporary(self.self_attn_layer_norm,[self.self_attn.q_proj, self.self_attn.k_proj, self.self_attn.v_proj],
                                    self.qkv_smooth_scale,self.qkv_smooth_shift)
            smooth_ln_fcs_temporary(self.final_layer_norm,[self.fc1],
                                    self.fc1_smooth_scale,self.fc1_smooth_shift)
            smooth_fc_fc_temporary(self.self_attn.v_proj,self.self_attn.out_proj,
                                self.out_smooth_scale, self.out_smooth_shift)
            smooth_q_k_temporary(self.self_attn.q_proj, self.self_attn.k_proj,
                                self.qkt_smooth_scale)
            self.fc2.temp_weight = self.fc2.weight
        else:
            for name, module in self.named_modules():
                if isinstance(module, QuantLinear):
                    module.temp_weight = module.weight
        # quant
        for name, module in self.named_modules():
            if isinstance(module, QuantLinear):
                if hasattr(module, "temp_weight"):
                    module.temp_weight = module.weight_quantizer(module.temp_weight)
                else:
                    module.temp_weight = module.weight_quantizer(module.weight)
                if not hasattr(module, "temp_bias"):
                    module.temp_bias = module.bias
                module.use_temporary_parameter=True


    def let_parameters(self, use_shift=True):
        params = []
        template = "smooth" if use_shift else "smooth_scale"
        for n, m in self.named_parameters():
            if n.find(template) > -1:
                params.append(m)
        return iter(params)  

    def lwc_parameters(self):
        params = []
        for n, m in self.named_parameters():
            if n.find('bound_factor') > -1:
                params.append(m)
        return iter(params)  

    def omni_parameters(self, use_shift=True):
        params = []
        template = "smooth" if use_shift else "smooth_scale"
        for n, m in self.named_parameters():
            if n.find('bound_factor') > -1 or n.find(template) > -1:
                params.append(m)
        return iter(params)  
    
    def omni_state_dict(self, destination=None, prefix='', keep_vars=False):
        if destination is None:
            destination = OrderedDict()
        for name, param in self.named_parameters():
            if name.find('smooth') > -1 or name.find('bound_factor') > -1:
                destination[prefix + name] = param if keep_vars else param.detach()
        return destination
    

    def register_scales_and_zeros(self):
        for name, module in self.named_modules():
            if isinstance(module, QuantLinear):
                module.weight_quantizer.register_scales_and_zeros()



                
    
class QuantBertLayer_12(nn.Module):
    def __init__(
        self,
        config: BertConfig,
        ori_layer,
        former_final_layernorm,
        args,
    ):
        super().__init__()
        self.chunk_size_feed_forward = config.chunk_size_feed_forward
        self.seq_len_dim = 1
        
        self.attention = QuantBertAttention_2to12(
            org_module=ori_layer.attention,
            former_final_layernorm=former_final_layernorm,
            hidden_size=config.hidden_size,
            num_attention_heads=config.num_attention_heads,
            is_decoder=False,
            args=args,
            position_embedding_type=None,
            config=config
        )


        self.is_decoder = False
        # self.add_cross_attention = config.add_cross_attention
        # self.dropout = config.dropout
        self.intermediate = QuantBertIntermediate(org_module=ori_layer.intermediate, args=args)
        
        self.output = QuantBertOutput_12(org_module=ori_layer.output, args=args)
        
        self.type = ori_layer.intermediate.dense.weight.dtype

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.FloatTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        encoder_hidden_states: Optional[torch.FloatTensor] = None,
        encoder_attention_mask: Optional[torch.FloatTensor] = None,
        past_key_value: Optional[Tuple[Tuple[torch.FloatTensor]]] = None,
        output_attentions: Optional[bool] = False,
    ) -> Tuple[torch.Tensor] :
       

        # Self Attention
        self_attn_past_key_value = past_key_value[:2] if past_key_value is not None else None
        self_attention_outputs = self.attention(
            hidden_states,
            attention_mask,
            head_mask,
            output_attentions=output_attentions,
            past_key_value=self_attn_past_key_value,
        )
        attention_output = self_attention_outputs[0]

        # if decoder, the last output is tuple of self-attn cache
        if self.is_decoder:
            outputs = self_attention_outputs[1:-1]
            present_key_value = self_attention_outputs[-1]
        else:
            outputs = self_attention_outputs[1:]  # add self attentions if we output attention weights

        # 일단 cross attention부분은 다 죽여버림.
        # cross_attn_present_key_value = None
        # if self.is_decoder and encoder_hidden_states is not None:
        #     if not hasattr(self, "crossattention"):
        #         raise ValueError(
        #             f"If `encoder_hidden_states` are passed, {self} has to be instantiated with cross-attention layers"
        #             " by setting `config.add_cross_attention=True`"
        #         )

        #     # cross_attn cached key/values tuple is at positions 3,4 of past_key_value tuple
        #     cross_attn_past_key_value = past_key_value[-2:] if past_key_value is not None else None
        #     cross_attention_outputs = self.crossattention(
        #         attention_output,
        #         attention_mask,
        #         head_mask,
        #         encoder_hidden_states,
        #         encoder_attention_mask,
        #         cross_attn_past_key_value,
        #         output_attentions,
        #     )
        #     attention_output = cross_attention_outputs[0]
        #     outputs = outputs + cross_attention_outputs[1:-1]  # add cross attentions if we output attention weights

        #     # add cross-attn cache to positions 3,4 of present_key_value tuple
        #     cross_attn_present_key_value = cross_attention_outputs[-1]
        #     present_key_value = present_key_value + cross_attn_present_key_value
        layer_output = apply_chunking_to_forward(
            self.feed_forward_chunk, self.chunk_size_feed_forward, self.seq_len_dim, attention_output
        )
        outputs = (layer_output,) + outputs

        return outputs
    
    def feed_forward_chunk(self, attention_output):
        intermediate_output = self.intermediate(attention_output)
        layer_output = self.output(intermediate_output, attention_output)
        return layer_output

       


    def set_quant_state(self, weight_quant: bool = False, act_quant: bool = False):
        # setting weight quantization here does not affect actual forward pass
        self.use_weight_quant = weight_quant
        self.use_act_quant = act_quant
        names = []
        for name, m in self.named_modules():
            if isinstance(m, (QuantLinear, QuantMatMul)):
                names.append(name)
                m.set_quant_state(weight_quant, act_quant)

    @torch.no_grad()
    def smooth_and_quant_inplace(self):
        # return
        if self.let:
            for name, module in self.named_parameters():
                if "smooth_scale" in name:
                    module.data = truncate_number(module)
            smooth_ln_fcs_inplace(self.self_attn_layer_norm,[self.self_attn.q_proj, self.self_attn.k_proj, self.self_attn.v_proj],
                                    self.qkv_smooth_scale,self.qkv_smooth_shift)
            smooth_ln_fcs_inplace(self.final_layer_norm,[self.fc1],
                                    self.fc1_smooth_scale,self.fc1_smooth_shift)
            smooth_fc_fc_inplace(self.self_attn.v_proj,self.self_attn.out_proj,
                                self.out_smooth_scale, self.out_smooth_shift)
            smooth_q_k_inplace(self.self_attn.q_proj, self.self_attn.k_proj,
                                self.qkt_smooth_scale)
        for name, module in self.named_modules():
            if isinstance(module, QuantLinear):
                module.weight = module.weight_quantizer(module.weight)
                module.use_temporary_parameter=False
                

    def clear_temp_variable(self):
        for name, module in self.named_modules():
            if isinstance(module, QuantLinear):
                del module.temp_weight
                del module.temp_bias

    def smooth_and_quant_temporary(self):
        if self.let:
            with torch.no_grad():
                for name, module in self.named_parameters():
                    if "smooth_scale" in name:
                        module.data = truncate_number(module)
            smooth_ln_fcs_temporary(self.self_attn_layer_norm,[self.self_attn.q_proj, self.self_attn.k_proj, self.self_attn.v_proj],
                                    self.qkv_smooth_scale,self.qkv_smooth_shift)
            smooth_ln_fcs_temporary(self.final_layer_norm,[self.fc1],
                                    self.fc1_smooth_scale,self.fc1_smooth_shift)
            smooth_fc_fc_temporary(self.self_attn.v_proj,self.self_attn.out_proj,
                                self.out_smooth_scale, self.out_smooth_shift)
            smooth_q_k_temporary(self.self_attn.q_proj, self.self_attn.k_proj,
                                self.qkt_smooth_scale)
            self.fc2.temp_weight = self.fc2.weight
        else:
            for name, module in self.named_modules():
                if isinstance(module, QuantLinear):
                    module.temp_weight = module.weight
        # quant
        for name, module in self.named_modules():
            if isinstance(module, QuantLinear):
                if hasattr(module, "temp_weight"):
                    module.temp_weight = module.weight_quantizer(module.temp_weight)
                else:
                    module.temp_weight = module.weight_quantizer(module.weight)
                if not hasattr(module, "temp_bias"):
                    module.temp_bias = module.bias
                module.use_temporary_parameter=True


    def let_parameters(self, use_shift=True):
        params = []
        template = "smooth" if use_shift else "smooth_scale"
        for n, m in self.named_parameters():
            if n.find(template) > -1:
                params.append(m)
        return iter(params)  

    def lwc_parameters(self):
        params = []
        for n, m in self.named_parameters():
            if n.find('bound_factor') > -1:
                params.append(m)
        return iter(params)  

    def omni_parameters(self, use_shift=True):
        params = []
        template = "smooth" if use_shift else "smooth_scale"
        for n, m in self.named_parameters():
            if n.find('bound_factor') > -1 or n.find(template) > -1:
                params.append(m)
        return iter(params)  
    
    def omni_state_dict(self, destination=None, prefix='', keep_vars=False):
        if destination is None:
            destination = OrderedDict()
        for name, param in self.named_parameters():
            if name.find('smooth') > -1 or name.find('bound_factor') > -1:
                destination[prefix + name] = param if keep_vars else param.detach()
        return destination
    

    def register_scales_and_zeros(self):
        for name, module in self.named_modules():
            if isinstance(module, QuantLinear):
                module.weight_quantizer.register_scales_and_zeros()
    



class QuantBertLayer_1(nn.Module):
    def __init__(
        self,
        config: BertConfig,
        ori_layer,
        args,
    ):
        super().__init__()
        self.chunk_size_feed_forward = config.chunk_size_feed_forward
        self.seq_len_dim = 1
        
        self.attention = QuantBertAttention_1(
            org_module=ori_layer.attention,
            hidden_size=config.hidden_size,
            num_attention_heads=config.num_attention_heads,
            is_decoder=False,
            args=args,
            position_embedding_type=None,
            config=config
        )


        self.is_decoder = False
        # self.add_cross_attention = config.add_cross_attention
        # self.dropout = config.dropout
        self.intermediate = QuantBertIntermediate(org_module=ori_layer.intermediate, args=args)
        
        self.output = QuantBertOutput_1to11(org_module=ori_layer.output, args=args)
        
        self.type = ori_layer.intermediate.dense.weight.dtype

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.FloatTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        encoder_hidden_states: Optional[torch.FloatTensor] = None,
        encoder_attention_mask: Optional[torch.FloatTensor] = None,
        past_key_value: Optional[Tuple[Tuple[torch.FloatTensor]]] = None,
        output_attentions: Optional[bool] = False,
    ) -> Tuple[torch.Tensor] :
       

        # Self Attention
        self_attn_past_key_value = past_key_value[:2] if past_key_value is not None else None
        self_attention_outputs = self.attention(
            hidden_states,
            attention_mask,
            head_mask,
            output_attentions=output_attentions,
            past_key_value=self_attn_past_key_value,
        )
        attention_output = self_attention_outputs[0]

        # if decoder, the last output is tuple of self-attn cache
        if self.is_decoder:
            outputs = self_attention_outputs[1:-1]
            present_key_value = self_attention_outputs[-1]
        else:
            outputs = self_attention_outputs[1:]  # add self attentions if we output attention weights

        # 일단 cross attention부분은 다 죽여버림.
        # cross_attn_present_key_value = None
        # if self.is_decoder and encoder_hidden_states is not None:
        #     if not hasattr(self, "crossattention"):
        #         raise ValueError(
        #             f"If `encoder_hidden_states` are passed, {self} has to be instantiated with cross-attention layers"
        #             " by setting `config.add_cross_attention=True`"
        #         )

        #     # cross_attn cached key/values tuple is at positions 3,4 of past_key_value tuple
        #     cross_attn_past_key_value = past_key_value[-2:] if past_key_value is not None else None
        #     cross_attention_outputs = self.crossattention(
        #         attention_output,
        #         attention_mask,
        #         head_mask,
        #         encoder_hidden_states,
        #         encoder_attention_mask,
        #         cross_attn_past_key_value,
        #         output_attentions,
        #     )
        #     attention_output = cross_attention_outputs[0]
        #     outputs = outputs + cross_attention_outputs[1:-1]  # add cross attentions if we output attention weights

        #     # add cross-attn cache to positions 3,4 of present_key_value tuple
        #     cross_attn_present_key_value = cross_attention_outputs[-1]
        #     present_key_value = present_key_value + cross_attn_present_key_value
        layer_output = apply_chunking_to_forward(
            self.feed_forward_chunk, self.chunk_size_feed_forward, self.seq_len_dim, attention_output
        )
        outputs = (layer_output,) + outputs

        return outputs
    
    def feed_forward_chunk(self, attention_output):
        intermediate_output = self.intermediate(attention_output)
        layer_output = self.output(intermediate_output, attention_output)
        return layer_output

       


    def set_quant_state(self, weight_quant: bool = False, act_quant: bool = False):
        # setting weight quantization here does not affect actual forward pass
        self.use_weight_quant = weight_quant
        self.use_act_quant = act_quant
        names = []
        for name, m in self.named_modules():
            if isinstance(m, (QuantLinear, QuantMatMul)):
                names.append(name)
                m.set_quant_state(weight_quant, act_quant)

    @torch.no_grad()
    def smooth_and_quant_inplace(self):
        # return
        if self.let:
            for name, module in self.named_parameters():
                if "smooth_scale" in name:
                    module.data = truncate_number(module)
            smooth_ln_fcs_inplace(self.self_attn_layer_norm,[self.self_attn.q_proj, self.self_attn.k_proj, self.self_attn.v_proj],
                                    self.qkv_smooth_scale,self.qkv_smooth_shift)
            smooth_ln_fcs_inplace(self.final_layer_norm,[self.fc1],
                                    self.fc1_smooth_scale,self.fc1_smooth_shift)
            smooth_fc_fc_inplace(self.self_attn.v_proj,self.self_attn.out_proj,
                                self.out_smooth_scale, self.out_smooth_shift)
            smooth_q_k_inplace(self.self_attn.q_proj, self.self_attn.k_proj,
                                self.qkt_smooth_scale)
        for name, module in self.named_modules():
            if isinstance(module, QuantLinear):
                module.weight = module.weight_quantizer(module.weight)
                module.use_temporary_parameter=False
                

    def clear_temp_variable(self):
        for name, module in self.named_modules():
            if isinstance(module, QuantLinear):
                del module.temp_weight
                del module.temp_bias

    def smooth_and_quant_temporary(self):
        if self.let:
            with torch.no_grad():
                for name, module in self.named_parameters():
                    if "smooth_scale" in name:
                        module.data = truncate_number(module)
            smooth_ln_fcs_temporary(self.self_attn_layer_norm,[self.self_attn.q_proj, self.self_attn.k_proj, self.self_attn.v_proj],
                                    self.qkv_smooth_scale,self.qkv_smooth_shift)
            smooth_ln_fcs_temporary(self.final_layer_norm,[self.fc1],
                                    self.fc1_smooth_scale,self.fc1_smooth_shift)
            smooth_fc_fc_temporary(self.self_attn.v_proj,self.self_attn.out_proj,
                                self.out_smooth_scale, self.out_smooth_shift)
            smooth_q_k_temporary(self.self_attn.q_proj, self.self_attn.k_proj,
                                self.qkt_smooth_scale)
            self.fc2.temp_weight = self.fc2.weight
        else:
            for name, module in self.named_modules():
                if isinstance(module, QuantLinear):
                    module.temp_weight = module.weight
        # quant
        for name, module in self.named_modules():
            if isinstance(module, QuantLinear):
                if hasattr(module, "temp_weight"):
                    module.temp_weight = module.weight_quantizer(module.temp_weight)
                else:
                    module.temp_weight = module.weight_quantizer(module.weight)
                if not hasattr(module, "temp_bias"):
                    module.temp_bias = module.bias
                module.use_temporary_parameter=True


    def let_parameters(self, use_shift=True):
        params = []
        template = "smooth" if use_shift else "smooth_scale"
        for n, m in self.named_parameters():
            if n.find(template) > -1:
                params.append(m)
        return iter(params)  

    def lwc_parameters(self):
        params = []
        for n, m in self.named_parameters():
            if n.find('bound_factor') > -1:
                params.append(m)
        return iter(params)  

    def omni_parameters(self, use_shift=True):
        params = []
        template = "smooth" if use_shift else "smooth_scale"
        for n, m in self.named_parameters():
            if n.find('bound_factor') > -1 or n.find(template) > -1:
                params.append(m)
        return iter(params)  
    
    def omni_state_dict(self, destination=None, prefix='', keep_vars=False):
        if destination is None:
            destination = OrderedDict()
        for name, param in self.named_parameters():
            if name.find('smooth') > -1 or name.find('bound_factor') > -1:
                destination[prefix + name] = param if keep_vars else param.detach()
        return destination
    

    def register_scales_and_zeros(self):
        for name, module in self.named_modules():
            if isinstance(module, QuantLinear):
                module.weight_quantizer.register_scales_and_zeros()
