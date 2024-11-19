# Motion Transformer (MTR): https://arxiv.org/abs/2209.13508
# Published at NeurIPS 2022
# Modified by Shaoshuai Shi 
# All Rights Reserved


"""
Modified from https://github.com/IDEA-opensource/DAB-DETR/blob/main/models/DAB_DETR/transformer.py
"""

from typing import Optional

import torch
from torch import nn, Tensor

from domain_expansion.hypermodel.multi_head_attention import MultiheadAttention
from domain_expansion.hypermodel.multi_head_attention_local import MultiheadAttentionLocal
from unitraj.models.mtr.transformer.transformer_encoder_layer import _get_activation_fn
from domain_expansion.hypermodel.mlp import MLP

class hyperTransformerDecoderLayer(nn.Module):

    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1,
                 activation="relu", normalize_before=False, keep_query_pos=True,
                 rm_self_attn_decoder=False, use_local_attn=False):
        super().__init__()
        self._param_shapes = {}
        # Decoder Self-Attention
        if not rm_self_attn_decoder:
            self_attention_params = {}
            self.sa_qcontent_proj = MLP(n_in=d_model, n_out=d_model, hidden_layers=[], activation_fn=None, no_weights=True)
            self_attention_params['sa_qcontent_proj'] = self.sa_qcontent_proj.param_shapes
            self.sa_qpos_proj = MLP(n_in=d_model, n_out=d_model, hidden_layers=[], activation_fn=None, no_weights=True)
            self_attention_params['sa_qpos_proj'] = self.sa_qpos_proj.param_shapes
            self.sa_kcontent_proj = MLP(n_in=d_model, n_out=d_model, hidden_layers=[], activation_fn=None, no_weights=True)
            self_attention_params['sa_kcontent_proj'] = self.sa_kcontent_proj.param_shapes
            self.sa_kpos_proj = MLP(n_in=d_model, n_out=d_model, hidden_layers=[], activation_fn=None, no_weights=True)
            self_attention_params['sa_kpos_proj'] = self.sa_kpos_proj.param_shapes
            self.sa_v_proj = MLP(n_in=d_model, n_out=d_model, hidden_layers=[], activation_fn=None, no_weights=True)
            self_attention_params['sa_v_proj'] = self.sa_v_proj.param_shapes
            self.self_attn = MultiheadAttention(d_model, nhead, dropout=dropout, vdim=d_model, without_weight=True)
            self_attention_params['self_attn'] = self.self_attn.param_shapes
            self.norm1 = ("layernorm", d_model)#nn.LayerNorm(d_model)
            self.dropout1 = nn.Dropout(dropout)
            self._param_shapes['self_attention'] = self_attention_params
        
        # Decoder Cross-Attention
        cross_attention_params = {}
        self.ca_qcontent_proj = MLP(n_in=d_model, n_out=d_model, hidden_layers=[], activation_fn=None, no_weights=True)
        cross_attention_params['ca_qcontent_proj'] = self.ca_qcontent_proj.param_shapes
        self.ca_qpos_proj = MLP(n_in=d_model, n_out=d_model, hidden_layers=[], activation_fn=None, no_weights=True)
        cross_attention_params['ca_qpos_proj'] = self.ca_qpos_proj.param_shapes
        self.ca_kcontent_proj = MLP(n_in=d_model, n_out=d_model, hidden_layers=[], activation_fn=None, no_weights=True)
        cross_attention_params['ca_kcontent_proj'] = self.ca_kcontent_proj.param_shapes
        self.ca_kpos_proj = MLP(n_in=d_model, n_out=d_model, hidden_layers=[], activation_fn=None, no_weights=True)
        cross_attention_params['ca_kpos_proj'] = self.ca_kpos_proj.param_shapes
        self.ca_v_proj = MLP(n_in=d_model, n_out=d_model, hidden_layers=[], activation_fn=None, no_weights=True)
        cross_attention_params['ca_v_proj'] = self.ca_v_proj.param_shapes
        self.ca_qpos_sine_proj = MLP(n_in=d_model, n_out=d_model, hidden_layers=[], activation_fn=None, no_weights=True)
        cross_attention_params['ca_qpos_sine_proj'] = self.ca_qpos_sine_proj.param_shapes

        self.use_local_attn = use_local_attn

        if self.use_local_attn:
            self.cross_attn = MultiheadAttentionLocal(d_model * 2, nhead, dropout=dropout, vdim=d_model,
                                                      without_weight=True)
            cross_attention_params['cross_attn'] = self.cross_attn.param_shapes
        else:
            self.cross_attn = MultiheadAttention(d_model * 2, nhead, dropout=dropout, vdim=d_model, without_weight=True)
            cross_attention_params['cross_attn'] = self.cross_attn.param_shapes
        
        self._param_shapes['cross_attention'] = cross_attention_params
        self.nhead = nhead
        self.rm_self_attn_decoder = rm_self_attn_decoder

        # Implementation of Feedforward model
        feedforward_params = {}
        self.linear1 = MLP(n_in=d_model, n_out=dim_feedforward, hidden_layers=[], activation_fn=None, no_weights=True)
        feedforward_params['linear1'] = self.linear1.param_shapes
        self.dropout = nn.Dropout(dropout)
        self.linear2 = MLP(n_in=dim_feedforward, n_out=d_model, hidden_layers=[], activation_fn=None, no_weights=True)
        feedforward_params['linear2'] = self.linear2.param_shapes
        self._param_shapes['feedforward'] = feedforward_params

        self.norm2 = ("layernorm", d_model)#nn.LayerNorm(d_model)
        self.norm3 = ("layernorm", d_model)#nn.LayerNorm(d_model)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)
        self.normalize_before = normalize_before
        self.keep_query_pos = keep_query_pos

        self._externalnorm_layers = [self.norm1, self.norm2, self.norm3]

    @property
    def param_shapes(self):
        return self._param_shapes
    
    @property
    def externalnorms(self):
        return self._externalnorm_layers
    
    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    def forward(self, tgt, memory, weights, extnorms,
                tgt_mask: Optional[Tensor] = None,
                memory_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None,
                query_pos: Optional[Tensor] = None,
                query_sine_embed=None,
                is_first=False,
                memory_key_padding_mask=None,
                # for local attn
                key_batch_cnt=None,  # (B)
                index_pair=None,  # (N1+N2..., K)
                index_pair_batch=None,  # (N1+N2...),
                memory_valid_mask=None,  # (M1+M2+...)  # TODO DO WE NEED TO MASK OUT INVALID DATA FOR LINEAR LAYER?
                ):
        """

        Args:
            tgt (num_query, B, C):
            memory (M1 + M2 + ..., C):
            weights 
            pos (M1 + M2 + ..., C):
            query_pos (num_query, B, C):
            query_sine_embed (num_query, B, C):
            is_first (bool, optional):

        Returns:
            _type_: _description_
        """
        num_queries, bs, n_model = tgt.shape
        # ========== Begin of Self-Attention =============
        if not self.rm_self_attn_decoder:
            self_attn_weights = weights['self_attention']
            # Apply projections here
            # shape: num_queries x batch_size x 256
            q_content = self.sa_qcontent_proj(tgt, self_attn_weights['sa_qcontent_proj'])  # target is the input of the first decoder layer. zero by default.
            q_pos = self.sa_qpos_proj(query_pos, self_attn_weights['sa_qpos_proj'])
            k_content = self.sa_kcontent_proj(tgt, self_attn_weights['sa_kcontent_proj'])
            k_pos = self.sa_kpos_proj(query_pos, self_attn_weights['sa_kpos_proj'])
            v = self.sa_v_proj(tgt, self_attn_weights['sa_v_proj'])

            num_queries, bs, n_model = q_content.shape
            hw, _, _ = k_content.shape

            q = q_content + q_pos
            k = k_content + k_pos

            tgt2 = self.self_attn(q, k, weights=self_attn_weights['self_attn'], value=v, attn_mask=tgt_mask,
                                  key_padding_mask=None)[0]
            # ========== End of Self-Attention =============

            tgt = tgt + self.dropout1(tgt2)
            # tgt = self.norm1(tgt)
            tgt = extnorms[0](tgt)

        if self.use_local_attn:
            # Transform the queries to stack format
            query_batch_cnt = torch.zeros_like(key_batch_cnt)
            query_batch_cnt.fill_(num_queries)

            query_pos = query_pos.permute(1, 0, 2).contiguous().view(-1, n_model)  # (B * num_q, C)
            query_sine_embed = query_sine_embed.permute(1, 0, 2).contiguous().view(-1, n_model)  # (B * num_q, C)
            tgt = tgt.permute(1, 0, 2).contiguous().view(-1, n_model)  # (B * num_q, C)

        # ========== Begin of Cross-Attention =============
        # Apply projections here
        # shape: num_queries x batch_size x 256
        cross_attn_weights = weights['cross_attention']
        q_content = self.ca_qcontent_proj(tgt, cross_attn_weights['ca_qcontent_proj'])

        if self.use_local_attn and memory_valid_mask is not None:
            valid_memory = memory[memory_valid_mask]

            k_content_valid = self.ca_kcontent_proj(valid_memory, cross_attn_weights['ca_kcontent_proj'])
            k_content = memory.new_zeros(memory.shape[0], k_content_valid.shape[-1])
            k_content[memory_valid_mask] = k_content_valid

            v_valid = self.ca_v_proj(valid_memory, cross_attn_weights['ca_v_proj'])
            v = memory.new_zeros(memory.shape[0], v_valid.shape[-1])
            v[memory_valid_mask] = v_valid

            valid_pos = pos[memory_valid_mask]
            k_pos_valid = self.ca_kpos_proj(valid_pos, cross_attn_weights['ca_kpos_proj'])
            k_pos = pos.new_zeros(memory.shape[0], k_pos_valid.shape[-1])
            k_pos[memory_valid_mask] = k_pos_valid
        else:
            k_content = self.ca_kcontent_proj(memory, cross_attn_weights['ca_kcontent_proj'])
            v = self.ca_v_proj(memory, cross_attn_weights['ca_v_proj'])
            k_pos = self.ca_kpos_proj(pos, cross_attn_weights['ca_kpos_proj'])

        # For the first decoder layer, we concatenate the positional embedding predicted from
        # the object query (the positional embedding) into the original query (key) in DETR.
        if is_first or self.keep_query_pos:
            q_pos = self.ca_qpos_proj(query_pos, cross_attn_weights['ca_qpos_proj'])
            q = q_content + q_pos
            k = k_content + k_pos
        else:
            q = q_content
            k = k_content

        query_sine_embed = self.ca_qpos_sine_proj(query_sine_embed, cross_attn_weights['ca_qpos_sine_proj'])

        if self.use_local_attn:
            num_q_all, n_model = q_content.shape
            num_k_all, _ = k_content.shape

            q = q.view(num_q_all, self.nhead, n_model // self.nhead)
            query_sine_embed = query_sine_embed.view(num_q_all, self.nhead, n_model // self.nhead)
            q = torch.cat([q, query_sine_embed], dim=-1).view(num_q_all, n_model * 2)

            k = k.view(num_k_all, self.nhead, n_model // self.nhead)
            k_pos = k_pos.view(num_k_all, self.nhead, n_model // self.nhead)
            k = torch.cat([k, k_pos], dim=-1).view(num_k_all, n_model * 2)

            assert num_q_all == len(index_pair)

            tgt2 = self.cross_attn(
                query=q, key=k, value=v, weights = cross_attn_weights['cross_attn'],
                index_pair=index_pair, query_batch_cnt=query_batch_cnt, key_batch_cnt=key_batch_cnt,
                index_pair_batch=index_pair_batch,
                attn_mask=None, vdim=n_model
            )[0]
        else:
            num_queries, bs, n_model = q_content.shape
            hw, _, _ = k_content.shape

            q = q.view(num_queries, bs, self.nhead, n_model // self.nhead)
            query_sine_embed = query_sine_embed.view(num_queries, bs, self.nhead, n_model // self.nhead)
            q = torch.cat([q, query_sine_embed], dim=3).view(num_queries, bs, n_model * 2)

            k = k.view(hw, bs, self.nhead, n_model // self.nhead)
            k_pos = k_pos.view(hw, bs, self.nhead, n_model // self.nhead)
            k = torch.cat([k, k_pos], dim=3).view(hw, bs, n_model * 2)

            tgt2 = self.cross_attn(query=q,
                                   key=k,
                                   value=v, weights=cross_attn_weights['cross_attn'], attn_mask=memory_mask,
                                   key_padding_mask=memory_key_padding_mask)[0]
        # ========== End of Cross-Attention =============
        feedforward_weights = weights['feedforward']    
        tgt = tgt + self.dropout2(tgt2)
        # tgt = self.norm2(tgt)
        tgt = extnorms[1](tgt)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt, feedforward_weights['linear1']))), feedforward_weights['linear2'])
        tgt = tgt + self.dropout3(tgt2)
        # tgt = self.norm3(tgt)
        tgt = extnorms[2](tgt)
        return tgt