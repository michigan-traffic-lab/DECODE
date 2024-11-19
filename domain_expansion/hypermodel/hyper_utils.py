import torch 
import torch.nn as nn 
from domain_expansion.hypermodel.mlp import MLP

def build_hyper_mlps(c_in, mlp_channels=None, ret_before_act=False, without_norm=False):
    # layers = []
    # num_layers = len(mlp_channels)

    # for k in range(num_layers):
    #     if k + 1 == num_layers and ret_before_act:
    #         layers.append(nn.Linear(c_in, mlp_channels[k], bias=True))
    #     else:
    #         if without_norm:
    #             layers.extend([nn.Linear(c_in, mlp_channels[k], bias=True), nn.ReLU()])
    #         else:
    #             layers.extend(
    #                 [nn.Linear(c_in, mlp_channels[k], bias=False), nn.BatchNorm1d(mlp_channels[k]), nn.ReLU()])
    #         c_in = mlp_channels[k]

    # return nn.Sequential(*layers)
    if without_norm:
        return MLP(n_in=c_in, n_out=mlp_channels[-1], hidden_layers=mlp_channels[:-1], ret_before_act=ret_before_act, activation_fn=torch.nn.ReLU(), no_weights=True)
    else:
        return MLP(n_in=c_in, n_out=mlp_channels[-1], hidden_layers=mlp_channels[:-1], ret_before_act=ret_before_act, use_external_norm=True, external_norm_type="batchnorm", activation_fn=torch.nn.ReLU(), no_weights=True)


def assign_weights(index_mapping, weight_dicts):
    def assign_from_indices(sub_index_mapping):
        if isinstance(sub_index_mapping, list):
            return [weight_dicts[i] for i in sub_index_mapping]
        elif isinstance(sub_index_mapping, dict):
            return {k: assign_from_indices(v) for k, v in sub_index_mapping.items()}
    return assign_from_indices(index_mapping)

def collect_target_shapes(index_mapping, flattened_target_shapes, key):
    if key not in index_mapping:
        return [], []

    def collect_indices(sub_dict):
        indices = []
        for k, v in sub_dict.items():
            if isinstance(v, list):
                indices.extend(v)
            elif isinstance(v, dict):
                indices.extend(collect_indices(v))
        return indices

    target_dict = index_mapping[key]
    if isinstance(target_dict, list):
        return [flattened_target_shapes[i] for i in target_dict], target_dict
    elif isinstance(target_dict, dict):
        indices = collect_indices(target_dict)
        return [flattened_target_shapes[i] for i in indices], indices

def flatten_dictionary(orig_dict):
    flat_list = []
    index_dict = {}

    def flatten(d, idx_dict):
        for k, v in d.items():
            if isinstance(v, list):
                start_index = len(flat_list)
                flat_list.extend(v)
                end_index = len(flat_list)
                idx_dict[k] = list(range(start_index, end_index))
            elif isinstance(v, dict):
                idx_dict[k] = {}
                flatten(v, idx_dict[k])

    flatten(orig_dict, index_dict)
    return flat_list, index_dict


        # # define the cross-attn layers
        # in_proj_center_obj_shapes, self.shape_indices['in_proj_center_obj'] = collect_target_shapes(self.index_mapping, self.flattened_target_shapes, 'in_proj_center_obj')
        # self.in_proj_center_obj = ChunkedHyperNetworkHandler(in_proj_center_obj_shapes, num_tasks, no_te_embs=True, chunk_dim=chunk_dims['in_proj_center_obj'], layers=layers, te_dim=te_dim, activation_fn=activation_fn, use_bias=use_bias, no_weights=no_weights, ce_dim=ce_dim, init_weights=init_weights, dropout_rate=dropout_rate, noise_dim=noise_dim, temb_std=temb_std)

        # in_proj_obj_shapes, self.shape_indices['in_proj_obj'] = collect_target_shapes(self.index_mapping, self.flattened_target_shapes, 'in_proj_obj')
        # self.in_proj_obj = ChunkedHyperNetworkHandler(in_proj_obj_shapes, num_tasks, no_te_embs=True, chunk_dim=chunk_dims['in_proj_obj'], layers=layers, te_dim=te_dim, activation_fn=activation_fn, use_bias=use_bias, no_weights=no_weights, ce_dim=ce_dim, init_weights=init_weights, dropout_rate=dropout_rate, noise_dim=noise_dim, temb_std=temb_std)
        # obj_decoder_layers_shapes, self.shape_indices['obj_decoder_layers'] = collect_target_shapes(self.index_mapping, self.flattened_target_shapes, 'obj_decoder_layers')
        # self.obj_decoder_layers = ChunkedHyperNetworkHandler(obj_decoder_layers_shapes, num_tasks, no_te_embs=True, chunk_dim=chunk_dims['obj_decoder_layers'], layers=layers, te_dim=te_dim, activation_fn=activation_fn, use_bias=use_bias, no_weights=no_weights, ce_dim=ce_dim, init_weights=init_weights, dropout_rate=dropout_rate, noise_dim=noise_dim, temb_std=temb_std)

        # in_proj_map_shapes, self.shape_indices['in_proj_map'] = collect_target_shapes(self.index_mapping, self.flattened_target_shapes, 'in_proj_map')
        # self.in_proj_map = ChunkedHyperNetworkHandler(in_proj_map_shapes, num_tasks, no_te_embs=True, chunk_dim=chunk_dims['in_proj_map'], layers=layers, te_dim=te_dim, activation_fn=activation_fn, use_bias=use_bias, no_weights=no_weights, ce_dim=ce_dim, init_weights=init_weights, dropout_rate=dropout_rate, noise_dim=noise_dim, temb_std=temb_std)
        # map_decoder_layers_shapes, self.shape_indices['map_decoder_layers'] = collect_target_shapes(self.index_mapping, self.flattened_target_shapes, 'map_decoder_layers')
        # self.map_decoder_layers = ChunkedHyperNetworkHandler(map_decoder_layers_shapes, num_tasks, no_te_embs=True, chunk_dim=chunk_dims['map_decoder_layers'], layers=layers, te_dim=te_dim, activation_fn=activation_fn, use_bias=use_bias, no_weights=no_weights, ce_dim=ce_dim, init_weights=init_weights, dropout_rate=dropout_rate, noise_dim=noise_dim, temb_std=temb_std)

        # if 'map_query_content_mlps' in self.flattened_target_shapes:
        #     map_query_content_mlps_shapes, self.shape_indices['map_query_content_mlps'] = collect_target_shapes(self.index_mapping, self.flattened_target_shapes, 'map_query_content_mlps')
        #     self.map_query_content_mlps = ChunkedHyperNetworkHandler(map_query_content_mlps_shapes, num_tasks, no_te_embs=True, chunk_dim=chunk_dims['map_query_content_mlps'], layers=layers, te_dim=te_dim, activation_fn=activation_fn, use_bias=use_bias, no_weights=no_weights, ce_dim=ce_dim, init_weights=init_weights, dropout_rate=dropout_rate, noise_dim=noise_dim, temb_std=temb_std)
            
        #     map_query_embed_mlps_shapes, self.shape_indices['map_query_embed_mlps'] = collect_target_shapes(self.index_mapping, self.flattened_target_shapes, 'map_query_embed_mlps')
        #     self.map_query_embed_mlps = ChunkedHyperNetworkHandler(map_query_embed_mlps_shapes, num_tasks, no_te_embs=True, chunk_dim=chunk_dims['map_query_embed_mlps'], layers=layers, te_dim=te_dim, activation_fn=activation_fn, use_bias=use_bias, no_weights=no_weights, ce_dim=ce_dim, init_weights=init_weights, dropout_rate=dropout_rate, noise_dim=noise_dim, temb_std=temb_std)
        # else:
        #     self.map_query_content_mlps = self.map_query_embed_mlps = None
        # # define the dense future prediction layers
        # obj_pos_encoding_layer_shapes, self.shape_indices['obj_pos_encoding_layer'] = collect_target_shapes(self.index_mapping, self.flattened_target_shapes, 'obj_pos_encoding_layer')
        # self.obj_pos_encoding_layer = ChunkedHyperNetworkHandler(obj_pos_encoding_layer_shapes, num_tasks, no_te_embs=True, chunk_dim=chunk_dims['obj_pos_encoding_layer'], layers=layers, te_dim=te_dim, activation_fn=activation_fn, use_bias=use_bias, no_weights=no_weights, ce_dim=ce_dim, init_weights=init_weights, dropout_rate=dropout_rate, noise_dim=noise_dim, temb_std=temb_std)

        # dense_future_head_shapes, self.shape_indices['dense_future_head'] = collect_target_shapes(self.index_mapping, self.flattened_target_shapes, 'dense_future_head')
        # self.dense_future_head = ChunkedHyperNetworkHandler(dense_future_head_shapes, num_tasks, no_te_embs=True, chunk_dim=chunk_dims['dense_future_head'], layers=layers, te_dim=te_dim, activation_fn=activation_fn, use_bias=use_bias, no_weights=no_weights, ce_dim=ce_dim, init_weights=init_weights, dropout_rate=dropout_rate, noise_dim=noise_dim, temb_std=temb_std)

        # future_traj_mlps_shapes, self.shape_indices['future_traj_mlps'] = collect_target_shapes(self.index_mapping, self.flattened_target_shapes, 'future_traj_mlps')
        # self.future_traj_mlps = ChunkedHyperNetworkHandler(future_traj_mlps_shapes, num_tasks, no_te_embs=True, chunk_dim=chunk_dims['future_traj_mlps'], layers=layers, te_dim=te_dim, activation_fn=activation_fn, use_bias=use_bias, no_weights=no_weights, ce_dim=ce_dim, init_weights=init_weights, dropout_rate=dropout_rate, noise_dim=noise_dim, temb_std=temb_std)

        # traj_fusion_mlps_shapes, self.shape_indices['traj_fusion_mlps'] = collect_target_shapes(self.index_mapping, self.flattened_target_shapes, 'traj_fusion_mlps')
        # self.traj_fusion_mlps = ChunkedHyperNetworkHandler(traj_fusion_mlps_shapes, num_tasks, no_te_embs=True, chunk_dim=chunk_dims['traj_fusion_mlps'], layers=layers, te_dim=te_dim, activation_fn=activation_fn, use_bias=use_bias, no_weights=no_weights, ce_dim=ce_dim, init_weights=init_weights, dropout_rate=dropout_rate, noise_dim=noise_dim, temb_std=temb_std)

        # # define the motion query
        # intention_query_mlps_shapes, self.shape_indices['intention_query_mlps'] = collect_target_shapes(self.index_mapping, self.flattened_target_shapes, 'intention_query_mlps')
        # self.intention_query_mlps = ChunkedHyperNetworkHandler(intention_query_mlps_shapes, num_tasks, no_te_embs=True, chunk_dim=chunk_dims['intention_query_mlps'], layers=layers, te_dim=te_dim, activation_fn=activation_fn, use_bias=use_bias, no_weights=no_weights, ce_dim=ce_dim, init_weights=init_weights, dropout_rate=dropout_rate, noise_dim=noise_dim, temb_std=temb_std)
        # # define the motion head
        # query_feature_fusion_layers_shapes, self.shape_indices['query_feature_fusion_layers'] = collect_target_shapes(self.index_mapping, self.flattened_target_shapes, 'query_feature_fusion_layers')
        # self.query_feature_fusion_layers = ChunkedHyperNetworkHandler(query_feature_fusion_layers_shapes, num_tasks, no_te_embs=True, chunk_dim=chunk_dims['query_feature_fusion_layers'], layers=layers, te_dim=te_dim, activation_fn=activation_fn, use_bias=use_bias, no_weights=no_weights, ce_dim=ce_dim, init_weights=init_weights, dropout_rate=dropout_rate, noise_dim=noise_dim, temb_std=temb_std)
        
        # motion_reg_heads_shapes, self.shape_indices['motion_reg_heads'] = collect_target_shapes(self.index_mapping, self.flattened_target_shapes, 'motion_reg_heads')
        # self.motion_reg_heads = ChunkedHyperNetworkHandler(motion_reg_heads_shapes, num_tasks, no_te_embs=True, chunk_dim=chunk_dims['motion_reg_heads'], layers=layers, te_dim=te_dim, activation_fn=activation_fn, use_bias=use_bias, no_weights=no_weights, ce_dim=ce_dim, init_weights=init_weights, dropout_rate=dropout_rate, noise_dim=noise_dim, temb_std=temb_std)

        # motion_cls_heads_shapes, self.shape_indices['motion_cls_heads'] = collect_target_shapes(self.index_mapping, self.flattened_target_shapes, 'motion_cls_heads')
        # self.motion_cls_heads = ChunkedHyperNetworkHandler(motion_cls_heads_shapes, num_tasks, no_te_embs=True, chunk_dim=chunk_dims['motion_cls_heads'], layers=layers, te_dim=te_dim, activation_fn=activation_fn, use_bias=use_bias, no_weights=no_weights, ce_dim=ce_dim, init_weights=init_weights, dropout_rate=dropout_rate, noise_dim=noise_dim, temb_std=temb_std)