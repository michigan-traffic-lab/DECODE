# Motion Transformer (MTR): https://arxiv.org/abs/2209.13508
# Published at NeurIPS 2022
# Written by Shaoshuai Shi
# All Rights Reserved


import copy
import pickle
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from easydict import EasyDict
import math
import unitraj.models.mtr.loss_utils as loss_utils
import unitraj.models.mtr.motion_utils as motion_utils
from domain_expansion.model.base_model.base_model import BaseModel
# from unitraj.models.base_model.base_model import BaseModel
from unitraj.models.mtr.MTR_utils import PointNetPolylineEncoder, get_batch_offsets, build_mlps
from domain_expansion.hypermodel.hyper_utils import build_hyper_mlps, flatten_dictionary, assign_weights, collect_target_shapes
from unitraj.models.mtr.ops.knn import knn_utils
from unitraj.models.mtr.transformer import transformer_decoder_layer, position_encoding_utils, transformer_encoder_layer
from domain_expansion.hypermodel.transformer_decoder_layer import hyperTransformerDecoderLayer
from domain_expansion.hypermodel.mlp import MLP
from domain_expansion.hypermodel.chunked_hyper_model import ChunkedHyperNetworkHandler
from domain_expansion.hypermodel.module_wrappers import CLHyperNetInterface
from domain_expansion.model.loss import ContinualLearningLoss
from domain_expansion.detector.flow import MainNormalizingFlow, MainCouplingLayer
import domain_expansion.hypermodel.optim_step as opstep

Type_dict = {0: 'UNSET', 1: 'VEHICLE', 2: 'PEDESTRIAN', 3: 'CYCLIST'}


class HyperMotionTransformer(BaseModel):

    def __init__(self, config):
        super(HyperMotionTransformer, self).__init__(config)
        # self.config = config
        self.model_cfg = EasyDict(config)
        self.pred_dicts = []
        self.training_domain_id = self.model_cfg.domain['training_domain_id']
        self.model_cfg.MOTION_DECODER['CENTER_OFFSET_OF_MAP'] = self.model_cfg['center_offset_of_map']
        self.model_cfg.MOTION_DECODER['NUM_FUTURE_FRAMES'] = self.model_cfg['future_len']
        self.model_cfg.MOTION_DECODER['OBJECT_TYPE'] = self.model_cfg['object_type']
        self.current_num_domains = self.model_cfg.domain['num_domains']
        self.context_encoder = MTREncoder(self.model_cfg.CONTEXT_ENCODER)
        self.mmotion_decoder = mainMTRDecoder(
            in_channels=self.context_encoder.num_out_channels,
            motion_config=self.model_cfg.MOTION_DECODER,
            ood_config=self.model_cfg.DETECTOR
        )
        chunk_dims = {}
        for layer_name in self.mmotion_decoder.param_shapes.keys():
            if "obj_decoder_layers" in layer_name:
                chunk_dims[layer_name] = 2048 * 8#5#10
            elif "map_decoder_layers" in layer_name:
                chunk_dims[layer_name] = 2048 * 4#2#5
            elif "flow" in layer_name:
                chunk_dims[layer_name] = 1280
            else:
                chunk_dims[layer_name] = 2048 * 2#2#5

        self.hmotion_decoder = hyperMTRDecoder(
            target_shapes = self.mmotion_decoder.param_shapes,
            target_norms = self.mmotion_decoder.extnorms,
            num_domains = self.model_cfg.domain['num_domains'],
            te_dim =  self.model_cfg.MOTION_DECODER['te_dim'],
            ce_dim = self.model_cfg.MOTION_DECODER['ce_dim'],
            chunk_dims = chunk_dims,
            layers=self.model_cfg.MOTION_DECODER['H_LAYERS']
        )
        self.continual_loss_func = ContinualLearningLoss() 
        self.automatic_optimization = False
        # load previous domain model
        if self.model_cfg.domain.get('last_domain_model_path', None) is not None:
            checkpoint_model = torch.load(self.model_cfg.domain.last_domain_model_path)
            self.load_state_dict(checkpoint_model['state_dict'])
            print("loading model from", self.model_cfg.domain.last_domain_model_path)
            # self.main_predictor_targets = {
            #     domain_id:self.hmotion_decoder.get_domain_targets(domain_id) for domain_id in range(self.current_num_domains)
            # }

        else:
            # load pre-trained model parameters 
            ckpt = torch.load(self.model_cfg['pretrained_model_path'])
            # self.load_state_dict(ckpt['state_dict']) #TODO change this to load only encoder parameters
            context_encoder_state_dict = {k[len("context_encoder."):]: v
                                        for k, v in ckpt['state_dict'].items()
                                        if k.startswith("context_encoder.")}
            self.context_encoder.load_state_dict(context_encoder_state_dict)
            # Filter the state_dict to only have motion_decoder parameters and ignore layers with 'head' in the name
            hyper_layer_names = []
            for i in range(self.mmotion_decoder.num_hyper_decoder_layers):
                hyper_layer_names.append('obj_decoder_layers.'+str(i+self.mmotion_decoder.num_decoder_layers))
                hyper_layer_names.append('map_decoder_layers.'+str(i+self.mmotion_decoder.num_decoder_layers))
                hyper_layer_names.append('map_query_content_mlps.'+str(i+self.mmotion_decoder.num_decoder_layers))
                hyper_layer_names.append('query_feature_fusion_layers.'+str(i+self.mmotion_decoder.num_decoder_layers))
                hyper_layer_names.append('motion_cls_heads.'+str(i+self.mmotion_decoder.num_decoder_layers))
                hyper_layer_names.append('motion_reg_heads.'+str(i+self.mmotion_decoder.num_decoder_layers))
            # motion_decoder_state_dict = {}
            # for k, v in ckpt['state_dict'].items():
            #     if k.startswith("motion_decoder."):
            #         condition = False 
            #         for layer_name in hyper_layer_names:
            #             if layer_name in k:
            #                 condition = True
            #                 break
            #         if condition:
            #             continue
            #         motion_decoder_state_dict[k[len("motion_decoder."):]] = v
            # self.mmotion_decoder.load_state_dict(motion_decoder_state_dict, strict=False)
            motion_decoder_state_dict = {}
            for k, v in ckpt['state_dict'].items():
                if k.startswith("motion_decoder."):
                    if "map_decoder_layers.5" in k or \
                        "obj_decoder_layers.5" in k or \
                        "map_query_content_mlps.5" in k or \
                        "query_feature_fusion_layers.5" in k or \
                        "motion_cls_heads.5" in k or "motion_reg_heads.5" in k:
                        continue
                    motion_decoder_state_dict[k[len("motion_decoder."):]] = v
            self.mmotion_decoder.load_state_dict(motion_decoder_state_dict, strict=False)
            self.main_predictor_targets = {}
        # freeze the context encoder
        for param in self.context_encoder.parameters():
            param.requires_grad = False
        
        for name, param in self.mmotion_decoder.named_parameters():
            """no need to filter because hyper layers does not store parameters in main model"""
            
            param.requires_grad = False

        if self.training_domain_id >= self.current_num_domains:
            self.create_new_domain()
        
        # freeze previous domain embeddings and norms
        for domain_id in range(self.current_num_domains):
            if domain_id == self.training_domain_id:
                continue
            domain_params = [self.hmotion_decoder.get_domain_emb(domain_id)] + list(self.hmotion_decoder.get_domain_norm(domain_id).parameters())
            for param in domain_params:
                param.requires_grad = False
        # regularized_params = list(self.hmotion_decoder.theta)
        # for param in regularized_params:
        #     param.requires_grad = False 

    def create_new_domain(self):
        self.hmotion_decoder.create_new_domain()
        self.current_num_domains += 1
        # self.main_predictor_targets[self.current_num_domains-1] = self.hmotion_decoder(self.current_num_domains-1)

    def get_latent_features(self, batch, domain_id=None):
        enc_dict = self.context_encoder(batch)
        return enc_dict['center_objects_feature']
    
    def find_max_evidence_domain(self, enc_dict):
        output_domain_id = None
        min_bpd = float('inf')
        log_evidences = []
        for domain_id in range(self.current_num_domains):
            weights = self.hmotion_decoder(domain_id)
            detector_weights = {key:value for key, value in weights.items() if 'flow' in key}
            x = enc_dict['center_objects_feature']
            z, log_det_sum = self.mmotion_decoder.mdetector.encode(x, detector_weights)
            dim = z.size(-1)
            # Compute log-probability
            const = dim * math.log(2 * math.pi)
            norm = torch.einsum("...ij,...ij->...i", z, z)
            normal_log_prob = -0.5 * (const + norm)
            log_prob = normal_log_prob + log_det_sum
            log_evidence = self.mmotion_decoder.mdetector.scaler.forward(log_prob)
            log_evidences.append(log_evidence)

            log_pz = self.mmotion_decoder.mdetector.prior.log_prob(z).sum(dim=-1)
            log_px = log_pz + log_det_sum
            nll = -log_px
            bpd = nll * np.log2(np.e) / np.prod(x.shape[1:])
            if bpd.mean() < min_bpd:
                min_bpd = bpd.mean()
                output_domain_id = domain_id
        log_evidences = torch.stack(log_evidences, dim=-1)
        enc_dict['log_evidences'] = log_evidences
        return output_domain_id
    
    def forward_val(self, batch, pred_targets=None, det_targets=None):
        self.context_encoder.eval()
        self.mmotion_decoder.eval()
        with torch.no_grad():
            enc_dict = self.context_encoder(batch)
        domain_id = self.find_max_evidence_domain(enc_dict)
        if pred_targets is None:
            pred_targets = self.hmotion_decoder(domain_id)
        extnorms = self.hmotion_decoder.get_domain_norm(domain_id)
        out_dict = self.mmotion_decoder(enc_dict, pred_targets, extnorms)
        mode_probs, out_dists = out_dict['pred_list'][-1]
        if not self.training:
            pred_scores, pred_trajs = self.mmotion_decoder.generate_final_prediction(pred_list=out_dict['pred_list'], batch_dict=out_dict)
            out_dict['pred_scores'] = pred_scores
            out_dict['pred_trajs'] = pred_trajs

        output = {}

        if self.training:
            output['predicted_probability'] = mode_probs  # #[B, c]
            output['predicted_trajectory'] = out_dists  # [B, c, T, 5] to be able to parallelize code
        else:
            output['predicted_probability'] = out_dict['pred_scores']  # #[B, c]
            output['predicted_trajectory'] = out_dict['pred_trajs']  # [B, c, T, 5] to be able to parallelize code

        # store the hidden features 
        output['center_objects_feature'] = enc_dict['center_objects_feature']
        output['all_query_contents'] = out_dict['all_query_contents']
        output['bpd'] = out_dict['bpd']
        output['domain_id'] = domain_id
        loss, tb_dict, disp_dict = self.mmotion_decoder.get_loss()
        
        return output, loss

    def forward(self, batch, domain_id=None, pred_targets=None, det_targets=None):
        # important: Do not let the batchnorm update the running statistics
        self.context_encoder.eval()
        self.mmotion_decoder.eval()
        with torch.no_grad():
            enc_dict = self.context_encoder(batch)
        if domain_id is None:
            domain_id = self.training_domain_id
        if pred_targets is None:
            pred_targets = self.hmotion_decoder(domain_id)
        extnorms = self.hmotion_decoder.get_domain_norm(domain_id)
        out_dict = self.mmotion_decoder(enc_dict, pred_targets, extnorms)
        mode_probs, out_dists = out_dict['pred_list'][-1]
        if not self.training:
            pred_scores, pred_trajs = self.mmotion_decoder.generate_final_prediction(pred_list=out_dict['pred_list'], batch_dict=out_dict)
            out_dict['pred_scores'] = pred_scores
            out_dict['pred_trajs'] = pred_trajs

        output = {}

        if self.training:
            output['predicted_probability'] = mode_probs  # #[B, c]
            output['predicted_trajectory'] = out_dists  # [B, c, T, 5] to be able to parallelize code
        else:
            output['predicted_probability'] = out_dict['pred_scores']  # #[B, c]
            output['predicted_trajectory'] = out_dict['pred_trajs']  # [B, c, T, 5] to be able to parallelize code

        # store the hidden features 
        output['center_objects_feature'] = enc_dict['center_objects_feature']
        output['all_query_contents'] = out_dict['all_query_contents']
        output['bpd'] = out_dict['bpd']
        loss, tb_dict, disp_dict = self.mmotion_decoder.get_loss()
        
        return output, loss

    def get_loss(self):
        loss, tb_dict, disp_dict = self.mmotion_decoder.get_loss()
        # hloss = self.hmotion_decoder.get_loss()
        return loss

    def configure_optimizers(self):
        # 
        regularized_params = list(self.hmotion_decoder.theta)
        theta_optimizer = torch.optim.AdamW(regularized_params, lr=self.model_cfg['learning_rate'],
                                      weight_decay=self.model_cfg['weight_decay'])
        domain_params = [self.hmotion_decoder.get_domain_emb(self.training_domain_id)] + list(self.hmotion_decoder.get_domain_norm(self.training_domain_id).parameters())
        domain_optimizer = torch.optim.AdamW(domain_params, lr=self.model_cfg['learning_rate'],
                                        weight_decay=self.model_cfg['weight_decay'])
        decay_steps = [x for x in self.model_cfg['learning_rate_sched']]
        def lr_lbmd(cur_epoch):
            cur_decay = 1
            for decay_step in decay_steps:
                if cur_epoch >= decay_step:
                    cur_decay = cur_decay * self.model_cfg['lr_decay']
            return max(cur_decay, self.model_cfg['lr_clip'] / self.model_cfg['learning_rate'])
        theta_scheduler = torch.optim.lr_scheduler.LambdaLR(theta_optimizer, lr_lbmd, last_epoch=-1, verbose=True)
        domain_scheduler = torch.optim.lr_scheduler.LambdaLR(domain_optimizer, lr_lbmd, last_epoch=-1, verbose=True)
        return [theta_optimizer, domain_optimizer], [theta_scheduler, domain_scheduler]

    def on_train_start(self) -> None:
        self.main_predictor_targets = {}
        for domain_id in range(self.current_num_domains):
            if domain_id == self.training_domain_id:
                continue
            self.main_predictor_targets[domain_id] = self.hmotion_decoder.get_domain_targets(domain_id)
            # self.main_predictor_targets[domain_id] = [W.to(self.device) for W in self.main_predictor_targets[domain_id]]
            
    def training_step(self, batch, batch_idx):
        prediction, loss = self.forward(batch)
        self.compute_official_evaluation(batch, prediction)
        self.log_info(batch, batch_idx, prediction, status='train')
        self.log_hidden(self.mmotion_decoder.forward_ret_dict, batch_idx, status='train')
        #############################
        theta_optimizer, domain_optimizer = self.optimizers()
        theta_optimizer.zero_grad()
        domain_optimizer.zero_grad()
        calc_reg = self.current_num_domains > 1
        self.manual_backward(loss, retain_graph=calc_reg, create_graph=self.model_cfg['backprop_dt'] and calc_reg)
        self.clip_gradients(domain_optimizer, gradient_clip_val=self.model_cfg['grad_clip_norm'], gradient_clip_algorithm='norm')
        domain_optimizer.step()
        if calc_reg:
            dTheta = opstep.calc_delta_theta(theta_optimizer, self.model_cfg['use_sgd_change'], lr=theta_optimizer.param_groups[0]['lr'], detach_dt=not self.model_cfg['backprop_dt'])
            dTembs = None 
            fisher_estimates = None 
            gloss_reg = self.continual_loss_func(self.hmotion_decoder, self.training_domain_id,
                targets=self.main_predictor_targets, dTheta=dTheta, dTembs=dTembs, mnet=self.mmotion_decoder,
                inds_of_out_heads=None,
                fisher_estimates=fisher_estimates, 
                reg_scaling = [1.0, 4.0])
            gloss_reg *= self.model_cfg['beta']
            self.manual_backward(gloss_reg)
            self.log("train/continual_reg", gloss_reg, on_step=False, on_epoch=True, sync_dist=True, batch_size=self.model_cfg['train_batch_size'])
        self.clip_gradients(theta_optimizer, gradient_clip_val=self.model_cfg['grad_clip_norm'], gradient_clip_algorithm='norm')
        theta_optimizer.step()
        # lr schedulers
        theta_scheduler, domain_scheduler = self.lr_schedulers()
        if batch_idx == 0:
            theta_scheduler.step()
            domain_scheduler.step()

    # def on_train_epoch_end(self):
    #     # This method is called at the end of each epoch
    #     current_epoch = self.trainer.current_epoch
    #     if current_epoch > 15:
    #         self.mmotion_decoder.bpd_weight = 1.0
    # def on_after_backward(self):
    #     # Called after the backward pass but before the optimizer step
    #     for name, param in self.named_parameters():
    #         if param.requires_grad and param.grad is None:
    #             print(f"{name} was not used to compute the loss.")

    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        prediction, loss = self.forward(batch, domain_id=dataloader_idx)
        # prediction, loss = self.forward_val(batch)
        self.compute_official_evaluation(batch, prediction)
        self.log_info(batch, batch_idx, prediction, status='val', record_dataloader=(dataloader_idx==self.training_domain_id))
        return loss

    # def configure_optimizers(self):
    #     decay_steps = [x for x in self.config['learning_rate_sched']]

    #     def lr_lbmd(cur_epoch):
    #         cur_decay = 1
    #         for decay_step in decay_steps:
    #             if cur_epoch >= decay_step:
    #                 cur_decay = cur_decay * self.config['lr_decay']
    #         return max(cur_decay, self.config['lr_clip'] / self.config['learning_rate'])

    #     optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, self.parameters()), lr=self.config['learning_rate'],
    #                                   weight_decay=self.config['weight_decay'])

    #     scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lbmd, last_epoch=-1, verbose=True)
    #     return [optimizer], [scheduler]


class MTREncoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.model_cfg = config

        # build polyline encoders
        self.agent_polyline_encoder = self.build_polyline_encoder(
            in_channels=self.model_cfg.NUM_INPUT_ATTR_AGENT + 1,
            hidden_dim=self.model_cfg.NUM_CHANNEL_IN_MLP_AGENT,
            num_layers=self.model_cfg.NUM_LAYER_IN_MLP_AGENT,
            out_channels=self.model_cfg.D_MODEL
        )
        self.map_polyline_encoder = self.build_polyline_encoder(
            in_channels=self.model_cfg.NUM_INPUT_ATTR_MAP,
            hidden_dim=self.model_cfg.NUM_CHANNEL_IN_MLP_MAP,
            num_layers=self.model_cfg.NUM_LAYER_IN_MLP_MAP,
            num_pre_layers=self.model_cfg.NUM_LAYER_IN_PRE_MLP_MAP,
            out_channels=self.model_cfg.D_MODEL
        )

        # build transformer encoder layers
        self.use_local_attn = self.model_cfg.get('USE_LOCAL_ATTN', False)
        self_attn_layers = []
        for _ in range(self.model_cfg.NUM_ATTN_LAYERS):
            self_attn_layers.append(self.build_transformer_encoder_layer(
                d_model=self.model_cfg.D_MODEL,
                nhead=self.model_cfg.NUM_ATTN_HEAD,
                dropout=self.model_cfg.get('DROPOUT_OF_ATTN', 0.1),
                normalize_before=False,
                use_local_attn=self.use_local_attn
            ))

        self.self_attn_layers = nn.ModuleList(self_attn_layers)
        self.num_out_channels = self.model_cfg.D_MODEL

    def build_polyline_encoder(self, in_channels, hidden_dim, num_layers, num_pre_layers=1, out_channels=None):
        ret_polyline_encoder = PointNetPolylineEncoder(
            in_channels=in_channels,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            num_pre_layers=num_pre_layers,
            out_channels=out_channels
        )
        return ret_polyline_encoder

    def build_transformer_encoder_layer(self, d_model, nhead, dropout=0.1, normalize_before=False,
                                        use_local_attn=False):
        single_encoder_layer = transformer_encoder_layer.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=d_model * 4, dropout=dropout,
            normalize_before=normalize_before, use_local_attn=use_local_attn
        )
        return single_encoder_layer

    def apply_global_attn(self, x, x_mask, x_pos):
        """

        Args:
            x (batch_size, N, d_model):
            x_mask (batch_size, N):
            x_pos (batch_size, N, 3):
        """
        assert torch.all(x_mask.sum(dim=-1) > 0)

        batch_size, N, d_model = x.shape
        x_t = x.permute(1, 0, 2)
        x_mask_t = x_mask.permute(1, 0, 2)
        x_pos_t = x_pos.permute(1, 0, 2)

        pos_embedding = position_encoding_utils.gen_sineembed_for_position(x_pos_t, hidden_dim=d_model)

        for k in range(len(self.self_attn_layers)):
            x_t = self.self_attn_layers[k](
                src=x_t,
                src_key_padding_mask=~x_mask_t,
                pos=pos_embedding
            )
        x_out = x_t.permute(1, 0, 2)  # (batch_size, N, d_model)
        return x_out

    def apply_local_attn(self, x, x_mask, x_pos, num_of_neighbors):
        """

        Args:
            x (batch_size, N, d_model):
            x_mask (batch_size, N):
            x_pos (batch_size, N, 3):
        """
        assert torch.all(x_mask.sum(dim=-1) > 0)
        batch_size, N, d_model = x.shape

        x_stack_full = x.view(-1, d_model)  # (batch_size * N, d_model)
        x_mask_stack = x_mask.view(-1)
        x_pos_stack_full = x_pos.view(-1, 3)
        batch_idxs_full = torch.arange(batch_size).type_as(x)[:, None].repeat(1, N).view(-1).int()  # (batch_size * N)

        # filter invalid elements
        x_stack = x_stack_full[x_mask_stack]
        x_pos_stack = x_pos_stack_full[x_mask_stack]
        batch_idxs = batch_idxs_full[x_mask_stack]

        # knn
        batch_offsets = get_batch_offsets(batch_idxs=batch_idxs, bs=batch_size).int()  # (batch_size + 1)
        batch_cnt = batch_offsets[1:] - batch_offsets[:-1]

        index_pair = knn_utils.knn_batch_mlogk(
            x_pos_stack, x_pos_stack, batch_idxs, batch_offsets, num_of_neighbors
        )  # (num_valid_elems, K)

        # positional encoding
        pos_embedding = \
            position_encoding_utils.gen_sineembed_for_position(x_pos_stack[None, :, 0:2], hidden_dim=d_model)[0]

        # local attn
        output = x_stack
        for k in range(len(self.self_attn_layers)):
            output = self.self_attn_layers[k](
                src=output,
                pos=pos_embedding,
                index_pair=index_pair,
                query_batch_cnt=batch_cnt,
                key_batch_cnt=batch_cnt,
                index_pair_batch=batch_idxs
            )

        ret_full_feature = torch.zeros_like(x_stack_full)  # (batch_size * N, d_model)
        ret_full_feature[x_mask_stack] = output

        ret_full_feature = ret_full_feature.view(batch_size, N, d_model)
        return ret_full_feature

    def forward(self, batch_dict):
        """
        Args:
            batch_dict:
              input_dict:
        """
        input_dict = batch_dict['input_dict']
        obj_trajs, obj_trajs_mask = input_dict['obj_trajs'], input_dict['obj_trajs_mask']
        map_polylines, map_polylines_mask = input_dict['map_polylines'], input_dict['map_polylines_mask']

        obj_trajs_last_pos = input_dict['obj_trajs_last_pos']
        map_polylines_center = input_dict['map_polylines_center']
        track_index_to_predict = input_dict['track_index_to_predict']

        assert obj_trajs_mask.dtype == torch.bool and map_polylines_mask.dtype == torch.bool

        num_center_objects, num_objects, num_timestamps, _ = obj_trajs.shape
        num_polylines = map_polylines.shape[1]

        # apply polyline encoder
        obj_trajs_in = torch.cat((obj_trajs, obj_trajs_mask[:, :, :, None].type_as(obj_trajs)), dim=-1)
        obj_polylines_feature = self.agent_polyline_encoder(obj_trajs_in,
                                                            obj_trajs_mask)  # (num_center_objects, num_objects, C)
        map_polylines_feature = self.map_polyline_encoder(map_polylines,
                                                          map_polylines_mask)  # (num_center_objects, num_polylines, C)

        # apply self-attn
        obj_valid_mask = (obj_trajs_mask.sum(dim=-1) > 0)  # (num_center_objects, num_objects)
        map_valid_mask = (map_polylines_mask.sum(dim=-1) > 0)  # (num_center_objects, num_polylines)

        global_token_feature = torch.cat((obj_polylines_feature, map_polylines_feature), dim=1)
        global_token_mask = torch.cat((obj_valid_mask, map_valid_mask), dim=1)
        global_token_pos = torch.cat((obj_trajs_last_pos, map_polylines_center), dim=1)

        if self.use_local_attn:
            global_token_feature = self.apply_local_attn(
                x=global_token_feature, x_mask=global_token_mask, x_pos=global_token_pos,
                num_of_neighbors=self.model_cfg.NUM_OF_ATTN_NEIGHBORS
            )
        else:
            global_token_feature = self.apply_global_attn(
                x=global_token_feature, x_mask=global_token_mask, x_pos=global_token_pos
            )

        obj_polylines_feature = global_token_feature[:, :num_objects]
        map_polylines_feature = global_token_feature[:, num_objects:]
        assert map_polylines_feature.shape[1] == num_polylines

        # organize return features
        center_objects_feature = obj_polylines_feature[torch.arange(num_center_objects), track_index_to_predict]

        batch_dict['center_objects_feature'] = center_objects_feature
        batch_dict['obj_feature'] = obj_polylines_feature
        batch_dict['map_feature'] = map_polylines_feature
        batch_dict['obj_mask'] = obj_valid_mask
        batch_dict['map_mask'] = map_valid_mask
        batch_dict['obj_pos'] = obj_trajs_last_pos
        batch_dict['map_pos'] = map_polylines_center

        return batch_dict

class hyperMTRDecoder(nn.Module, CLHyperNetInterface):
    def __init__(self, target_shapes, target_norms, num_domains, chunk_dims,
                layers=[50, 100], te_dim=8, activation_fn=torch.nn.ReLU(),
                use_bias=True, no_weights=False, ce_dim=None,
                init_weights=None, dropout_rate=-1, noise_dim=-1,
                temb_std=-1):
        nn.Module.__init__(self)
        CLHyperNetInterface.__init__(self)
        self.te_dim = te_dim
        self.shape_indices = {}
        # shared domain embeddings
        self._domain_embs = nn.ParameterList()
        for _ in range(num_domains):
            self._domain_embs.append(nn.Parameter(data=torch.Tensor(te_dim//2),
                                                requires_grad=True))
            torch.nn.init.normal_(self._domain_embs[-1], mean=0., std=1.)
        # shared domain norms
        self._domain_norms_name = target_norms
        self._domain_norms = nn.ModuleList()
        for _ in range(num_domains):
            self._domain_norms.append(nn.ModuleDict())
            for name, norms in target_norms.items():
                normlist = []
                for normtype, hidden_dim in norms:
                    if normtype == 'batchnorm':
                        normlist.append(nn.BatchNorm1d(hidden_dim))
                    elif normtype == 'layernorm':
                        normlist.append(nn.LayerNorm(hidden_dim))
                    else:
                        raise ValueError('Unknown normalization type: %s' % normtype)
                self._domain_norms[-1][name] = nn.ModuleList(normlist)
        self.flattened_target_shapes, self.index_mapping = flatten_dictionary(target_shapes)

        self._thetas, self._theta_shapes = [], []
        self.hyper_layers = nn.ModuleDict()
        for layer_name in target_shapes.keys():
            layer_shapes, self.shape_indices[layer_name] = collect_target_shapes(self.index_mapping, self.flattened_target_shapes, layer_name)
            self.hyper_layers[layer_name] = ChunkedHyperNetworkHandler(layer_shapes, num_domains, no_te_embs=True, chunk_dim=chunk_dims[layer_name], layers=layers, te_dim=te_dim, activation_fn=activation_fn, use_bias=use_bias, no_weights=no_weights, ce_dim=ce_dim, init_weights=init_weights, dropout_rate=dropout_rate, noise_dim=noise_dim, temb_std=temb_std)
            self._thetas.extend(self.hyper_layers[layer_name].theta)
            self._theta_shapes.extend(self.hyper_layers[layer_name].theta_shapes)
        
    
    def create_new_domain(self):
        self._domain_embs.append(nn.Parameter(data=torch.Tensor(self.te_dim//2),
                                                requires_grad=True))
        torch.nn.init.normal_(self._domain_embs[-1], mean=0., std=1.)

        self._domain_norms.append(nn.ModuleDict())
        for name, norms in self._domain_norms_name.items():
            normlist = []
            for normtype, hidden_dim in norms:
                if normtype == 'batchnorm':
                    normlist.append(nn.BatchNorm1d(hidden_dim))
                elif normtype == 'layernorm':
                    normlist.append(nn.LayerNorm(hidden_dim))
                else:
                    raise ValueError('Unknown normalization type: %s' % normtype)
            self._domain_norms[-1][name] = nn.ModuleList(normlist)

    def get_domain_emb(self, domain_id):
        return self._domain_embs[domain_id]
    
    def get_domain_norm(self, domain_id):
        return self._domain_norms[domain_id]
    
    def get_domain_targets(self, domain_id):
        hnet_mode = self.training
        self.eval()
        ret = None

        def collect_W(sub_dict):
            Ws = []
            for k, v in sub_dict.items():
                if isinstance(v, list):
                    Ws.extend(v)
                elif isinstance(v, dict):
                    Ws.extend(collect_W(v))
            return Ws
        with torch.no_grad():
            W = self.forward(domain_id=domain_id)
            W = collect_W(W)
            ret = [d.detach().clone() for d in W]

        self.train(mode=hnet_mode)

        return ret
    
    @property
    def has_theta(self):
        """Getter for read-only attribute ``has_theta``."""
        return True
    
    @property
    def theta(self):
        return self._thetas

    @property
    def theta_shapes(self):
        return self._theta_shapes
    
    def forward(self, domain_id=None, theta=None, dTheta=None, domain_emb=None, ext_inputs=None, squeeze=True):
        weight_dicts = {}
        if domain_emb is None:
            domain_emb = self.get_domain_emb(domain_id)
        last_index = 0
        for layer_name in self.hyper_layers.keys():
            if theta is not None:
                layer_theta = theta[layer_name]
            else:
                layer_theta = None
            if dTheta is not None:
                layer_dTheta = dTheta[last_index:last_index+len(self.hyper_layers[layer_name].theta_shapes)]
            else:
                layer_dTheta = None
            if ext_inputs is not None:
                layer_ext_inputs = ext_inputs[layer_name]
            else:
                layer_ext_inputs = None
            last_index += len(self.hyper_layers[layer_name].theta_shapes)
            weights_flatten = self.hyper_layers[layer_name](domain_id, theta=layer_theta, dTheta=layer_dTheta, task_emb=domain_emb, ext_inputs=layer_ext_inputs)
            for i, w in zip(self.shape_indices[layer_name], weights_flatten):
                weight_dicts[i] = w
        weights = assign_weights(self.index_mapping, weight_dicts)
        return weights

    # def get_loss(self, tb_pre_tag=''):
        
    #     dTheta = opstep.calc_delta_theta(g_th_opt, self.configs.hyper.use_sgd_change, lr=self.configs.train.g_lr, detach_dt=not self.configs.hyper.backprop_dt)
    #     dTembs = None
    #     # Regularizer targets.
    #     if len(self.predictor_fishers[self.training_task_id]) > 0:
    #         fisher_estimates = self.predictor_fishers
    #     else:
    #         fisher_estimates = None
    #     gloss_reg = self.continual_loss_func(self.hpredictor.hyper_generator, self.training_task_id,
    #         targets=self.main_predictor_targets, dTheta=dTheta, dTembs=dTembs, mnet=self.mpredictor.generator,
    #         inds_of_out_heads=None,
    #         fisher_estimates=fisher_estimates)
    #     gloss_reg *= self.configs.hyper.beta

    #     return total_loss, tb_dict, disp_dict

class mainMTRDecoder(nn.Module):
    def __init__(self, in_channels, motion_config, ood_config):
        super().__init__()
        self.model_cfg = motion_config
        self.object_type = self.model_cfg.OBJECT_TYPE
        self.num_future_frames = self.model_cfg.NUM_FUTURE_FRAMES
        self.num_motion_modes = self.model_cfg.NUM_MOTION_MODES
        self.use_place_holder = self.model_cfg.get('USE_PLACE_HOLDER', False)
        self.d_model = self.model_cfg.D_MODEL
        self.num_decoder_layers = self.model_cfg.NUM_DECODER_LAYERS
        self.num_hyper_decoder_layers = self.model_cfg.NUM_HYPER_DECODER_LAYERS
        self.bpd_weight = 10
        self._param_shapes = {}
        self._externnorms = {}
        # define the cross-attn layers
        self.in_proj_center_obj = nn.Sequential(
            nn.Linear(in_channels, self.d_model),
            nn.ReLU(),
            nn.Linear(self.d_model, self.d_model),
        )
        # self.in_proj_center_obj = MLP(n_in=in_channels, n_out=self.d_model, hidden_layers=[self.d_model], activation_fn=torch.nn.ReLU(), no_weights=True)
        # self._param_shapes['in_proj_center_obj'] = self.in_proj_center_obj.param_shapes
        self.in_proj_obj, self.obj_decoder_layers = self.build_transformer_decoder(
            in_channels=in_channels,
            d_model=self.d_model,
            nhead=self.model_cfg.NUM_ATTN_HEAD,
            dropout=self.model_cfg.get('DROPOUT_OF_ATTN', 0.1),
            num_decoder_layers=self.num_decoder_layers,
            num_hyper_decoder_layers=self.num_hyper_decoder_layers, 
            use_local_attn=False
        )
        # self._param_shapes['in_proj_obj'] = self.in_proj_obj.param_shapes
        for i in range(self.num_hyper_decoder_layers):
            self._param_shapes['obj_decoder_layers'+str(i+self.num_decoder_layers)+'self_attention'] = self.obj_decoder_layers[i+self.num_decoder_layers].param_shapes['self_attention']
            self._param_shapes['obj_decoder_layers'+str(i+self.num_decoder_layers)+'cross_attention'] = self.obj_decoder_layers[i+self.num_decoder_layers].param_shapes['cross_attention']
            self._param_shapes['obj_decoder_layers'+str(i+self.num_decoder_layers)+'feedforward'] = self.obj_decoder_layers[i+self.num_decoder_layers].param_shapes['feedforward']
            
            self._externnorms['obj_decoder_layers'+str(i+self.num_decoder_layers)] = self.obj_decoder_layers[i+self.num_decoder_layers].externalnorms
        
        map_d_model = self.model_cfg.get('MAP_D_MODEL', self.d_model)
        self.in_proj_map, self.map_decoder_layers = self.build_transformer_decoder(
            in_channels=in_channels,
            d_model=map_d_model,
            nhead=self.model_cfg.NUM_ATTN_HEAD,
            dropout=self.model_cfg.get('DROPOUT_OF_ATTN', 0.1),
            num_decoder_layers=self.num_decoder_layers,
            num_hyper_decoder_layers=self.num_hyper_decoder_layers, 
            use_local_attn=True
        )
        # self._param_shapes['in_proj_map'] = self.in_proj_map.param_shapes
        for i in range(self.num_hyper_decoder_layers):
            self._param_shapes['map_decoder_layers'+str(i+self.num_decoder_layers)+'self_attention'] = self.map_decoder_layers[i+self.num_decoder_layers].param_shapes['self_attention']
            self._param_shapes['map_decoder_layers'+str(i+self.num_decoder_layers)+'cross_attention'] = self.map_decoder_layers[i+self.num_decoder_layers].param_shapes['cross_attention']
            self._param_shapes['map_decoder_layers'+str(i+self.num_decoder_layers)+'feedforward'] = self.map_decoder_layers[i+self.num_decoder_layers].param_shapes['feedforward']
            self._externnorms['map_decoder_layers'+str(i+self.num_decoder_layers)] = self.map_decoder_layers[i+self.num_decoder_layers].externalnorms

        if map_d_model != self.d_model:
            temp_layer = nn.Linear(self.d_model, map_d_model)
            hyper_temp_layer = MLP(n_in=self.d_model, n_out=map_d_model, hidden_layers=[], activation_fn=None, no_weights=True)
            map_query_content_mlps = []
            for _ in range(self.num_decoder_layers):
                map_query_content_mlps.append(copy.deepcopy(temp_layer))
            for _ in range(self.num_hyper_decoder_layers):
                map_query_content_mlps.append(copy.deepcopy(hyper_temp_layer))
            self.map_query_content_mlps = nn.ModuleList(map_query_content_mlps)
            for i in range(self.num_hyper_decoder_layers):
                self._param_shapes['map_query_content_mlps'+str(i+self.num_decoder_layers)] = self.map_query_content_mlps[i+self.num_decoder_layers].param_shapes  
            # self.map_query_embed_mlps = MLP(n_in=self.d_model, n_out=map_d_model, hidden_layers=[], activation_fn=None, no_weights=True)
            # self._param_shapes['map_query_embed_mlps'] = self.map_query_embed_mlps.param_shapes
            self.map_query_embed_mlps = nn.Linear(self.d_model, map_d_model)
        else:
            self.map_query_content_mlps = self.map_query_embed_mlps = None

        # define the dense future prediction layers
        self.build_dense_future_prediction_layers(
            hidden_dim=self.d_model, num_future_frames=self.num_future_frames
        )

        # define the motion query
        self.intention_points, self.intention_query, self.intention_query_mlps = self.build_motion_query(
            self.d_model, use_place_holder=self.use_place_holder
        )
        # self._param_shapes['intention_query_mlps'] = self.intention_query_mlps.param_shapes
        # self._externnorms['intention_query_mlps'] = self.intention_query_mlps.externalnorms

        # define the motion head
        temp_layer = build_mlps(c_in=self.d_model * 2 + map_d_model, mlp_channels=[self.d_model, self.d_model], ret_before_act=True)
        # hyper_temp_layer = build_hyper_mlps(c_in=self.d_model + map_d_model, mlp_channels=[self.d_model, self.d_model//2], ret_before_act=True)
        hyper_temp_layer = build_hyper_mlps(c_in=self.d_model * 2 + map_d_model, mlp_channels=[self.d_model//2], ret_before_act=True)
        query_feature_fusion_layers = []
        for _ in range(self.num_decoder_layers):
            query_feature_fusion_layers.append(copy.deepcopy(temp_layer))
        for _ in range(self.num_hyper_decoder_layers):
            query_feature_fusion_layers.append(copy.deepcopy(hyper_temp_layer))
        self.query_feature_fusion_layers = nn.ModuleList(query_feature_fusion_layers)
        for i in range(self.num_hyper_decoder_layers):
            self._param_shapes['query_feature_fusion_layers'+str(i+self.num_decoder_layers)] = self.query_feature_fusion_layers[i+self.num_decoder_layers].param_shapes
            self._externnorms['query_feature_fusion_layers'+str(i+self.num_decoder_layers)] = self.query_feature_fusion_layers[i+self.num_decoder_layers].externalnorms
        self.motion_reg_heads, self.motion_cls_heads, self.motion_vel_heads = self.build_motion_head(
            in_channels=self.d_model, hidden_size=self.d_model, num_decoder_layers=self.num_decoder_layers, num_hyper_decoder_layers=self.num_hyper_decoder_layers
        )
        for i in range(self.num_hyper_decoder_layers):
            self._param_shapes['motion_reg_heads'+str(i+self.num_decoder_layers)] = self.motion_reg_heads[i+self.num_decoder_layers].param_shapes 
            self._param_shapes['motion_cls_heads'+str(i+self.num_decoder_layers)] = self.motion_cls_heads[i+self.num_decoder_layers].param_shapes 

            self._externnorms['motion_reg_heads'+str(i+self.num_decoder_layers)] = self.motion_reg_heads[i+self.num_decoder_layers].externalnorms
            self._externnorms['motion_cls_heads'+str(i+self.num_decoder_layers)] = self.motion_cls_heads[i+self.num_decoder_layers].externalnorms
        
        self.mdetector = self.build_ood_flow(ood_config)
        
        for key, value in self.mdetector.param_shapes.items():
            self._param_shapes[key] = value
        self.forward_ret_dict = {}

    @property
    def param_shapes(self):
        return self._param_shapes
    @property
    def extnorms(self):
        return self._externnorms
    
    def build_ood_flow(self, ood_config):
        flow_layers = []
        for i in range(ood_config.n_layers):
            flow_layers.append(
                MainCouplingLayer(
                    input_dim = ood_config.c_in,
                    invert = (i % 2 == 1)
                )
            )
        flow_model = MainNormalizingFlow(flow_layers, ood_config)
        return flow_model

    def build_dense_future_prediction_layers(self, hidden_dim, num_future_frames):
        self.obj_pos_encoding_layer = build_mlps(
            c_in=2, mlp_channels=[hidden_dim, hidden_dim, hidden_dim], ret_before_act=True, without_norm=True
        )
        # self._param_shapes['obj_pos_encoding_layer'] = self.obj_pos_encoding_layer.param_shapes
        self.dense_future_head = build_mlps(
            c_in=hidden_dim * 2,
            mlp_channels=[hidden_dim, hidden_dim, num_future_frames * 7], ret_before_act=True
        )
        # self._param_shapes['dense_future_head'] = self.dense_future_head.param_shapes
        # self._externnorms['dense_future_head'] = self.dense_future_head.externalnorms
        self.future_traj_mlps = build_mlps(
            c_in=4 * self.num_future_frames, mlp_channels=[hidden_dim, hidden_dim, hidden_dim], ret_before_act=True,
            without_norm=True
        )
        # self._param_shapes['future_traj_mlps'] = self.future_traj_mlps.param_shapes
        self.traj_fusion_mlps = build_mlps(
            c_in=hidden_dim * 2, mlp_channels=[hidden_dim, hidden_dim, hidden_dim], ret_before_act=True,
            without_norm=True
        )
        # self._param_shapes['traj_fusion_mlps'] = self.traj_fusion_mlps.param_shapes

    def build_transformer_decoder(self, in_channels, d_model, nhead, dropout=0.1, num_decoder_layers=1, num_hyper_decoder_layers=1,
                                  use_local_attn=False):
        in_proj_layer = nn.Sequential(
            nn.Linear(in_channels, d_model),
            nn.ReLU(),
            nn.Linear(d_model, d_model),
        )
        # in_proj_layer = MLP(n_in=in_channels, n_out=d_model, hidden_layers=[d_model], activation_fn=torch.nn.ReLU(), no_weights=True)

        decoder_layer = transformer_decoder_layer.TransformerDecoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=d_model * 4, dropout=dropout,
            activation="relu", normalize_before=False, keep_query_pos=True,
            rm_self_attn_decoder=False, use_local_attn=use_local_attn
        )
        hyper_decoder_layer = hyperTransformerDecoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=d_model * 4, dropout=dropout,
            activation="relu", normalize_before=False, keep_query_pos=True,
            rm_self_attn_decoder=False, use_local_attn=use_local_attn
        )
        decoder_layers = []
        for _ in range(num_decoder_layers):
            decoder_layers.append(copy.deepcopy(decoder_layer))
        for _ in range(num_hyper_decoder_layers):
            decoder_layers.append(copy.deepcopy(hyper_decoder_layer))
        decoder_layers = nn.ModuleList(decoder_layers)
        return in_proj_layer, decoder_layers

    def build_motion_query(self, d_model, use_place_holder=False):
        intention_points = intention_query = intention_query_mlps = None

        if use_place_holder:
            raise NotImplementedError
        else:
            intention_points_file = self.model_cfg.INTENTION_POINTS_FILE
            with open(intention_points_file, 'rb') as f:
                intention_points_dict = pickle.load(f)

            intention_points = {}
            for cur_type in self.object_type:
                cur_intention_points = intention_points_dict[cur_type]
                cur_intention_points = torch.from_numpy(cur_intention_points).float().view(-1, 2)
                intention_points[cur_type] = cur_intention_points

            intention_query_mlps = build_mlps(
                c_in=d_model, mlp_channels=[d_model, d_model], ret_before_act=True
            )
        return intention_points, intention_query, intention_query_mlps

    def build_motion_head(self, in_channels, hidden_size, num_decoder_layers, num_hyper_decoder_layers):
        motion_reg_head = build_mlps(
            c_in=in_channels,
            mlp_channels=[hidden_size, hidden_size, self.num_future_frames * 7], ret_before_act=True
        )
        motion_cls_head = build_mlps(
            c_in=in_channels,
            mlp_channels=[hidden_size, hidden_size, 1], ret_before_act=True
        )

        # hyper_motion_reg_head = build_hyper_mlps(
        #     c_in=in_channels,
        #     mlp_channels=[hidden_size, hidden_size, self.num_future_frames * 7], ret_before_act=True
        # )
        # hyper_motion_cls_head = build_hyper_mlps(
        #     c_in=in_channels,
        #     mlp_channels=[hidden_size, hidden_size, 1], ret_before_act=True
        # )
        hyper_motion_reg_head = build_hyper_mlps(
            c_in=in_channels//2,
            mlp_channels=[hidden_size//2, self.num_future_frames * 7], ret_before_act=True
        )
        hyper_motion_cls_head = build_hyper_mlps(
            c_in=in_channels//2,
            mlp_channels=[hidden_size//2, 1], ret_before_act=True
        )
        motion_reg_heads, motion_cls_heads = [], []
        for _ in range(num_decoder_layers):
            motion_reg_heads.append(copy.deepcopy(motion_reg_head))
            motion_cls_heads.append(copy.deepcopy(motion_cls_head))
        for _ in range(num_hyper_decoder_layers):
            motion_reg_heads.append(copy.deepcopy(hyper_motion_reg_head))
            motion_cls_heads.append(copy.deepcopy(hyper_motion_cls_head))

        motion_reg_heads = nn.ModuleList(motion_reg_heads)
        motion_cls_heads = nn.ModuleList(motion_cls_heads)
        motion_vel_heads = None
        return motion_reg_heads, motion_cls_heads, motion_vel_heads

    def apply_dense_future_prediction(self, obj_feature, obj_mask, obj_pos):
        num_center_objects, num_objects, _ = obj_feature.shape

        # dense future prediction
        obj_pos_valid = obj_pos[obj_mask][..., 0:2]
        obj_feature_valid = obj_feature[obj_mask]
        obj_pos_feature_valid = self.obj_pos_encoding_layer(obj_pos_valid)
        obj_fused_feature_valid = torch.cat((obj_pos_feature_valid, obj_feature_valid), dim=-1)

        pred_dense_trajs_valid = self.dense_future_head(obj_fused_feature_valid)
        pred_dense_trajs_valid = pred_dense_trajs_valid.view(pred_dense_trajs_valid.shape[0], self.num_future_frames, 7)

        temp_center = pred_dense_trajs_valid[:, :, 0:2] + obj_pos_valid[:, None, 0:2]
        pred_dense_trajs_valid = torch.cat((temp_center, pred_dense_trajs_valid[:, :, 2:]), dim=-1)

        # future feature encoding and fuse to past obj_feature
        obj_future_input_valid = pred_dense_trajs_valid[:, :, [0, 1, -2, -1]].flatten(start_dim=1,
                                                                                      end_dim=2)  # (num_valid_objects, C)
        obj_future_feature_valid = self.future_traj_mlps(obj_future_input_valid)

        obj_full_trajs_feature = torch.cat((obj_feature_valid, obj_future_feature_valid), dim=-1)
        obj_feature_valid = self.traj_fusion_mlps(obj_full_trajs_feature)

        ret_obj_feature = torch.zeros_like(obj_feature)
        ret_obj_feature[obj_mask] = obj_feature_valid

        ret_pred_dense_future_trajs = obj_feature.new_zeros(num_center_objects, num_objects, self.num_future_frames, 7)
        ret_pred_dense_future_trajs[obj_mask] = pred_dense_trajs_valid
        self.forward_ret_dict['pred_dense_trajs'] = ret_pred_dense_future_trajs

        return ret_obj_feature, ret_pred_dense_future_trajs

    def get_motion_query(self, center_objects_type, weights, extnorms):
        num_center_objects = len(center_objects_type)
        if self.use_place_holder:
            raise NotImplementedError
        else:
            intention_points = torch.stack([
                self.intention_points[Type_dict[center_objects_type[obj_idx]]]
                for obj_idx in range(num_center_objects)], dim=0).cuda()
            intention_points = intention_points.permute(1, 0, 2)  # (num_query, num_center_objects, 2)

            intention_query = position_encoding_utils.gen_sineembed_for_position(intention_points,
                                                                                 hidden_dim=self.d_model)
            intention_query = self.intention_query_mlps(intention_query.view(-1, self.d_model)).view(-1,
                                                                                                     num_center_objects,
                                                                                                     self.d_model)  # (num_query, num_center_objects, C)
        return intention_query, intention_points

    def apply_cross_attention(self, kv_feature, kv_mask, kv_pos, query_content, query_embed, attention_layer, attention_layer_weights=None, attention_layer_extnorms=None,
                              dynamic_query_center=None, layer_idx=0, use_local_attn=False, query_index_pair=None,
                              query_content_pre_mlp=None, query_embed_pre_mlp=None, query_content_weights=None):
        """
        Args:
            kv_feature (B, N, C):
            kv_mask (B, N):
            kv_pos (B, N, 3):
            query_tgt (M, B, C):
            query_embed (M, B, C):
            dynamic_query_center (M, B, 2): . Defaults to None.
            attention_layer (layer):

            query_index_pair (B, M, K)

        Returns:
            attended_features: (B, M, C)
            attn_weights:
        """
        if query_content_pre_mlp is not None:
            query_content = query_content_pre_mlp(query_content, query_content_weights) if query_content_weights is not None else query_content_pre_mlp(query_content)
        if query_embed_pre_mlp is not None:
            query_embed = query_embed_pre_mlp(query_embed)

        num_q, batch_size, d_model = query_content.shape
        searching_query = position_encoding_utils.gen_sineembed_for_position(dynamic_query_center, hidden_dim=d_model)
        kv_pos = kv_pos.permute(1, 0, 2)[:, :, 0:2]
        kv_pos_embed = position_encoding_utils.gen_sineembed_for_position(kv_pos, hidden_dim=d_model)

        if not use_local_attn:
            if attention_layer_weights is None:
                query_feature = attention_layer(
                    tgt=query_content,
                    query_pos=query_embed,
                    query_sine_embed=searching_query,
                    memory=kv_feature.permute(1, 0, 2),
                    memory_key_padding_mask=~kv_mask,
                    pos=kv_pos_embed,
                    is_first=(layer_idx == 0)
                )  # (M, B, C)
            else:
                query_feature = attention_layer(
                tgt=query_content,
                query_pos=query_embed,
                weights=attention_layer_weights, 
                extnorms=attention_layer_extnorms,
                query_sine_embed=searching_query,
                memory=kv_feature.permute(1, 0, 2),
                memory_key_padding_mask=~kv_mask,
                pos=kv_pos_embed,
                is_first=(layer_idx == 0)
            )  # (M, B, C)
        else:
            batch_size, num_kv, _ = kv_feature.shape

            kv_feature_stack = kv_feature.flatten(start_dim=0, end_dim=1)
            kv_pos_embed_stack = kv_pos_embed.permute(1, 0, 2).contiguous().flatten(start_dim=0, end_dim=1)
            kv_mask_stack = kv_mask.view(-1)

            key_batch_cnt = num_kv * torch.ones(batch_size).int().to(kv_feature.device)
            query_index_pair = query_index_pair.view(batch_size * num_q, -1)
            index_pair_batch = torch.arange(batch_size).type_as(key_batch_cnt)[:, None].repeat(1, num_q).view(
                -1)  # (batch_size * num_q)
            assert len(query_index_pair) == len(index_pair_batch)
            if attention_layer_weights is None:
                query_feature = attention_layer(
                    tgt=query_content,
                    query_pos=query_embed,
                    query_sine_embed=searching_query,
                    memory=kv_feature_stack,
                    memory_valid_mask=kv_mask_stack,
                    pos=kv_pos_embed_stack,
                    is_first=(layer_idx == 0),
                    key_batch_cnt=key_batch_cnt,
                    index_pair=query_index_pair,
                    index_pair_batch=index_pair_batch
                )
            else:
                query_feature = attention_layer(
                    tgt=query_content,
                    query_pos=query_embed,
                    weights=attention_layer_weights, 
                    extnorms=attention_layer_extnorms,
                    query_sine_embed=searching_query,
                    memory=kv_feature_stack,
                    memory_valid_mask=kv_mask_stack,
                    pos=kv_pos_embed_stack,
                    is_first=(layer_idx == 0),
                    key_batch_cnt=key_batch_cnt,
                    index_pair=query_index_pair,
                    index_pair_batch=index_pair_batch
                )
            query_feature = query_feature.view(batch_size, num_q, d_model).permute(1, 0, 2)  # (M, B, C)

        return query_feature

    def apply_dynamic_map_collection(self, map_pos, map_mask, pred_waypoints, base_region_offset, num_query,
                                     num_waypoint_polylines=128, num_base_polylines=256, base_map_idxs=None):
        map_pos = map_pos.clone()
        map_pos[~map_mask] = 10000000.0
        num_polylines = map_pos.shape[1]

        if base_map_idxs is None:
            base_points = torch.tensor(base_region_offset).type_as(map_pos)
            base_dist = (map_pos[:, :, 0:2] - base_points[None, None, :]).norm(
                dim=-1)  # (num_center_objects, num_polylines)
            base_topk_dist, base_map_idxs = base_dist.topk(k=min(num_polylines, num_base_polylines), dim=-1,
                                                           largest=False)  # (num_center_objects, topk)
            base_map_idxs[base_topk_dist > 10000000] = -1
            base_map_idxs = base_map_idxs[:, None, :].repeat(1, num_query,
                                                             1)  # (num_center_objects, num_query, num_base_polylines)
            if base_map_idxs.shape[-1] < num_base_polylines:
                base_map_idxs = F.pad(base_map_idxs, pad=(0, num_base_polylines - base_map_idxs.shape[-1]),
                                      mode='constant', value=-1)

        dynamic_dist = (pred_waypoints[:, :, None, :, 0:2] - map_pos[:, None, :, None, 0:2]).norm(
            dim=-1)  # (num_center_objects, num_query, num_polylines, num_timestamps)
        dynamic_dist = dynamic_dist.min(dim=-1)[0]  # (num_center_objects, num_query, num_polylines)

        dynamic_topk_dist, dynamic_map_idxs = dynamic_dist.topk(k=min(num_polylines, num_waypoint_polylines), dim=-1,
                                                                largest=False)
        dynamic_map_idxs[dynamic_topk_dist > 10000000] = -1
        if dynamic_map_idxs.shape[-1] < num_waypoint_polylines:
            dynamic_map_idxs = F.pad(dynamic_map_idxs, pad=(0, num_waypoint_polylines - dynamic_map_idxs.shape[-1]),
                                     mode='constant', value=-1)

        collected_idxs = torch.cat((base_map_idxs, dynamic_map_idxs),
                                   dim=-1)  # (num_center_objects, num_query, num_collected_polylines)

        # remove duplicate indices
        sorted_idxs = collected_idxs.sort(dim=-1)[0]
        duplicate_mask_slice = (sorted_idxs[..., 1:] - sorted_idxs[...,
                                                       :-1] != 0)  # (num_center_objects, num_query, num_collected_polylines - 1)
        duplicate_mask = torch.ones_like(collected_idxs).bool()
        duplicate_mask[..., 1:] = duplicate_mask_slice
        sorted_idxs[~duplicate_mask] = -1

        return sorted_idxs.int(), base_map_idxs

    def apply_transformer_decoder(self, center_objects_feature, center_objects_type, obj_feature, obj_mask, obj_pos,
                                  map_feature, map_mask, map_pos, weights, extnorms):
        intention_query, intention_points = self.get_motion_query(center_objects_type, weights, extnorms)
        query_content = torch.zeros_like(intention_query)
        self.forward_ret_dict['intention_points'] = intention_points.permute(1, 0,
                                                                             2)  # (num_center_objects, num_query, 2)

        num_center_objects = query_content.shape[1]
        num_query = query_content.shape[0]

        center_objects_feature = center_objects_feature[None, :, :].repeat(num_query, 1,
                                                                           1)  # (num_query, num_center_objects, C)

        base_map_idxs = None
        pred_waypoints = intention_points.permute(1, 0, 2)[:, :, None, :]  # (num_center_objects, num_query, 1, 2)
        dynamic_query_center = intention_points

        pred_list = []
        all_query_contents = []
        for layer_idx in range(self.num_decoder_layers+self.num_hyper_decoder_layers):
            # query object feature
            if layer_idx < self.num_decoder_layers:
                obj_query_feature = self.apply_cross_attention(
                    kv_feature=obj_feature, kv_mask=obj_mask, kv_pos=obj_pos,
                    query_content=query_content, query_embed=intention_query,
                    attention_layer=self.obj_decoder_layers[layer_idx],
                    dynamic_query_center=dynamic_query_center,
                    layer_idx=layer_idx
                )
            else:
                ow = {'self_attention': weights['obj_decoder_layers'+str(layer_idx)+'self_attention'],
                     'cross_attention': weights['obj_decoder_layers'+str(layer_idx)+'cross_attention'],
                     'feedforward': weights['obj_decoder_layers'+str(layer_idx)+'feedforward']}
                obj_query_feature = self.apply_cross_attention(
                    kv_feature=obj_feature, kv_mask=obj_mask, kv_pos=obj_pos,
                    query_content=query_content, query_embed=intention_query,
                    attention_layer=self.obj_decoder_layers[layer_idx],
                    dynamic_query_center=dynamic_query_center,
                    layer_idx=layer_idx, 
                    attention_layer_weights=ow, 
                    attention_layer_extnorms=extnorms['obj_decoder_layers'+str(layer_idx)]
                )

            # query map feature
            collected_idxs, base_map_idxs = self.apply_dynamic_map_collection(
                map_pos=map_pos, map_mask=map_mask,
                pred_waypoints=pred_waypoints,
                base_region_offset=self.model_cfg.CENTER_OFFSET_OF_MAP,
                num_waypoint_polylines=self.model_cfg.NUM_WAYPOINT_MAP_POLYLINES,
                num_base_polylines=self.model_cfg.NUM_BASE_MAP_POLYLINES,
                base_map_idxs=base_map_idxs,
                num_query=num_query
            )
            if layer_idx < self.num_decoder_layers:
                map_query_feature = self.apply_cross_attention(
                    kv_feature=map_feature, kv_mask=map_mask, kv_pos=map_pos,
                    query_content=query_content, query_embed=intention_query,
                    attention_layer=self.map_decoder_layers[layer_idx],
                    layer_idx=layer_idx,
                    dynamic_query_center=dynamic_query_center,
                    use_local_attn=True,
                    query_index_pair=collected_idxs,
                    query_content_pre_mlp=self.map_query_content_mlps[layer_idx],
                    query_embed_pre_mlp=self.map_query_embed_mlps
                )
            else:
                mw = {'self_attention': weights['map_decoder_layers'+str(layer_idx)+'self_attention'],
                        'cross_attention': weights['map_decoder_layers'+str(layer_idx)+'cross_attention'],
                        'feedforward': weights['map_decoder_layers'+str(layer_idx)+'feedforward']}
                map_query_feature = self.apply_cross_attention(
                    kv_feature=map_feature, kv_mask=map_mask, kv_pos=map_pos,
                    query_content=query_content, query_embed=intention_query,
                    attention_layer=self.map_decoder_layers[layer_idx],
                    layer_idx=layer_idx,
                    dynamic_query_center=dynamic_query_center,
                    use_local_attn=True,
                    query_index_pair=collected_idxs,
                    query_content_pre_mlp=self.map_query_content_mlps[layer_idx],
                    query_embed_pre_mlp=self.map_query_embed_mlps,
                    attention_layer_weights=mw, 
                    attention_layer_extnorms=extnorms['map_decoder_layers'+str(layer_idx)],
                    query_content_weights=weights['map_query_content_mlps'+str(layer_idx)],
                )

            query_feature = torch.cat([center_objects_feature, obj_query_feature, map_query_feature], dim=-1)
            if layer_idx < self.num_decoder_layers:
                query_content = self.query_feature_fusion_layers[layer_idx](
                query_feature.flatten(start_dim=0, end_dim=1)).view(num_query, num_center_objects, -1)
            else:
                # query_feature = torch.cat([center_objects_feature+obj_query_feature, map_query_feature], dim=-1)
                query_content = self.query_feature_fusion_layers[layer_idx](
                    query_feature.flatten(start_dim=0, end_dim=1), weights['query_feature_fusion_layers'+str(layer_idx)], extnorms['query_feature_fusion_layers'+str(layer_idx)]
                ).view(num_query, num_center_objects, -1)
                self.forward_ret_dict['query_feature'+str(layer_idx)] = query_feature
                self.forward_ret_dict['query_content'+str(layer_idx)] = query_content

            # motion prediction
            query_content_t = query_content.permute(1, 0, 2).contiguous().view(num_center_objects * num_query, -1)
            if layer_idx < self.num_decoder_layers:
                pred_scores = self.motion_cls_heads[layer_idx](query_content_t).view(num_center_objects, num_query)
            else:
                pred_scores = self.motion_cls_heads[layer_idx](query_content_t, 
                                                           weights['motion_cls_heads'+str(layer_idx)],
                                                           extnorms['motion_cls_heads'+str(layer_idx)]).view(num_center_objects, num_query)
            if self.motion_vel_heads is not None:
                if layer_idx < self.num_decoder_layers:
                    pred_trajs = self.motion_reg_heads[layer_idx](query_content_t).view(num_center_objects, num_query,
                                                                                        self.num_future_frames, 5)
                    pred_vel = self.motion_vel_heads[layer_idx](query_content_t).view(num_center_objects, num_query,
                                                                                    self.num_future_frames, 2)
                else:
                    pred_trajs = self.motion_reg_heads[layer_idx](query_content_t, 
                                                                weights['motion_reg_heads'+str(layer_idx)],
                                                                extnorms['motion_reg_heads'+str(layer_idx)]).view(num_center_objects, num_query,
                                                                                        self.num_future_frames, 5)
                    temp_center = torch.cumsum(pred_trajs[:, :, :, 0:2], dim=2)
                    pred_trajs = torch.cat((temp_center, pred_trajs[:, :, :, 2:]), dim=-1)
                    pred_vel = self.motion_vel_heads[layer_idx](query_content_t, 
                                                                weights['motion_vel_heads'+str(layer_idx)],
                                                                extnorms['motion_vel_heads'+str(layer_idx)]).view(num_center_objects, num_query,
                                                                                    self.num_future_frames, 2)
                pred_trajs = torch.cat((pred_trajs, pred_vel), dim=-1)
            else:
                if layer_idx < self.num_decoder_layers:
                    pred_trajs = self.motion_reg_heads[layer_idx](query_content_t).view(num_center_objects, num_query,
                                                                                        self.num_future_frames, 7)
                else:
                    pred_trajs = self.motion_reg_heads[layer_idx](query_content_t,
                                                                weights['motion_reg_heads'+str(layer_idx)],
                                                                extnorms['motion_reg_heads'+str(layer_idx)]).view(num_center_objects, num_query,
                                                                                        self.num_future_frames, 7)
                    temp_center = torch.cumsum(pred_trajs[:, :, :, 0:2], dim=2)
                    pred_trajs = torch.cat((temp_center, pred_trajs[:, :, :, 2:]), dim=-1)
            all_query_contents.append(query_content_t)
            pred_list.append([pred_scores, pred_trajs])

            # update
            pred_waypoints = pred_trajs[:, :, :, 0:2]
            dynamic_query_center = pred_trajs[:, :, -1, 0:2].contiguous().permute(1, 0,
                                                                                  2)  # (num_query, num_center_objects, 2)
        self.forward_ret_dict['all_query_contents'] = all_query_contents
        if self.use_place_holder:
            raise NotImplementedError

        assert len(pred_list) == self.num_decoder_layers+self.num_hyper_decoder_layers
        return pred_list

    def get_decoder_loss(self, tb_pre_tag=''):
        center_gt_trajs = self.forward_ret_dict['center_gt_trajs']
        center_gt_trajs_mask = self.forward_ret_dict['center_gt_trajs_mask']
        center_gt_final_valid_idx = self.forward_ret_dict['center_gt_final_valid_idx'].long()
        assert center_gt_trajs.shape[-1] == 4

        pred_list = self.forward_ret_dict['pred_list']
        intention_points = self.forward_ret_dict['intention_points']  # (num_center_objects, num_query, 2)

        num_center_objects = center_gt_trajs.shape[0]
        center_gt_goals = center_gt_trajs[torch.arange(num_center_objects), center_gt_final_valid_idx,
                          0:2]  # (num_center_objects, 2)

        if not self.use_place_holder:
            dist = (center_gt_goals[:, None, :] - intention_points).norm(dim=-1)  # (num_center_objects, num_query)
            center_gt_positive_idx = dist.argmin(dim=-1)  # (num_center_objects)
        else:
            raise NotImplementedError

        tb_dict = {}
        disp_dict = {}
        total_loss = 0
        for layer_idx in range(self.num_decoder_layers+self.num_hyper_decoder_layers):
            # if layer_idx < self.num_decoder_layers:
            #     continue
            if self.use_place_holder:
                raise NotImplementedError

            pred_scores, pred_trajs = pred_list[layer_idx]
            assert pred_trajs.shape[-1] == 7
            pred_trajs_gmm, pred_vel = pred_trajs[:, :, :, 0:5], pred_trajs[:, :, :, 5:7]

            loss_reg_gmm, center_gt_positive_idx = loss_utils.nll_loss_gmm_direct(
                pred_scores=pred_scores, pred_trajs=pred_trajs_gmm,
                gt_trajs=center_gt_trajs[:, :, 0:2], gt_valid_mask=center_gt_trajs_mask,
                pre_nearest_mode_idxs=center_gt_positive_idx,
                timestamp_loss_weight=None, use_square_gmm=False,
            )

            pred_vel = pred_vel[torch.arange(num_center_objects), center_gt_positive_idx]
            loss_reg_vel = F.l1_loss(pred_vel, center_gt_trajs[:, :, 2:4], reduction='none')
            loss_reg_vel = (loss_reg_vel * center_gt_trajs_mask[:, :, None]).sum(dim=-1).sum(dim=-1)

            # loss_cls = F.cross_entropy(input=pred_scores, target=center_gt_positive_idx, reduction='none')

            # total loss
            weight_cls = self.model_cfg.LOSS_WEIGHTS.get('cls', 1.0)
            weight_reg = self.model_cfg.LOSS_WEIGHTS.get('reg', 1.0)
            weight_vel = self.model_cfg.LOSS_WEIGHTS.get('vel', 0.2)

            layer_loss = loss_reg_gmm * weight_reg + loss_reg_vel * weight_vel #+ loss_cls.sum(dim=-1) * weight_cls
            layer_loss = layer_loss.mean()
            total_loss += layer_loss
            tb_dict[f'{tb_pre_tag}loss_layer{layer_idx}'] = layer_loss.item()
            tb_dict[f'{tb_pre_tag}loss_layer{layer_idx}_reg_gmm'] = loss_reg_gmm.mean().item() * weight_reg
            tb_dict[f'{tb_pre_tag}loss_layer{layer_idx}_reg_vel'] = loss_reg_vel.mean().item() * weight_vel
            # tb_dict[f'{tb_pre_tag}loss_layer{layer_idx}_cls'] = loss_cls.mean().item() * weight_cls

            if layer_idx + 1 == self.num_decoder_layers+self.num_hyper_decoder_layers:
                layer_tb_dict_ade = motion_utils.get_ade_of_each_category(
                    pred_trajs=pred_trajs_gmm[:, :, :, 0:2],
                    gt_trajs=center_gt_trajs[:, :, 0:2], gt_trajs_mask=center_gt_trajs_mask,
                    object_types=self.forward_ret_dict['center_objects_type'],
                    valid_type_list=self.object_type,
                    post_tag=f'_layer_{layer_idx}',
                    pre_tag=tb_pre_tag
                )
                tb_dict.update(layer_tb_dict_ade)
                disp_dict.update(layer_tb_dict_ade)

        total_loss = total_loss / (self.num_decoder_layers+self.num_hyper_decoder_layers) #self.num_hyper_decoder_layers
        return total_loss, tb_dict, disp_dict

    def get_dense_future_prediction_loss(self, tb_pre_tag='', tb_dict=None, disp_dict=None):
        obj_trajs_future_state = self.forward_ret_dict['obj_trajs_future_state']
        obj_trajs_future_mask = self.forward_ret_dict['obj_trajs_future_mask']
        pred_dense_trajs = self.forward_ret_dict[
            'pred_dense_trajs']  # (num_center_objects, num_objects, num_future_frames, 7)
        assert pred_dense_trajs.shape[-1] == 7
        assert obj_trajs_future_state.shape[-1] == 4

        pred_dense_trajs_gmm, pred_dense_trajs_vel = pred_dense_trajs[:, :, :, 0:5], pred_dense_trajs[:, :, :, 5:7]

        loss_reg_vel = F.l1_loss(pred_dense_trajs_vel, obj_trajs_future_state[:, :, :, 2:4], reduction='none')
        loss_reg_vel = (loss_reg_vel * obj_trajs_future_mask[:, :, :, None]).sum(dim=-1).sum(dim=-1)

        num_center_objects, num_objects, num_timestamps, _ = pred_dense_trajs.shape
        fake_scores = pred_dense_trajs.new_zeros((num_center_objects, num_objects)).view(-1,
                                                                                         1)  # (num_center_objects * num_objects, 1)

        temp_pred_trajs = pred_dense_trajs_gmm.contiguous().view(num_center_objects * num_objects, 1, num_timestamps, 5)
        temp_gt_idx = torch.zeros(num_center_objects * num_objects).long()  # (num_center_objects * num_objects)
        temp_gt_trajs = obj_trajs_future_state[:, :, :, 0:2].contiguous().view(num_center_objects * num_objects,
                                                                               num_timestamps, 2)
        temp_gt_trajs_mask = obj_trajs_future_mask.view(num_center_objects * num_objects, num_timestamps)
        loss_reg_gmm, _ = loss_utils.nll_loss_gmm_direct(
            pred_scores=fake_scores, pred_trajs=temp_pred_trajs, gt_trajs=temp_gt_trajs,
            gt_valid_mask=temp_gt_trajs_mask,
            pre_nearest_mode_idxs=temp_gt_idx,
            timestamp_loss_weight=None, use_square_gmm=False,
        )
        loss_reg_gmm = loss_reg_gmm.view(num_center_objects, num_objects)

        loss_reg = loss_reg_vel + loss_reg_gmm

        obj_valid_mask = obj_trajs_future_mask.sum(dim=-1) > 0

        loss_reg = (loss_reg * obj_valid_mask.float()).sum(dim=-1) / torch.clamp_min(obj_valid_mask.sum(dim=-1),
                                                                                     min=1.0)
        loss_reg = loss_reg.mean()

        if tb_dict is None:
            tb_dict = {}
        if disp_dict is None:
            disp_dict = {}

        tb_dict[f'{tb_pre_tag}loss_dense_prediction'] = loss_reg.item()
        return loss_reg, tb_dict, disp_dict

    def get_loss(self, tb_pre_tag=''):
        loss_decoder, tb_dict, disp_dict = self.get_decoder_loss(tb_pre_tag=tb_pre_tag)
        loss_dense_prediction, tb_dict, disp_dict = self.get_dense_future_prediction_loss(tb_pre_tag=tb_pre_tag,
                                                                                          tb_dict=tb_dict,
                                                                                          disp_dict=disp_dict)
        detector_loss = self.mdetector.get_loss(self.forward_ret_dict)
        bpd = self.forward_ret_dict['bpd']
        total_loss = loss_decoder + loss_dense_prediction + 10 * (detector_loss + bpd.mean())
        tb_dict[f'{tb_pre_tag}loss'] = total_loss.item()
        disp_dict[f'{tb_pre_tag}loss'] = total_loss.item()

        return total_loss, tb_dict, disp_dict

    def generate_final_prediction(self, pred_list, batch_dict):
        pred_scores, pred_trajs = pred_list[-1]
        # pred_scores = torch.softmax(pred_scores, dim=-1)  # (num_center_objects, num_query)

        num_center_objects, num_query, num_future_timestamps, num_feat = pred_trajs.shape
        if self.num_motion_modes != num_query:
            assert num_query > self.num_motion_modes
            pred_trajs_final, pred_scores_final, selected_idxs = motion_utils.batch_nms(
                pred_trajs=pred_trajs, pred_scores=pred_scores,
                dist_thresh=self.model_cfg.NMS_DIST_THRESH,
                num_ret_modes=self.num_motion_modes
            )
        else:
            pred_trajs_final = pred_trajs
            pred_scores_final = pred_scores

        return pred_scores_final, pred_trajs_final

    def forward(self, batch_dict, weights, extnorms, single=False):
        input_dict = batch_dict['input_dict']
        obj_feature, obj_mask, obj_pos = batch_dict['obj_feature'], batch_dict['obj_mask'], batch_dict['obj_pos']
        map_feature, map_mask, map_pos = batch_dict['map_feature'], batch_dict['map_mask'], batch_dict['map_pos']
        center_objects_feature = batch_dict['center_objects_feature']
        num_center_objects, num_objects, _ = obj_feature.shape
        num_polylines = map_feature.shape[1]

        # input projection
        center_objects_feature = self.in_proj_center_obj(center_objects_feature)
        obj_feature_valid = self.in_proj_obj(obj_feature[obj_mask])
        obj_feature = obj_feature.new_zeros(num_center_objects, num_objects, obj_feature_valid.shape[-1])
        obj_feature[obj_mask] = obj_feature_valid

        map_feature_valid = self.in_proj_map(map_feature[map_mask])
        map_feature = map_feature.new_zeros(num_center_objects, num_polylines, map_feature_valid.shape[-1])
        map_feature[map_mask] = map_feature_valid

        # dense future prediction
        obj_feature, pred_dense_future_trajs = self.apply_dense_future_prediction(
            obj_feature=obj_feature, obj_mask=obj_mask, obj_pos=obj_pos
        )
        # decoder layers
        pred_list = self.apply_transformer_decoder(
            center_objects_feature=center_objects_feature,
            center_objects_type=input_dict['center_objects_type'],
            obj_feature=obj_feature, obj_mask=obj_mask, obj_pos=obj_pos,
            map_feature=map_feature, map_mask=map_mask, map_pos=map_pos,
            weights=weights, extnorms=extnorms
        )
        # get the evidence posterior
        detector_weights = {key:value for key, value in weights.items() if 'flow' in key}
        if extnorms.training or single:
            mode_posteriors, log_probs, bpd = self.mdetector.posteriors(batch_dict, pred_list[-1][0], detector_weights)
            pred_list[-1] = [mode_posteriors.maximum_a_posteriori().expected_sufficient_statistics(), pred_list[-1][1]]
        else:
            mode_posteriors, log_probs, bpd = self.mdetector.posteriors2(batch_dict, pred_list[-1][0], pred_list[-2][0], detector_weights)
            pred_list[-1] = [mode_posteriors.maximum_a_posteriori().expected_sufficient_statistics(), 
                                         torch.cat([pred_list[-1][1], 
                                                    pred_list[-2][1]], dim=1)]
            
        self.forward_ret_dict['pred_list'] = pred_list
        batch_dict['pred_list'] = pred_list

        self.forward_ret_dict['center_gt_trajs'] = input_dict['center_gt_trajs']
        self.forward_ret_dict['center_gt_trajs_mask'] = input_dict['center_gt_trajs_mask']
        self.forward_ret_dict['center_gt_final_valid_idx'] = input_dict['center_gt_final_valid_idx']
        self.forward_ret_dict['obj_trajs_future_state'] = input_dict['obj_trajs_future_state']
        self.forward_ret_dict['obj_trajs_future_mask'] = input_dict['obj_trajs_future_mask']
        self.forward_ret_dict['bpd'] = bpd
        self.forward_ret_dict['mode_posteriors'] = mode_posteriors

        self.forward_ret_dict['center_objects_type'] = input_dict['center_objects_type']

        
        batch_dict['all_query_contents'] = self.forward_ret_dict['all_query_contents']
        batch_dict['bpd'] = bpd
        return batch_dict
