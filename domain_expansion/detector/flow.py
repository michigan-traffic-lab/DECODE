import torch
import torch.nn as nn
import pytorch_lightning as pl
import numpy as np
from torch import optim
import time 
from tqdm import tqdm
from domain_expansion.hypermodel.mlp import MLP
from domain_expansion.hypermodel.chunked_hyper_model import ChunkedHyperNetworkHandler
from domain_expansion.hypermodel.module_wrappers import CLHyperNetInterface
# from domain_expansion.hypermodel.hyperSMCTT import assign_weights, collect_target_shapes, flatten_dictionary
import math 

def create_flow(configs):
    flow_layers = []
    for i in range(configs.model.n_layers):
        flow_layers.append(
            CouplingLayer(
                input_dim = configs.model.c_in,
                invert = (i % 2 == 1)
            )
        )
    flow_model = NormalizingFlow(flow_layers, configs)
    return flow_model

def create_main_flow(configs):
    flow_layers = []
    for i in range(configs.n_layers):
        flow_layers.append(
            MainCouplingLayer(
                input_dim = configs.c_in,
                invert = (i % 2 == 1)
            )
        )
    flow_model = MainNormalizingFlow(flow_layers, configs)
    return flow_model

class MainCouplingLayer(nn.Module):
    def __init__(self, input_dim, invert=False):
        """Main Coupling layer inside a normalizing flow.

        Args:
            network: A PyTorch nn.Module constituting the deep neural network for mu and sigma.
                      Output shape should be twice the channel size as the input.
            mask: Binary mask (0 or 1) where 0 denotes that the element should be transformed,
                   while 1 means the latent will be used as input to the NN.
            c_in: Number of input channels
        """
        super().__init__()
        self.input_dim = input_dim
        # Define the scale and translation networks
        self.scale_net = MLP(n_in=input_dim//2, n_out= input_dim // 2, hidden_layers=[128], activation_fn=torch.nn.ReLU(), no_weights=True)
        self.translate_net = MLP(n_in=input_dim//2, n_out= input_dim // 2, hidden_layers=[128], activation_fn=torch.nn.ReLU(), no_weights=True)
        self.invert = invert
        # self.scaling_factor = nn.Parameter(torch.zeros(input_dim // 2))
        self._param_shapes = {"scale_net": self.scale_net.param_shapes, 
                              "translate_net": self.translate_net.param_shapes, 
                              "scaling_factor": [[input_dim // 2]]}
    
    @property
    def param_shapes(self):
        return self._param_shapes
    
    def forward(self, z, ldj, weights, reverse=False):
        """Forward.

        Args:
            z: Latent input to the flow
            ldj:
                The current ldj of the previous flows. The ldj of this layer will be added to this tensor.
            reverse: If True, we apply the inverse of the layer.
            orig_img:
                Only needed in VarDeq. Allows external input to condition the flow on (e.g. original image)
        """
        # Apply network to masked input
        if self.invert:
            z1, z2 = z[:, :self.input_dim // 2], z[:, self.input_dim // 2:]
        else:
            z2, z1 = z[:, :self.input_dim // 2], z[:, self.input_dim // 2:]
        
        s = self.scale_net(z1, weights['scale_net'])
        t = self.translate_net(z1, weights['translate_net'])
        
        # Stabilize scaling output
        s_fac = weights['scaling_factor'][0].exp().view(1, -1)
        s = torch.tanh(s / s_fac) * s_fac


        # Affine transformation
        if not reverse:
            # Whether we first shift and then scale, or the other way round,
            # is a design choice, and usually does not have a big impact
            z2 = (z2 + t) * torch.exp(s)
            ldj += s.sum(dim=1)
        else:
            z2 = (z2 * torch.exp(-s)) - t
            ldj -= s.sum(dim=[1, 2, 3])
        z = torch.cat([z1, z2], dim=1) if self.invert else torch.cat([z2, z1], dim=1)
        return z, ldj
    
class CouplingLayer(nn.Module):
    def __init__(self, input_dim, invert=False):
        """Coupling layer inside a normalizing flow.

        Args:
            network: A PyTorch nn.Module constituting the deep neural network for mu and sigma.
                      Output shape should be twice the channel size as the input.
            mask: Binary mask (0 or 1) where 0 denotes that the element should be transformed,
                   while 1 means the latent will be used as input to the NN.
            c_in: Number of input channels
        """
        super().__init__()
        self.input_dim = input_dim
        # Define the scale and translation networks
        self.scale_net = nn.Sequential(
            nn.Linear(input_dim // 2, 128),
            nn.ReLU(),
            nn.Linear(128, input_dim // 2)
        )
        self.translate_net = nn.Sequential(
            nn.Linear(input_dim // 2, 128),
            nn.ReLU(),
            nn.Linear(128, input_dim // 2)
        )
        self.invert = invert
        self.scaling_factor = nn.Parameter(torch.zeros(input_dim // 2))
        
    
    def forward(self, z, ldj, reverse=False):
        """Forward.

        Args:
            z: Latent input to the flow
            ldj:
                The current ldj of the previous flows. The ldj of this layer will be added to this tensor.
            reverse: If True, we apply the inverse of the layer.
            orig_img:
                Only needed in VarDeq. Allows external input to condition the flow on (e.g. original image)
        """
        # Apply network to masked input
        if self.invert:
            z1, z2 = z[:, :self.input_dim // 2], z[:, self.input_dim // 2:]
        else:
            z2, z1 = z[:, :self.input_dim // 2], z[:, self.input_dim // 2:]
        
        s = self.scale_net(z1)
        t = self.translate_net(z1)
        
        # Stabilize scaling output
        s_fac = self.scaling_factor.exp().view(1, -1)
        s = torch.tanh(s / s_fac) * s_fac


        # Affine transformation
        if not reverse:
            # Whether we first shift and then scale, or the other way round,
            # is a design choice, and usually does not have a big impact
            z2 = (z2 + t) * torch.exp(s)
            ldj += s.sum(dim=1)
        else:
            z2 = (z2 * torch.exp(-s)) - t
            ldj -= s.sum(dim=[1, 2, 3])
        z = torch.cat([z1, z2], dim=1) if self.invert else torch.cat([z2, z1], dim=1)
        return z, ldj

class NormalizingFlow(pl.LightningModule):
    def __init__(self, flows, configs):
        super().__init__()
        self.flows = nn.ModuleList(flows)
        self.prior = torch.distributions.normal.Normal(loc=0.0, scale=1.0)
        self.configs = configs
        # epoch stats
        self.start_epoch = 0
        self.start_time = None
        self.epoch_loss = []

    def forward(self, x):
        return self._get_likelihood(x)
    
    def encode(self, x):
        z, ldj = x, torch.zeros(x.size(0), device=self.device)
        for flow in self.flows:
            z, ldj = flow(z, ldj, reverse=False)
        return z, ldj

    def _get_likelihood(self, x, return_ll=False):
        z, ldj = self.encode(x)
        log_pz = self.prior.log_prob(z).sum(dim=-1)
        log_px = log_pz + ldj
        nll = -log_px
        bpd = nll * np.log2(np.e) / np.prod(x.shape[1:])
        return bpd.mean() if not return_ll else nll
    
    def predict_ood_score(self, x):
        # with torch.no_grad():
        return self._get_likelihood(x, return_ll=True)

    def get_distance(self, x):
        # Encode x to get its latent representation z
        z, _ = self.encode(x)
        # Since the mean is 0, the difference is just z itself
        # And assuming an isotropic Gaussian, the covariance matrix is I, simplifying the computation
        # Mahalanobis distance in this context reduces to the Euclidean distance
        distance = torch.sqrt(torch.sum(z**2, dim=1))
        return distance
            
    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=self.configs.train.lr)
        # An scheduler is optional, but can help in flows to get the last bpd improvement
        scheduler = optim.lr_scheduler.StepLR(optimizer, self.configs.train.lr_scheduler.step_size, gamma=self.configs.train.lr_scheduler.gamma)
        return [optimizer], [scheduler]

    def on_train_start(self) -> None:
        print('on_train_start')
        self.start_epoch = self.current_epoch
        self.start_time = time.time()

    def training_step(self, batch, batch_idx):
        # Normalizing flows are trained by maximum likelihood => return bpd
        loss = self._get_likelihood(batch)
        self.log("train_bpd", loss, prog_bar=True, batch_size=self.configs.train.batch_size)
        return loss
    
    def validation_step(self, batch, batch_idx):
        loss = self._get_likelihood(batch)
        self.log("val_bpd", loss, prog_bar=True, batch_size=self.configs.train.batch_size)
        self.epoch_loss.append(loss.item())

    def on_validation_epoch_end(self):
        avg_loss = np.mean(self.epoch_loss)
        self.epoch_loss.clear()

        if self.start_time is not None:
            time_spent = time.time() - self.start_time
            epoch_left = self.configs.train.epochs - self.current_epoch
            time_ETA = int(time_spent / (self.current_epoch - self.start_epoch + 1) * epoch_left)
            hours, rem_seconds = divmod(time_ETA, 3600)
            minutes = rem_seconds // 60
            tqdm.write("Training ETA: {:02}h {:02}m | Val @ epoch {}: Loss {:.2f}".format(hours, minutes, self.current_epoch, avg_loss))

import domain_expansion.natural.distributions as D
from domain_expansion.natural.scaler import EvidenceScaler, CertaintyBudget
from domain_expansion.natural.output import CategoricalOutput
from domain_expansion.natural.loss import BayesianLoss

class MainNormalizingFlow(pl.LightningModule):
    def __init__(self, flows, configs):
        super().__init__()
        self.flows = nn.ModuleList(flows)
        self.prior = torch.distributions.normal.Normal(loc=0.0, scale=1.0)
        self.configs = configs
        self._param_shapes = {}
        for i, flow in enumerate(flows):
            self._param_shapes[f'flow_{i}'] = flow.param_shapes
        self.output = CategoricalOutput(configs.num_classes)
        self.scaler = EvidenceScaler(configs.c_in, "normal")
        self.loss = BayesianLoss(configs.entropy_weight)

    @property
    def param_shapes(self):
        return self._param_shapes
    
    def forward(self, x, weights):
        return self._get_likelihood(x, weights)
    
    def encode(self, x, weights):
        z, ldj = x, torch.zeros(x.size(0), device=self.device)
        for i, flow in enumerate(self.flows):
            z, ldj = flow(z, ldj, weights[f'flow_{i}'],  reverse=False)
        return z, ldj

    def _get_likelihood(self, x, weights, return_ll=False):
        z, ldj = self.encode(x, weights)
        log_pz = self.prior.log_prob(z).sum(dim=-1)
        log_px = log_pz + ldj
        nll = -log_px
        bpd = nll * np.log2(np.e) / np.prod(x.shape[1:])
        return bpd.mean() if not return_ll else nll
    
    def posteriors(self, enc_dict, mode_probs, weights):
        x = enc_dict['center_objects_feature']
        z, log_det_sum = self.encode(x, weights)
        dim = z.size(-1)

        # Compute log-probability
        const = dim * math.log(2 * math.pi)
        norm = torch.einsum("...ij,...ij->...i", z, z)
        normal_log_prob = -0.5 * (const + norm)
        log_prob = normal_log_prob + log_det_sum
        log_evidence = self.scaler.forward(log_prob).detach()

        mode_output = self.output.forward(mode_probs)#D.Categorical(mode_probs.log_softmax(dim=-1))
        sufficient_statistics = mode_output.expected_sufficient_statistics()
        
        posterior_update = D.PosteriorUpdate(sufficient_statistics, log_evidence)
        mode_posteriors = self.output.prior.update(posterior_update)

        log_pz = self.prior.log_prob(z).sum(dim=-1)
        log_px = log_pz + log_det_sum
        nll = -log_px
        bpd = nll * np.log2(np.e) / np.prod(x.shape[1:])
        return mode_posteriors, log_prob, bpd
    
    def posteriors2(self, enc_dict, mode_probs, prior_mode_probs, weights):
        assert self.training is False
        x = enc_dict['center_objects_feature']
        z, log_det_sum = self.encode(x, weights)
        dim = z.size(-1)

        # Compute log-probability
        const = dim * math.log(2 * math.pi)
        norm = torch.einsum("...ij,...ij->...i", z, z)
        normal_log_prob = -0.5 * (const + norm)
        log_prob = normal_log_prob + log_det_sum
        log_evidence = self.scaler.forward(log_prob)#.detach()

        mode_output = self.output.forward(mode_probs)#D.Categorical(mode_probs.log_softmax(dim=-1))
        sufficient_statistics = mode_output.expected_sufficient_statistics()
        sufficient_statistics = torch.cat([sufficient_statistics, torch.zeros_like(sufficient_statistics)], dim=-1)
        prior_sufficient_statistics = torch.softmax(prior_mode_probs, dim=-1)
        prior_sufficient_statistics = torch.cat([torch.zeros_like(prior_sufficient_statistics), prior_sufficient_statistics], dim=-1)
        enc_dict['log_evidence'] = log_evidence
        posterior_update = D.PosteriorUpdate(sufficient_statistics, log_evidence)
        mode_posteriors = self.output.prior.update2(posterior_update, prior_sufficient_statistics)

        log_pz = self.prior.log_prob(z).sum(dim=-1)
        log_px = log_pz + log_det_sum
        nll = -log_px
        bpd = nll * np.log2(np.e) / np.prod(x.shape[1:])
        return mode_posteriors, log_prob, bpd
    
    
    def get_loss(self, forward_ret_dict):
        center_gt_trajs = forward_ret_dict['center_gt_trajs']
        mode_posteriors = forward_ret_dict['mode_posteriors']
        center_gt_final_valid_idx = forward_ret_dict['center_gt_final_valid_idx'].long()
        assert center_gt_trajs.shape[-1] == 4
        intention_points = forward_ret_dict['intention_points']
        num_center_objects = center_gt_trajs.shape[0]
        center_gt_goals = center_gt_trajs[torch.arange(num_center_objects), center_gt_final_valid_idx, 0:2] 
        dist = (center_gt_goals[:, None, :] - intention_points).norm(dim=-1)  # (num_center_objects, num_query)
        center_gt_positive_idx = dist.argmin(dim=-1)  # (num_center_objects)
        loss = self.loss.forward(mode_posteriors, center_gt_positive_idx)
        return loss 
    
    def _get_likelihood_with_latent(self, x, weights, return_ll=False):
        z, ldj = self.encode(x, weights)
        log_pz = self.prior.log_prob(z).sum(dim=-1)
        log_px = log_pz + ldj
        nll = -log_px
        bpd = nll * np.log2(np.e) / np.prod(x.shape[1:])
        return bpd.mean() if not return_ll else nll, z
    
    def predict_ood_score(self, x, weights):
        # with torch.no_grad():
        return self._get_likelihood(x, weights, return_ll=True)

    def predict_ood_score_with_latent(self, x, weights):
        # with torch.no_grad():
        return self._get_likelihood_with_latent(x, weights, return_ll=True)
    
    def get_distance(self, x, weights):
        # Encode x to get its latent representation z
        z, _ = self.encode(x, weights)
        # Since the mean is 0, the difference is just z itself
        # And assuming an isotropic Gaussian, the covariance matrix is I, simplifying the computation
        # Mahalanobis distance in this context reduces to the Euclidean distance
        distance = torch.sqrt(torch.sum(z**2, dim=1))
        return distance

class HyperNormalizingFlow(pl.LightningModule, CLHyperNetInterface):  
    def __init__(self, target_shapes, num_tasks, chunk_dims,
            layers=[256, 256], te_dim=8, activation_fn=torch.nn.ReLU(),
            use_bias=True, no_weights=False, ce_dim=None,
            init_weights=None, dropout_rate=-1, noise_dim=-1,
            temb_std=-1):
        pl.LightningModule.__init__(self)
        CLHyperNetInterface.__init__(self)
        self.flattened_target_shapes, self.index_mapping = flatten_dictionary(target_shapes)
        self.flow_net_indices = {}
        self.flow_nets = {}
        for key in target_shapes:
            flow_net_shapes, flow_net_indices = collect_target_shapes(self.index_mapping, self.flattened_target_shapes, key)
            hyper_flow = ChunkedHyperNetworkHandler(flow_net_shapes, num_tasks, no_te_embs=True, chunk_dim=chunk_dims[key], layers=layers, te_dim=te_dim, activation_fn=activation_fn, use_bias=use_bias, no_weights=no_weights, ce_dim=ce_dim, init_weights=init_weights, dropout_rate=dropout_rate, noise_dim=noise_dim, temb_std=temb_std)
            self.flow_net_indices[key] = flow_net_indices
            self.flow_nets[key] = hyper_flow
        self.flow_nets = nn.ModuleDict(self.flow_nets)
        self.te_dim = te_dim
        # shared task embeddings
        self._task_embs = nn.ParameterList()
        for _ in range(num_tasks):
            self._task_embs.append(nn.Parameter(data=torch.Tensor(te_dim//2),
                                                requires_grad=True))
            torch.nn.init.normal_(self._task_embs[-1], mean=0., std=1.)

    def create_new_task(self):
        self._task_embs.append(nn.Parameter(data=torch.Tensor(self.te_dim),
                                            requires_grad=True))
        torch.nn.init.normal_(self._task_embs[-1], mean=0., std=1.)

    def get_task_targets(self, task_id):
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
            W = self.forward(task_id=task_id)
            W = collect_W(W)
            ret = [d.detach().clone() for d in W]

        self.train(mode=hnet_mode)

        return ret
    
    def get_task_emb(self, task_id):
        return self._task_embs[task_id]
        
    def forward(self, task_id=None, theta=None, dTheta=None, task_emb=None,
            ext_inputs=None, squeeze=True):
        if task_emb is None:
            task_emb = self.get_task_emb(task_id)
        weight_dicts = {}
        last_index = 0
        for key in self.flow_nets:
            if theta is not None:
                theta_flow = theta[key]
            else:
                theta_flow = None 
            if dTheta is not None:
                dTheta_flow = dTheta[last_index:last_index+len(self.flow_nets[key].theta_shapes)]
                last_index += len(self.flow_nets[key].theta_shapes)
            else:
                dTheta_flow = None
            if ext_inputs is not None:
                ext_inputs_flow = ext_inputs[key]
            else:
                ext_inputs_flow = None
            flatten_weights_flow = self.flow_nets[key](task_id, theta_flow, dTheta_flow, task_emb, ext_inputs_flow, squeeze)
            for i, weight in zip(self.flow_net_indices[key], flatten_weights_flow):
                weight_dicts[i] = weight
        weights = assign_weights(self.index_mapping, weight_dicts)
        return weights

    @property
    def theta(self):
        th = []
        for key in self.flow_nets:
            th.extend(self.flow_nets[key].theta)
        return th
    
    @property
    def has_theta(self):
        """Getter for read-only attribute ``has_theta``."""
        return True
    @property
    def theta_shapes(self):
        th_shape = []
        for key in self.flow_nets:
            th_shape.extend(self.flow_nets[key].theta_shapes)
        return th_shape
    
    # def configure_optimizers(self):
    #     optimizer = optim.Adam(self.parameters(), lr=self.configs.train.lr)
    #     # An scheduler is optional, but can help in flows to get the last bpd improvement
    #     scheduler = optim.lr_scheduler.StepLR(optimizer, self.configs.train.lr_scheduler.step_size, gamma=self.configs.train.lr_scheduler.gamma)
    #     return [optimizer], [scheduler]

    # def on_train_start(self) -> None:
    #     print('on_train_start')
    #     self.start_epoch = self.current_epoch
    #     self.start_time = time.time()

    # def training_step(self, batch, batch_idx):
    #     # Normalizing flows are trained by maximum likelihood => return bpd
    #     loss = self._get_likelihood(batch)
    #     self.log("train_bpd", loss, prog_bar=True, batch_size=self.configs.train.batch_size)
    #     return loss
    
    # def validation_step(self, batch, batch_idx):
    #     loss = self._get_likelihood(batch)
    #     self.log("val_bpd", loss, prog_bar=True, batch_size=self.configs.train.batch_size)
    #     self.epoch_loss.append(loss.item())

    # def on_validation_epoch_end(self):
    #     avg_loss = np.mean(self.epoch_loss)
    #     self.epoch_loss.clear()

    #     if self.start_time is not None:
    #         time_spent = time.time() - self.start_time
    #         epoch_left = self.configs.train.epochs - self.current_epoch
    #         time_ETA = int(time_spent / (self.current_epoch - self.start_epoch + 1) * epoch_left)
    #         hours, rem_seconds = divmod(time_ETA, 3600)
    #         minutes = rem_seconds // 60
    #         tqdm.write("Training ETA: {:02}h {:02}m | Val @ epoch {}: Loss {:.2f}".format(hours, minutes, self.current_epoch, avg_loss))
