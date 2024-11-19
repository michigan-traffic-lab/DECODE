import torch
import torch.nn as nn
from torch.nn import functional as F
from torch import Tensor
from typing import Any, Callable, Dict, List, Optional, Tuple, Type, Union
import pytorch_lightning as pl
import numpy as np 

class ContinualLearningLoss(pl.LightningDataModule):
    def __init__(self):
        super(ContinualLearningLoss, self).__init__()
        self.continual_learning_loss = nn.BCELoss()
    
    def __call__(self, hnet, task_id, targets=None, dTheta=None, dTembs=None,
                        mnet=None, inds_of_out_heads=None,
                        fisher_estimates=None, prev_theta=None,
                        prev_task_embs=None, batch_size=None, reg_scaling=None):
        ######################################################
        ## Adoped from https://github.com/chrhenning/hypercl #
        ######################################################
        r"""This regularizer simply restricts the output-mapping for previous
        task embeddings. I.e., for all :math:`j < \text{task\_id}` minimize:

        .. math::
            \lVert \text{target}_j - h(c_j, \theta + \Delta\theta) \rVert^2

        where :math:`c_j` is the current task embedding for task :math:`j` (and we
        assumed that `dTheta` was passed).

        Args:
            hnet: The hypernetwork whose output should be regularized. Has to
                implement the interface CLHyperNetInterface.
            task_id: The ID of the current task (the one that is used to
                compute dTheta.
            targets: A list of outputs of the hypernetwork. Each list entry must
                have the output shape as returned by the forward method of this
                class. Note, this method doesn't detach targets. If desired,
                that should be done before calling this method.
            dTheta (optional): The current direction of weight change for the
                internal weights of the hypernetwork evaluated on the task-specific
                loss, i.e., the weight change that would be applied to theta. This
                regularizer aims to modify this direction, such that the hypernet
                output for embeddings of previous tasks remains unaffected.
                Note, this function does not detach dTheta. It is up to the
                user to decide whether dTheta should be a constant vector or
                might depend on parameters of the hypernet.
            dTembs (optional): The current direction of weight change for the task
                embeddings of all tasks been learned already.
                See dTheta for details.
            mnet: Instance of the main network. Has to be given if
                `inds_of_out_heads` are specified.
            inds_of_out_heads: (optional): List of lists of integers, denoting which
                output neurons of the fully-connected output layer of the main
                network are used for predictions of the corresponding previous task.
                This will ensure that only weights of output neurons involved in
                solving a task are regularized.
                Note, this may only be used for main networks that have a fully-
                connected output layer.
            fisher_estimates (optional): A list of list of tensors, containing
                estimates of the Fisher Information matrix for each weight
                tensor in the main network and each task.
                Note, that :code:`len(fisher_estimates) == task_id`.
                The Fisher estimates are used as importance weights for single
                weights when computing the regularizer.
            prev_theta (optional): If given, `prev_task_embs` but not `targets`
                has to be specified. `prev_theta` is expected to be the internal
                weights theta prior to learning the current task. Hence, it can be
                used to compute the targets on the fly (which is more memory
                efficient (constant memory), but more computationally demanding).
                The computed targets will be detached from the computational graph.
                Independent of the current hypernet mode, the targets are computed
                in "eval" mode.
            prev_task_embs (optional): If given, `prev_theta` but not `targets`
                has to be specified. "prev_task_embs" are the task embeddings 
                learned prior to learning the current task. It is sufficient to
                only pass the task embeddings for tasks with ID smaller than the
                current one (only those tasks that are regularized).
                See docstring of "prev_theta" for more details.
            batch_size (optional): If specified, only a random subset of previous
                task mappings is regularized. If the given number is bigger than the
                number of previous tasks, all previous tasks are regularized.
            reg_scaling (optional): If specified, the regulariation terms for the 
                different tasks are scaled arcording to the entries of this list.
        Returns:
            The value of the regularizer.
        """
        # assert(task_id > 0)
        assert(hnet.has_theta) # We need parameters to be regularized.
        # assert(targets is None or len(targets) == task_id)
        assert(inds_of_out_heads is None or mnet is not None)
        assert(inds_of_out_heads is None or len(inds_of_out_heads) >= task_id)
        assert(targets is None or (prev_theta is None and prev_task_embs is None))
        assert(prev_theta is None or prev_task_embs is not None)
        assert(prev_task_embs is None or len(prev_task_embs) >= task_id)
        assert(dTembs is None or len(dTembs) >= task_id)
        assert(reg_scaling is None or len(reg_scaling) >= task_id)

        # Number of tasks to be regularized.
        if isinstance(targets, list):
            num_regs = task_id
            ids_to_reg = list(range(num_regs))
        elif isinstance(targets, dict):
            if fisher_estimates is not None:
                assert task_id in targets
                ids_to_reg = [k for k in targets.keys()] 
            else:
                ids_to_reg = [k for k in targets.keys() if k != task_id] 
            num_regs = len(ids_to_reg)
        
        # if batch_size is not None:
        #     if num_regs > batch_size:
        #         ids_to_reg = np.random.choice(num_regs, size=batch_size,
        #                                     replace=False).tolist()
        #         num_regs = batch_size

        reg = 0
        def collect_W(sub_dict):
            Ws = []
            for k, v in sub_dict.items():
                if isinstance(v, list):
                    Ws.extend(v)
                elif isinstance(v, dict):
                    Ws.extend(collect_W(v))
            return Ws
        
        for i in ids_to_reg:
            if dTembs is None:
                weights_predicted = hnet.forward(domain_id=i, dTheta=dTheta)
            else:
                temb = hnet.get_domain_emb(i) + dTembs[i]
                weights_predicted = hnet.forward(dTheta=dTheta, domain_emb=temb)

            weights_predicted = collect_W(weights_predicted)
            if targets is not None:
                target = targets[i]
            else:
                # Compute targets in eval mode!
                hnet_mode = hnet.training
                hnet.eval()

                # Compute target on the fly using previous hnet.
                with torch.no_grad():
                    target = hnet.forward(theta=prev_theta,
                                        domain_emb=prev_task_embs[i])
                target = [d.detach().clone() for d in target]

                hnet.train(mode=hnet_mode)

            if inds_of_out_heads is not None:
                # Regularize all weights of the main network except for the weights
                # belonging to output heads of the target network other than the
                # current one (defined by task id).
                W_target = flatten_and_remove_out_heads(mnet, target,
                                                        inds_of_out_heads[i])
                W_predicted = flatten_and_remove_out_heads(mnet, weights_predicted,
                                                        inds_of_out_heads[i])
            else:
                # Regularize all weights of the main network.
                W_target = torch.cat([w.view(-1) for w in target])
                W_predicted = torch.cat([w.view(-1) for w in weights_predicted])

            if fisher_estimates is not None and i == task_id:
                _assert_shape_equality(weights_predicted, fisher_estimates[i])
                FI = torch.cat([w.view(-1) for w in fisher_estimates[i]])

                reg_i = (FI * (W_target - W_predicted).pow(2)).sum() * 1e-4
            else:
                reg_i = (W_target - W_predicted).pow(2).sum()

            if reg_scaling is not None:
                reg += reg_scaling[i] * reg_i
            else:
                reg += reg_i

        return reg / num_regs

######################################################
## Adoped from https://github.com/chrhenning/hypercl #
######################################################

def _assert_shape_equality(list1, list2):
    """Ensure that 2 lists of tensors have the same shape."""
    assert(len(list1) == len(list2))
    for i in range(len(list1)):
        assert(np.all(np.equal(list(list1[i].shape), list(list2[i].shape))))

def flatten_and_remove_out_heads(mnet, weights, allowed_outputs):
    """Flatten a list of target network tensors to a single vector, such that
    output neurons that belong to other than the current output head are
    dropped.

    Note, this method assumes that the main network has a fully-connected output
    layer.

    Args:
        mnet: Main network instance.
        weights: A list of weight tensors of the main network (must adhere the
            corresponding weight shapes).
        allowed_outputs: List of integers, denoting which output neurons of
            the fully-connected output layer belong to the current head.

    Returns:
        The flattened weights with those output weights not belonging to the
        current head being removed.
    """
    # FIXME the option `mask_fc_out` did not exist in a previous version of the
    # main network interface, which is why we need to ensure downwards
    # compatibility.
    # Previously, it was assumed sufficient for masking if `has_fc_out` was set
    # to True.
    assert(mnet.has_fc_out)
    assert(not hasattr(mnet, 'mask_fc_out') or \
           (mnet.has_fc_out and mnet.mask_fc_out))

    obias_ind = len(weights)-1 if mnet.has_bias else -1
    oweights_ind = len(weights)-2 if mnet.has_bias else len(weights)-1

    ret = []
    for i, w in enumerate(weights):
        if i == obias_ind: # Output bias
            ret.append(w[allowed_outputs])
        elif i == oweights_ind: # Output weights
            ret.append(w[allowed_outputs, :].view(-1))
        else:
            ret.append(w.view(-1))

    return torch.cat(ret)