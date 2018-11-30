# -*- coding: utf-8 -*-
import torch
from torch.optim.optimizer import Optimizer, required
import numpy as np

class Vanilla_Unitary(Optimizer):
    """Implements gradient descent for unitary matrix.
        
    Args:
        params (iterable): iterable of parameters to optimize or dicts defining
            parameter groups
        lr (float): learning rate
        


    .. note::
        This is the vanilla version of the gradient descent for unitary matrix, 
        i.e. formula (6) in H. D. Tagare. Notes on optimization on Stiefel manifolds. 
        Technical report, Yale University, 2011, and formula (6) in Scott Wisdom, 
        Thomas Powers, John Hershey, Jonathan Le Roux, and Les Atlas. Full-capacity 
        unitary recurrentneural networks. In NIPS 2016. 

        .. math::
                  A = G^H*W - W^H*G \\
                  W_new = (I+lr/2 * A)^(-1)*(I-lr/2 * A)*W

        where W, G and lr denote the parameters, gradient
        and learning rate respectively.
    """

    def __init__(self, params, lr=required):
        if lr is not required and lr < 0.0:
            raise ValueError("Invalid learning rate: {}".format(lr))


        defaults = dict(lr=lr)
        super(Vanilla_Unitary, self).__init__(params, defaults)

    def __setstate__(self, state):
        super(Vanilla_Unitary, self).__setstate__(state)
        for group in self.param_groups:
            group.setdefault('nesterov', False)

    def step(self, closure=None):
        """Performs a single optimization step.

        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
#            weight_decay = group['weight_decay']
#            momentum = group['momentum']
#            dampening = group['dampening']
#            nesterov = group['nesterov']
            lr = group['lr']
        
            for p in group['params']:
                
#                print(p.shape)
                if p.grad is None:
                    continue
                
                d_p = p.grad.data #G
                
                G = d_p[:,:,0].numpy()+1j* d_p[:,:,1].numpy()               
                W = p.data[:,:,0].numpy()+1j* p.data[:,:,1].numpy()   
                
                #A = G^H W - W^H G
                A_skew = np.matmul(np.matrix.getH(G),W) - np.matmul(np.matrix.getH(W),G)
                
                #W_new = (I+lr/2 * A)^(-1)*(I-lr/2 * A)*W
                identity = np.eye(A_skew.shape[0])
                cayley_denom =  np.linalg.inv(identity + (lr/2)* A_skew)
                cayley_numer = identity - (lr/2)* A_skew
                W_new = np.matmul(np.matmul(cayley_denom,cayley_numer),W)
                
                p_new_real = torch.tensor(W_new.real)
                p_new_imag = torch.tensor(W_new.imag)
                p_new = torch.cat((p_new_real.unsqueeze(2),p_new_imag.unsqueeze(2)),2)

                p.data = p_new.float()


        return loss
