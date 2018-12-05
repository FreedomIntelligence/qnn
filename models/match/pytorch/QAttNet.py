# -*- coding: utf-8 -*-

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class QAttNet(nn.Module):
    def __init__(self, opt):
        super(QAttNet, self).__init__()
        self.opt = opt
        embedding_matrix = torch.tensor(opt.lookup_table, dtype=torch.float).transpose(1,0)
        # tricks of utilize pretrained real-valued word vectors as complex-valued vectors 
        sign_matrix = torch.sign(embedding_matrix)
        phase_embedding_matrix = math.pi * (sign_matrix - 1) / 2
        amplitude_embedding_matrix = F.normalize(sign_matrix * embedding_matrix, p=2, dim=1, eps=1e-12)
        self.amplitude_embed = nn.Embedding.from_pretrained(amplitude_embedding_matrix)
        self.phase_embed = nn.Embedding.from_pretrained(phase_embedding_matrix, freeze=False)
        self.dense = nn.Linear(opt.embed_dim**2, opt.polarities_dim)

    def forward(self, inputs):
        text_raw_indices = inputs[0]
        aspect_indices = inputs[1]

        ctx_amp = self.amplitude_embed(text_raw_indices).unsqueeze(-1) # batch_size x seq_len x embed_dim x 1
        ctx_pha = self.phase_embed(text_raw_indices).unsqueeze(-1) # batch_size x seq_len x embed_dim x 1
        ctx_real = ctx_amp * torch.cos(ctx_pha) # batch_size x seq_len x embed_dim x 1
        ctx_imag = ctx_amp * torch.sin(ctx_pha) # batch_size x seq_len x embed_dim x 1
        asp_amp = self.amplitude_embed(aspect_indices).unsqueeze(-1) # batch_size x seq_len x embed_dim x 1
        asp_pha = self.phase_embed(aspect_indices).unsqueeze(-1) # batch_size x seq_len x embed_dim x 1
        asp_real = asp_amp * torch.cos(asp_pha) # batch_size x seq_len x embed_dim x 1
        asp_imag = asp_amp * torch.sin(asp_pha) # batch_size x seq_len x embed_dim x 1

        # (RE + j*IM)*(RE.T - j*IM.T) = (RE * RE.T + IM * IM.T) + j*(IM * RE.T - RE * IM.T)
        ctx_prod_real = torch.matmul(ctx_real, ctx_real.transpose(2, 3)) \
            + torch.matmul(ctx_imag, ctx_imag.transpose(2, 3)) # batch_size x (ctx) seq_len x embed_dim x embed_dim
        ctx_prod_imag = - torch.matmul(ctx_real, ctx_imag.transpose(2, 3)) \
            + torch.matmul(ctx_imag, ctx_real.transpose(2, 3)) # batch_size x (ctx) seq_len x embed_dim x embed_dim
        asp_prod_real = torch.matmul(asp_real, asp_real.transpose(2, 3)) \
            + torch.matmul(asp_imag, asp_imag.transpose(2, 3)) # batch_size x (asp) seq_len x embed_dim x embed_dim
        asp_prod_imag = - torch.matmul(asp_real, asp_imag.transpose(2, 3)) \
            + torch.matmul(asp_imag, asp_real.transpose(2, 3)) # batch_size x (asp) seq_len x embed_dim x embed_dim
        
        ctx_flatten_real = ctx_prod_real.view(ctx_prod_real.size(0), ctx_prod_real.size(1), -1) # batch_size x (ctx) seq_len x embed_dim^2
        ctx_flatten_imag = ctx_prod_imag.view(ctx_prod_imag.size(0), ctx_prod_imag.size(1), -1) # batch_size x (ctx) seq_len x embed_dim^2
        asp_flatten_real = asp_prod_real.view(asp_prod_real.size(0), asp_prod_real.size(1), -1) # batch_size x (asp) seq_len x embed_dim^2
        asp_flatten_imag = asp_prod_imag.view(asp_prod_imag.size(0), asp_prod_imag.size(1), -1) # batch_size x (asp) seq_len x embed_dim^2
        
        del ctx_amp
        del ctx_pha
        del ctx_real
        del ctx_imag
        del asp_amp
        del asp_pha
        del asp_real
        del asp_imag
        del ctx_prod_real
        del ctx_prod_imag
        del asp_prod_real
        del asp_prod_imag
        
        # only real part is needed as trace is real
        alpha_mat_real = torch.matmul(ctx_flatten_real, torch.transpose(asp_flatten_real, 1, 2)) \
            + torch.matmul(ctx_flatten_imag, torch.transpose(asp_flatten_imag, 1, 2)) # batch_size x (ctx) seq_len x (asp) seq_len
        alpha = F.softmax(alpha_mat_real.sum(2, keepdim=True), dim=1) # batch_size x (ctx) seq_len x 1
        mixture_real = torch.matmul(torch.transpose(ctx_flatten_real, 1, 2), alpha).squeeze(-1) # batch_size x embed_dim^2
        mixture_imag = torch.matmul(torch.transpose(ctx_flatten_imag, 1, 2), alpha).squeeze(-1) # batch_size x embed_dim^2
        out = self.dense(mixture_real) + self.dense(mixture_imag) # bathc_size x polarity_dim
        out = out.sum()

        return out
