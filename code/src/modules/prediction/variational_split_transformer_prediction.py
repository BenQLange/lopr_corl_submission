from functools import partial

import numpy as np
import torch
import torch.nn as nn
from einops import rearrange, repeat
from src.modules.x_transformer import ( 
    Decoder, Encoder, TransformerWrapper)
from torch.distributions import Normal
from torch.distributions.kl import kl_divergence

def generate_square_subsequent_mask(sz):
    """Generates an upper-triangular matrix of ``-inf``, with zeros on ``diag``."""
    return torch.triu(torch.ones(sz, sz) * float('-inf'), diagonal=1)

class AbsolutePositionalEmbedding(nn.Module):
    def __init__(self, dim, max_seq_len):
        super().__init__()
        self.emb = nn.Embedding(max_seq_len, dim)
        self.init_()

    def init_(self):
        nn.init.normal_(self.emb.weight, std=0.02)

    def forward(self, x):
        n = torch.arange(x.shape[1], device=x.device)
        return self.emb(n)[None, :, :]

def get_padding_mask(batch, ignore_last_n=5, prob_mask=0.0):
    batch_size, seq_len = batch.shape[0], batch.shape[1]

    mask_np = np.random.choice([True, False], size=(batch_size,ignore_last_n), p=[prob_mask, 1-prob_mask])
    mask = torch.zeros((batch_size, seq_len), dtype=bool)
    if seq_len > ignore_last_n:
        mask[:, -ignore_last_n:] = torch.from_numpy(mask_np)
    return mask.to(batch.device)    

class VariationalSplitTransformerPredictorAutoregressive(nn.Module):
    def __init__(self, n_embed, n_enc_layer, n_dec_layer, n_output_tokens, n_head, dropout, dim_feedforward, obs_horizon, future_horizon, beta, beta_low, linear_schedule, deterministic_epochs=2, low_reg_epochs=10, max_seq_len_encoder=77, max_seq_len_decoder=77, device="cuda"):
        super().__init__()
        self.device = device
        self.n_output_tokens = n_output_tokens
        self.obs_horizon = obs_horizon
        self.future_horizon = future_horizon
        self.beta = beta
        self.deterministic_epochs = deterministic_epochs
        self.low_reg_epochs = low_reg_epochs
        self.beta_low = beta_low
        self.linear_schedule = linear_schedule

        self.enc_obs = nn.Linear(n_embed, n_embed)
        self.enc_maps = nn.Linear(n_embed, n_embed)
        self.enc_cam = nn.Linear(n_embed, n_embed)
        encoder = nn.TransformerEncoderLayer(d_model=n_embed, nhead=n_head, dropout=dropout, dim_feedforward=dim_feedforward, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder, num_layers=n_enc_layer)

        decoder = nn.TransformerDecoderLayer(d_model=n_embed, nhead=n_head, dropout=dropout, dim_feedforward=dim_feedforward, batch_first=True)
        self.transformer_decoder = nn.TransformerDecoder(decoder, num_layers=n_dec_layer)
        self.linear_out = nn.Linear(n_embed, n_embed)

        self.posterior_compress = nn.Linear(n_embed, n_embed)
        self.posterior_encoder = nn.TransformerEncoderLayer(d_model=n_embed, nhead=n_head, dropout=dropout, dim_feedforward=dim_feedforward, batch_first=True)
        self.posterior_transformer_encoder = nn.TransformerEncoder(self.posterior_encoder, num_layers=1)
        self.noise_compress = nn.Linear(n_embed, n_embed//2)
        self.noise_decompress = nn.Linear(n_embed//4, n_embed)
        # self.posterior_mean_std = nn.Linear(n_embed, 2*n_embed)
        self.pos_emb_enc_pos = AbsolutePositionalEmbedding(n_embed, max_seq_len_encoder)
        self.pos_emb_enc = AbsolutePositionalEmbedding(n_embed, max_seq_len_encoder)
        self.pos_emb_dec = AbsolutePositionalEmbedding(n_embed, max_seq_len_decoder)

        self.stoch_token = nn.Parameter(torch.randn(1, 1, n_embed))
        self.last_step = None

    def splitenzie(self, x):
        B, T, C, H, W = x.shape
        assert H == W == 4, 'Height and Width must be 4'
        
        x = x.reshape(B, T, C, H//2, 2, W//2, 2)
        x = x.permute(0, 1, 3, 5, 2, 4, 6).contiguous()
        x = x.reshape(B, 4*T, 4*C)
        
        return x

    def unsplitenzie(self, x):
        
        B, T, C = x.shape
        C //= 4
        T //= 4
        # print(f'Unsplitting shape: {B, T, C}. X shape: {x.shape}')
        x = x.reshape(B, T, 4, C, 4)
        x = x.permute(0, 1, 3, 2, 4)
        x = x.reshape(B, T, C, 2, 2, 2, 2)
        x = x.permute(0, 1, 2, 3, 5, 4, 6)
        x = x.reshape(B, T, C, 4, 4)
        
        return x

    def forward(self, obs, future=None, maps_input=None, camera_input=None, extrap_t=5, n_steps=15, masking=False, prob_masking=0.0, return_loss=False, prefix='val', epoch_number=0, steps=0, test=False):

        #All inputs should be in their original unmodified form. 
        B,T_past, C, H, W = obs.shape
        T_future = n_steps

        org_obs = obs.clone()
        if test == False:
            assert obs.shape[1] == 20, f'Future should be included for training. Obs shape:{obs.shape}'
        posterior_input = obs.clone()

        posterior_input = self.splitenzie(posterior_input)
        posterior_input = self.posterior_compress(posterior_input)
    
        obs = self.splitenzie(obs)        
        obs = self.enc_obs(obs)

        org_dec_tokens = (obs[:, :extrap_t*4, :]).clone()
        enc_tokens = torch.cat([obs[:,:self.obs_horizon*4]], dim=1)

        stoch_token = repeat(self.stoch_token, '() 1 d -> b 1 d', b=posterior_input.shape[0])
        posterior_input = torch.cat([posterior_input, stoch_token], dim=1) #It used to be before #Until 3.48 used the old one
        posterior_input = posterior_input + self.pos_emb_enc_pos(posterior_input)

        posterior_out = self.posterior_transformer_encoder(posterior_input)[:,-1,:][:,None,:] #[:,-1,:][:,None,:]
        posterior_out = self.noise_compress(posterior_out)
        mean, log_var = torch.chunk(posterior_out, 2, dim=-1)
        std = torch.exp(0.5 * log_var)
        std = torch.clamp(std, min=1e-3)
        dist = Normal(mean, std)
        target_dist = Normal(torch.zeros(mean.shape).to(obs.device), torch.ones(std.shape).to(obs.device))

        if self.last_step is None:
            self.last_step = steps

        if epoch_number < self.deterministic_epochs:
            stochastic_z_vec = target_dist.rsample()
            beta = 0
        elif epoch_number < self.low_reg_epochs:
            stochastic_z_vec = dist.rsample()
            # beta = 0.00001
            beta = self.beta_low
            self.last_step = steps
        else:
            stochastic_z_vec = dist.rsample()
            beta = torch.linspace(0, self.beta, self.linear_schedule).numpy()
            if steps - self.last_step < self.linear_schedule:
                beta = beta[steps-self.last_step] 
            else:
                beta = beta[-1]

        if test or prefix == 'val':
            print(f' \n TEST: Sampling from prior \n')
            stochastic_z_vec = target_dist.rsample()

        if maps_input is not None:
            maps_input = maps_input.to(self.device)
            maps_input = self.splitenzie(maps_input)
            maps_enc = self.enc_maps(maps_input)

            num_maps_tokens = maps_input.shape[1]
            enc_tokens = torch.cat([enc_tokens, maps_enc], dim=1)
        
        if camera_input is not None:
            camera_input = camera_input.to(self.device)
            camera_input = self.splitenzie(camera_input)
            camera_enc = self.enc_cam(camera_input)

            num_camera_tokens = camera_enc.shape[1]
            enc_tokens = torch.cat([enc_tokens, camera_enc], dim=1)
     
        enc_tokens = enc_tokens + self.pos_emb_enc(enc_tokens)
        if masking:    
            mems = self.transformer_encoder(enc_tokens, src_key_padding_mask=get_padding_mask(enc_tokens, ignore_last_n=num_maps_tokens + num_camera_tokens, prob_mask=prob_masking)) 
        else:
            mems = self.transformer_encoder(enc_tokens)

        stochastic_z_vec = self.noise_decompress(stochastic_z_vec)
        mems = torch.cat([mems, stochastic_z_vec], dim=1)

        for t in range(self.future_horizon - extrap_t):
            for split in range(4):
                self.tgt_mask = generate_square_subsequent_mask(org_dec_tokens.shape[1]).to(self.device)
                dec_tokens = org_dec_tokens.clone() + self.pos_emb_dec(org_dec_tokens)
                pred = self.transformer_decoder(dec_tokens, mems, tgt_mask=self.tgt_mask)
                if t == 0:
                    org_dec_tokens = pred
                    break
                else:
                    org_dec_tokens = torch.cat([org_dec_tokens, pred[:, -1, :][:,None,:]], dim=1)

        pred = org_dec_tokens 
        pred = self.linear_out(pred)
        pred = self.unsplitenzie(pred)

        if return_loss and not test:
            loss_val, loss_dict = self.loss(pred, org_obs, dist, target_dist, beta)

        if return_loss and not test:
            return pred, loss_val, loss_dict
        else:
            return pred, None, None

    
    def loss(self, pred, future, dist, target_dist, beta):
        kld_loss = kl_divergence(dist, target_dist).mean()
        optimized_loss = ((pred[:,-19:] - future[:,-19:])**2) 
        loss_org = ((pred[:,-15:] - future[:,-15:])**2)

        optimized_loss = optimized_loss.mean()
        loss_org = loss_org.mean()

        loss = optimized_loss + beta * kld_loss

        loss_dict = {
            f'loss': loss,
            f'kl_divergence': kld_loss,
            f'loss_mse_latents': loss_org,
            f'optimized_loss': optimized_loss,
            'beta': beta
        }
        return loss, loss_dict

    def predict(self, obs, n_steps=15):
        pred = self.forward(obs, n_steps=n_steps)
        return pred