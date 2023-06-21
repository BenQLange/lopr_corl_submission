import numpy as np
import torch
import torch.nn as nn
from einops import rearrange, repeat
from src.modules.x_transformer import (  
    Decoder, Encoder, TransformerWrapper)


class AbstractEncoder(nn.Module):
    def __init__(self):
        super().__init__()

    def encode(self, *args, **kwargs):
        raise NotImplementedError



class ClassEmbedder(nn.Module):
    def __init__(self, embed_dim, n_classes=1000, key='class'):
        super().__init__()
        self.key = key
        self.embedding = nn.Embedding(n_classes, embed_dim)

    def forward(self, batch, key=None):
        if key is None:
            key = self.key
        # this is for use in crossattn
        c = batch[key][:, None]
        c = self.embedding(c)
        return c


class TransformerEmbedder(AbstractEncoder):
    """Some transformer encoder layers"""
    def __init__(self, n_embed, n_layer, max_seq_len=77, device="cuda"):
        super().__init__()
        self.device = device
        self.transformer = TransformerWrapper(max_seq_len=max_seq_len,
                                              use_pos_emb=True,
                                              attn_layers=Encoder(dim=n_embed, depth=n_layer))

    def forward(self, tokens):
        tokens = tokens.to(self.device)  # meh
        z = self.transformer(tokens)
        # self.emb = mems
        return z

    def encode(self, x):
        return self(x)

    
class TransformerDecoderAutoregressive(nn.Module):
    def __init__(self, n_embed, n_layer, n_output_tokens, max_seq_len=77, decoder=False, device="cuda"):
        super().__init__()
        self.device = device
        self.n_output_tokens = n_output_tokens
        if not decoder:
            raise ValueError
            # attn_layer = Encoder(dim=n_embed, depth=n_layer)
        else:
            attn_layer = Decoder(dim=n_embed, depth=n_layer)
        self.transformer = TransformerWrapper(max_seq_len=max_seq_len,
                                              use_pos_emb=True,
                                              attn_layers=attn_layer)
        
        self.pred_latent = nn.Parameter(torch.randn(1, n_output_tokens, n_embed))

    def forward(self, tokens, memory, intermediate_noise=None):

        """
        TODO: intermediate tokens
        """

        tokens = tokens.to(self.device)
        B, C, H, W = tokens.shape    
        tokens = tokens.reshape((B,1, -1)) 
        # print(f'Tokens:{tokens.shape}. Pred latent:{self.pred_latent.shape}. Mem:{memory.shape}')
        pred_latent = self.pred_latent.repeat(B,1,1)

        if intermediate_noise is None:
            tokens = torch.cat([pred_latent, tokens], dim=1)
        else:
            #print(f'Tokens:{tokens.shape}. Pred latent:{pred_latent.shape}. Mem:{memory.shape}. Noise:{intermediate_noise.shape}')
            tokens = torch.cat([intermediate_noise, tokens], dim=1)

        #Concatenate it with parameter
        memory = memory.to(self.device)
        out = self.transformer(tokens, context=memory)

        # Pick n-1 tokens.
        
        idx = np.linspace(0, out.shape[1]-1, self.n_output_tokens + 1, dtype=int)[1:]
        assert idx[-1] == out.shape[1]-1, f'Last index is {idx[-1]} but should be {out.shape[1]-1}'
        out = out[:,idx,:]  
        out = out.reshape((B, self.n_output_tokens, C, H, W))

        return out