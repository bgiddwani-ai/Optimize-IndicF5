"""Identical prepared BF16 DiT graph for eager and AOTInductor execution."""
import torch


class BF16Step(torch.nn.Module):
    def __init__(self, dit):
        super().__init__()
        self.dit = dit

    def forward(self, x, cond, text_embed, time_embedding, rope_freqs):
        with torch.autocast('cuda', dtype=torch.bfloat16):
            d = self.dit
            t = time_embedding
            x = d.input_embed(x, cond, text_embed, drop_audio_cond=False)
            residual = x
            for block in d.transformer_blocks:
                x = block(x, t, mask=None, rope=(rope_freqs, 1.0))
            if d.long_skip_connection is not None:
                x = d.long_skip_connection(torch.cat((x, residual), dim=-1))
            return d.proj_out(d.norm_out(x, t))
