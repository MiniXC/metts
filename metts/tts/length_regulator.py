import torch
from torch import nn
import lco

class LengthRegulator(nn.Module):
    def __init__(self):
        super().__init__()
        self.target_length = lco["max_lengths"]["frame"]

    def forward(self, x, durations, val_ind):
        MAX_FRAMES = self.target_length
        MAX_PHONES = x.shape[1]
        BATCH_SIZE = x.shape[0]
        EMB_DIM = x.shape[-1]

        ind = val_ind + (MAX_PHONES * torch.arange(BATCH_SIZE)).unsqueeze(1)
        val = x.reshape((-1, EMB_DIM))

        comp_ind = torch.arange(MAX_PHONES * BATCH_SIZE).reshape((BATCH_SIZE, MAX_PHONES))        

        x = torch.nn.functional.embedding(ind.to(x.device), val)
        tgt_mask = val_ind != (MAX_PHONES - 1)
        
        return x, tgt_mask