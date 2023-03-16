import torch
from torch import nn
import lco

class LengthRegulator(nn.Module):
    def __init__(self):
        super().__init__()
        self.target_length = lco["max_lengths"]["frame"]

    def forward(self, x, durations, val_ind=None):
        MAX_FRAMES = self.target_length
        MAX_PHONES = x.shape[1]
        BATCH_SIZE = x.shape[0]
        EMB_DIM = x.shape[-1]

        # inference
        if val_ind is None:
            dur_sums = durations.sum(-1)
            if any(dur_sums > MAX_FRAMES):
                # loop over durations (from highest to lowest) and reduce them until the sum is less than MAX_FRAMES
                for i in range(BATCH_SIZE):
                    while dur_sums[i] > MAX_FRAMES:
                        # get the index of the highest duration
                        max_ind = torch.argmax(durations[i])
                        # reduce the duration by 1
                        durations[i, max_ind] -= 1
                        # update the sum
                        dur_sums[i] -= 1
            durations[:, -1] = MAX_FRAMES - durations.sum(-1)
            val_ind = torch.arange(0, MAX_PHONES).repeat(BATCH_SIZE).reshape(BATCH_SIZE, MAX_PHONES)
            val_ind = val_ind.flatten().repeat_interleave(durations.flatten(), dim=0).reshape(BATCH_SIZE, MAX_FRAMES)

        ind = val_ind + (MAX_PHONES * torch.arange(BATCH_SIZE)).unsqueeze(1)
        val = x.reshape((-1, EMB_DIM))

        comp_ind = torch.arange(MAX_PHONES * BATCH_SIZE).reshape((BATCH_SIZE, MAX_PHONES))        

        x = torch.nn.functional.embedding(ind.to(x.device), val)
        tgt_mask = val_ind != (MAX_PHONES - 1)
        
        return x, tgt_mask.unsqueeze(-1)