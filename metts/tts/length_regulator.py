class LengthRegulator(nn.Module):
    def __init__(self):
        super().__init__()
        self.target_length = lco["max_lengths"]["frame"]

    def forward(self, x, durations):
        MAX_FRAMES = self.target_length
        MAX_PHONES = x.shape[1]
        BATCH_SIZE = x.shape[0]
        EMB_DIM = x.shape[-1]

        val_ind = (torch.zeros((MAX_FRAMES, BATCH_SIZE), dtype=torch.int64).to(x.device)
            .scatter(
                0,
                durations.cumsum(-1).T,
                torch.ones(MAX_FRAMES, BATCH_SIZE, dtype=torch.int64).to(x.device)
            )
            .T.cumsum(-1)
        )

        ind = val_ind + (MAX_PHONES * torch.arange(BATCH_SIZE)).unsqueeze(1)
        val = x.reshape((-1, EMB_DIM))

        x = torch.nn.functional.embedding(ind.to(x.device), val, padding_idx=0)
        tgt_mask = ~(val_ind.view(x.shape[0], -1) == durations.shape[1]-1)
        
        return x