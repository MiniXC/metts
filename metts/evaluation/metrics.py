from torch.utils.data import DataLoader
import lco
import torchaudio
from mel_cepstral_distance import get_metrics_wavs
from copy import deepcopy
import torch
from pathlib import Path
from pystoi import stoi
from pesq import pesq
from torch import nn
import torchaudio.functional as F
import numpy as np
import scipy.io.wavfile as wf

class Metrics():
    def __init__(self, dataset, collator, num_examples=4, batch_size=1, save_audio=True):
        self.dataset = dataset
        self.num_examples = num_examples
        
        collator = deepcopy(collator)
        collator.include_audio = True

        self.loader = DataLoader(
            dataset=dataset,
            batch_size=batch_size,
            collate_fn=collator.collate_fn,
            num_workers=lco["evaluation"]["num_workers"],
            shuffle=False,
        )

        self.save_audio = save_audio

        self.chunk_size = lco["evaluation"]["chunk_size"]
        self.hop_length = lco["audio"]["hop_length"]

    def set_trainer(self, trainer):
        self.trainer = trainer

    def compute_mse(self, audio_preds, audio_trues):
        mse = 0
        for audio_pred, audio_true in zip(audio_preds, audio_trues):
            mse += np.square(np.subtract(audio_pred, audio_true)).mean()
        mse /= len(audio_preds)
        return mse

    def compute_mel_cepstral_distance(self, audio_preds, audio_trues):
        mcd = 0
        count = 0
        error_count = 0
        for audio_pred, audio_true in zip(audio_preds, audio_trues):
            # try:
            pred_path = Path("/tmp/pred.wav")
            true_path = Path("/tmp/true.wav")
            wf.write(pred_path, 16_000, audio_pred[0]+1e-5)
            wf.write(true_path, 16_000, audio_true[0]+1e-5)
            metric, pen, frames = get_metrics_wavs(pred_path, true_path, use_dtw=False)
            mcd += metric
            count += 1
        mcd /= count
        return mcd

    def compute_stoi(self, audio_preds, audio_trues):
        stoi_val = 0
        count = 0
        for audio_pred, audio_true in zip(audio_preds, audio_trues):
            metric = stoi(audio_true.flatten(), audio_pred.flatten(), 16_000)
            stoi_val += metric
            count += 1
        stoi_val /= count
        return stoi_val

    def compute_pesq(self, audio_preds, audio_trues):
        pesq_val = 0
        count = 0
        for audio_pred, audio_true in zip(audio_preds, audio_trues):
            metric = pesq(16_000, audio_true.flatten(), audio_pred.flatten(), "wb")
            pesq_val += metric
            count += 1
        pesq_val /= count
        return pesq_val

    @staticmethod
    def drc(x, C=1, clip_val=1e-5, log10=True):
        """Dynamic Range Compression"""
        if log10:
            return torch.log10(torch.clamp(x, min=clip_val) * C)
        else:
            return torch.log(torch.clamp(x, min=clip_val) * C)

    def compute_metrics(self, *args, **kwargs):
        model = self.trainer._wrap_model(self.trainer.model, training=False)
        
        audio_preds = []
        audio_trues = []

        device = model.device

        count = 0

        prc_index = self.trainer.args.process_index

        if self.save_audio:
            save_path = Path(f"examples/audio/{self.trainer.args.run_name}")
            save_path.mkdir(exist_ok=True, parents=True)

        for i, batch in enumerate(self.loader):
            if i >= prc_index:
                audio_true = batch["full_audio"]
                # pad to multiple of chunk_size * hop_length
                multiple_of = (audio_true.shape[1] // (self.chunk_size * self.hop_length))
                pad_size = (multiple_of + 1) * (self.chunk_size * self.hop_length) - audio_true.shape[1]
                audio_true = nn.ConstantPad1d((0, pad_size), 0)(audio_true)
                mel = Metrics.drc(model.mel(audio_true.to(device)))
                # use model.generate for every chunk
                audio_pred = torch.zeros_like(audio_true) + 1e-5
                for j in range(0, audio_true.shape[1]-1, self.chunk_size):
                    mel_chunk = mel[:, :, j:j+self.chunk_size]
                    if mel_chunk.shape[2] <= 1:
                        break
                    audio_pred[:, j*self.hop_length:(j+self.chunk_size)*self.hop_length] = \
                    model.generate(mel_chunk, lco["evaluation"]["sampling_T"])
                audio_preds.append(audio_pred.cpu().detach().numpy())
                audio_trues.append(audio_true.cpu().detach().numpy())
                if self.save_audio:
                    wf.write(
                        save_path / f"{i}.wav",
                        lco["audio"]["sampling_rate"],
                        audio_pred.cpu().detach().numpy().flatten(),
                    )
                    wf.write(
                        save_path / f"{i}_true.wav",
                        lco["audio"]["sampling_rate"],
                        audio_true.cpu().detach().numpy().flatten(),
                    )
                count += 1
            if count >= self.num_examples:
                break
                
        mse = self.compute_mse(audio_preds, audio_trues)
        mcd = self.compute_mel_cepstral_distance(audio_preds, audio_trues)
        stoi = self.compute_stoi(audio_preds, audio_trues)
        pesq = self.compute_pesq(audio_preds, audio_trues)

        return {
            "MSE": mse,
            "MCD": mcd,
            "STOI": stoi,
            "PESQ": pesq,
        }
        