from torch.utils.data import DataLoader
import lco
import torchaudio
from mel_cepstral_distance import get_metrics_wavs
from copy import deepcopy

class Metrics():
    def __init__(self, dataset, collator, num_examples=10, batch_size=1, num_workers=0):
        self.dataset = dataset
        self.num_examples = num_examples
        
        collator = deepcopy(collator)
        collator.include_audio = True
        self.loader = DataLoader(
            dataset=dataset,
            batch_size=batch_size,
            collate_fn=collator.collate_fn,
            num_workers=num_workers,
        )

    def set_trainer(self, trainer):
        self.trainer = trainer

    def compute_rmse(self, audio_preds, audio_trues):
        rmse = 0
        for audio_pred, audio_true in zip(audio_preds, audio_trues):
            rmse += torch.sqrt(torch.mean((audio_pred - audio_true) ** 2))
        rmse /= len(audio_preds)
        return rmse.item()

    def compute_mel_cepstral_distance(self, audio_preds, audio_trues):
        mcd = 0
        for audio_pred, audio_true in zip(audio_preds, audio_trues):
            torchaudio.save("/tmp/pred.wav", audio_pred, lco["audio"]["sampling_rate"])
            torchaudio.save("/tmp/true.wav", audio_true, lco["audio"]["sampling_rate"])
            metric, pen, frames = get_metrics_wavs(audio_pred, audio_true, use_dtw=False)
            mcd += metric
        mcd /= len(audio_preds)
        return mcd

    def compute_metrics(self, *args, **kwargs):
        model = trainer._wrap_model(self.trainer.model, training=False)
        
        audio_preds = []
        audio_trues = []

        for i, batch in enumerate(self.loader):
            audio_pred = model.generate(batch["mel"], lco["evaluation"]["sampling_T"])
            audio_true = batch["audio"]
            audio_preds.append(audio_pred)
            audio_trues.append(audio_true)
            if i == self.num_examples:
                break

        rmse = self.compute_rmse(audio_preds, audio_trues)
        mcd = self.compute_mel_cepstral_distance(audio_preds, audio_trues)

        return {
            "RMSE": rmse,
            "MCD": mcd,
        }
        