import lco
lco.init("config/config.yaml")

import json

from datasets import load_dataset
import torch
from torch.utils.data import DataLoader
from matplotlib import pyplot as plt
import seaborn as sns
sns.set_style('whitegrid', {'legend.frameon':True})
import pandas as pd
from librosa.feature.inverse import mel_to_audio
import torchaudio
from tqdm.auto import tqdm

from metts.dataset.data_collator import MeTTSCollator
# from metts.dataset.plotting import plot_item, plot_batch_meta
from metts.dataset.measure import PitchMeasure, EnergyMeasure, SRMRMeasure, SNRMeasure

from metts.tts.consistency_predictor import ConformerConsistencyPredictorWithDVector

import time

# model = MeTTS.from_pretrained("output/checkpoint-60000")
# model.eval()

eval_data = load_dataset("metts/dataset/dataset.py", "libritts", split="train")

speaker2idx = json.load(open("data/speaker2idx.json"))
phone2idx = json.load(open("data/phone2idx.json"))
measure_stats = json.load(open("data/measure_stats.json"))
idx2phone = {v: k for k, v in phone2idx.items()}

collator = MeTTSCollator(
    speaker2idx=speaker2idx,
    phone2idx=phone2idx,
    measure_stats=measure_stats,
    keys=["mel", "measures",  "dvector"],
    measures=[PitchMeasure(), EnergyMeasure(), SRMRMeasure(), SNRMeasure()],
)

dl = DataLoader(
    eval_data,
    batch_size=8,
    collate_fn=collator.collate_fn,
    shuffle=True,
    num_workers=96,
)

model = ConformerConsistencyPredictorWithDVector.from_pretrained("models/consistency_scalers_final")
# eval
model.eval()
# disable dropout

measure_order = ["energy", "pitch", "srmr", "snr"]

loss_dicts = []

for i, item in tqdm(enumerate(dl)):
    mel = item["mel"]
    result = model(mel, measures=item["measures"], dvector=item["dvector"])
    # plot predicted measures against ground truth and save to file
    # first we construct a dataframe, then we create one plot per measure (inkl. ground truth and predicted)
    # then we save the plot to a file
    # plot first element in batch for each measure as lineplot and save to file
    # if i <= 200:
    #     continue
    # else:
    #     # save model 
    #     model.save_pretrained(f"models/new")
    #     break
    if i == 0:
        df = pd.DataFrame()
        for j in range(4):
            for k, measure in enumerate(measure_order):
                unscaled_measure_true = item["measures"][measure][j].detach().cpu().numpy().tolist()
                unscaled_measure_pred = model.scalers[measure].inverse_transform(result["logits"][:, k][j]).detach().cpu().numpy().tolist()
                scaled_measure_true = model.scalers[measure].transform(item["measures"][measure][j]).detach().cpu().numpy().tolist()
                scaled_measure_pred = result["logits"][:, k][j].detach().cpu().numpy().tolist()
                df[f"{measure}_y_{j}"] = scaled_measure_true + scaled_measure_pred
                df[f"{measure}_x_{j}"] = list(range(len(item["measures"][measure][j]))) + list(range(len(item["measures"][measure][j])))
                df[f"{measure}_type_{j}"] = ["ground truth"] * len(item["measures"][measure][j]) + ["predicted"] * len(item["measures"][measure][j])
            for measure in measure_order:
                sns.lineplot(data=df, x=f"{measure}_x_{j}", y=f"{measure}_y_{j}", hue=f"{measure}_type_{j}", alpha=0.8)
                # add mel spectrogram with extent (in grayscale)
                plt.imshow(mel[j].detach().cpu().numpy().T, aspect="auto", origin="lower", extent=[0, len(mel[j]), df[f"{measure}_y_{j}"].min(), df[f"{measure}_y_{j}"].max()], cmap="gray", alpha=0.8)
                legend = plt.legend()
                frame = legend.get_frame()
                frame.set_facecolor('white')
                plt.savefig(f"examples/new/{j}_{measure}.png")
                plt.clf()
    loss_dicts.append(result["compound_losses"])
    if i >= 10:
        break

# for measure in measure_order:
#     loss = sum([loss_dict[measure] for loss_dict in loss_dicts])
#     print(f"{measure}: {loss / len(loss_dicts)}")