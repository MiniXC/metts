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
import soundfile as sf

from metts.dataset.data_collator import MeTTSCollator, FastSpeechWithConsistencyCollator
from metts.tts.model import MeTTS
from metts.tts.model import FastSpeechWithConsistency

from metts.tts.consistency_predictor import ConformerConsistencyPredictor, ConformerConsistencyPredictorWithDVector
import time

from metts.hifigan import Synthesiser

# model = MeTTS.from_pretrained("output/checkpoint-60000")
# model.eval()

eval_data = load_dataset("metts/dataset/dataset.py", "libritts", split="train")

speaker2idx = json.load(open("data/speaker2idx.json"))
phone2idx = json.load(open("data/phone2idx.json"))
measure_stats = json.load(open("data/measure_stats.json"))
idx2phone = {v: k for k, v in phone2idx.items()}

collator = FastSpeechWithConsistencyCollator(
        speaker2idx=speaker2idx,
        phone2idx=phone2idx,
        measure_stats=measure_stats,
        keys="all",
    )

dl = DataLoader(
    eval_data,
    batch_size=128,
    collate_fn=collator.collate_fn,
    shuffle=False,
    num_workers=96,
)

consistency_net = ConformerConsistencyPredictorWithDVector.from_pretrained("models/consistencynet_small")
model = FastSpeechWithConsistency.from_pretrained("models/full_consistency_l1", consistency_net=consistency_net)

# eval
#model.eval()

# max_phone_length = float("-inf")

synth = Synthesiser(device="cpu")

for i, item in tqdm(enumerate(dl), total=len(dl)):

    # max_phone_length = max(max_phone_length, torch.max(torch.sum(item["phones"]!=0, dim=1)).item())
    # print(max_phone_length)

    mel = item["mel"]
    # denormalise
    mel = mel * 1.0958075523376465 + -0.19927863776683807

    result_inf = model(
        item["phones"],
        item["phone_durations"],
        item["durations"],
        item["mel"],
        item["val_ind"],
        item["speaker"],
        inference=True,
    )

    result_tf = model(
        item["phones"],
        item["phone_durations"],
        item["durations"],
        item["mel"],
        item["val_ind"],
        item["speaker"],
        force_tf=True,
    )

    # mel_shape (batch, mel_len, mel_dim)
    # mask such that only sections with 0s are masked
    mask = result_inf["mask"].squeeze(-1)
    synth_mel = result_inf["mel"][0].T[mask[0]]
    audio = synth(synth_mel)
    if len(audio.shape) == 1:
        audio = torch.tensor(audio).unsqueeze(0)
    else:
        audio = torch.tensor(audio)
    torchaudio.save("test.wav", audio, 22050)
    
    # plot ground truth vs predicted (tf) and predicted (inf)
    mel_min, mel_max = torch.min(mel), torch.max(mel)
    plt.figure(figsize=(20, 10))
    plt.subplot(3, 1, 1)
    plt.imshow(item["mel"][0].T, aspect="auto", origin="lower", vmin=mel_min, vmax=mel_max)
    plt.title("Ground Truth")
    plt.subplot(3, 1, 2)
    plt.imshow(result_tf["mel"][0].detach().numpy(), aspect="auto", origin="lower", vmin=mel_min, vmax=mel_max)
    plt.title("Predicted (TF)")
    plt.subplot(3, 1, 3)
    plt.imshow(result_inf["mel"][0].detach().numpy(), aspect="auto", origin="lower", vmin=mel_min, vmax=mel_max)
    plt.title("Predicted (Inf)")

    # save to examples
    plt.savefig(f"examples/{i}_mel_tf.png")

    print("inference mode losses")
    for loss in result_inf["loss_dict"]:
        print(loss, result_inf["loss_dict"][loss].item())

    print("force tf mode losses")
    for loss in result_tf["loss_dict"]:
        print(loss, result_tf["loss_dict"][loss].item())

    if i == 4:
        break