import lco
lco.init("config/config.yaml")

import json

from datasets import load_dataset
import torch
from torch.utils.data import DataLoader
from matplotlib import pyplot as plt
from librosa.feature.inverse import mel_to_audio
import torchaudio

from metts.dataset.data_collator import MeTTSCollator
from metts.tts.model import MeTTS
from metts.dataset.plotting import plot_item



model = MeTTS.from_pretrained("output/checkpoint-60000")
model.eval()

eval_data = load_dataset("metts/dataset/dataset.py", "libritts", split="dev")

speaker2idx = json.load(open("data/speaker2idx.json"))
phone2idx = json.load(open("data/phone2idx.json"))
idx2phone = {v: k for k, v in phone2idx.items()}

collator = MeTTSCollator(
    speaker2idx=speaker2idx,
    phone2idx=phone2idx,
)

dl = DataLoader(
    eval_data,
    batch_size=1,
    collate_fn=collator.collate_fn,
)


for i, item in enumerate(dl):
    y = model.generate(item["phones"], item["phone_durations"], item["val_ind"], item["audio"]).detach()
    # fig = plot_item(
    #     item["audio"][0], 
    #     item["phones"][0],
    #     item["phone_durations"][0],
    #     eval_data[i]["text"],
    #     eval_data[i]["audio"]["path"],
    #     idx2phone,
    #     y
    # ) 
    print(y.shape)
    torchaudio.save("test_eval.wav", y.unsqueeze(0), lco["audio"]["sampling_rate"])
    #plt.imshow(y[0], aspect="auto", origin="lower", interpolation="none")
    plt.savefig("test_eval.png")
    break