import json
import multiprocessing

from datasets import load_dataset
from torch.utils.data import DataLoader
from matplotlib import pyplot as plt
import lco

from metts.dataset.data_collator import MeTTSCollator
from metts.dataset.measure import PitchMeasure, EnergyMeasure, SRMRMeasure, SNRMeasure
from metts.dataset.plotting import plot_batch

if __name__ == "__main__":
    lco.init("config/config.yaml")
    dataset = load_dataset("metts/dataset/dataset.py", "libritts")
    dev_ds = dataset["train"]
    speaker2idx = json.load(open("data/speaker2idx.json"))
    phone2idx = json.load(open("data/phone2idx.json"))
    collator = MeTTSCollator(
        speaker2idx=speaker2idx,
        phone2idx=phone2idx,
        measures=[
            PitchMeasure(),
        ],
    )
    dev = DataLoader(
        dataset=dev_ds,
        batch_size=8,
        shuffle=False,
        collate_fn=collator.collate_fn,
        num_workers=0,
    )
    for batch in dev:
        print(batch)
        # print("audio, frame, phone")
        # print(collator.max_audio_length, collator.max_frame_length, collator.max_phone_length)
        #fig = plot_batch(batch)
        #plt.savefig("test_audio.png")
        break