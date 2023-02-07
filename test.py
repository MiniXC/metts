import json
import multiprocessing

import torch
from tqdm.auto import tqdm
from datasets import load_dataset
from torch.utils.data import DataLoader
from matplotlib import pyplot as plt
import lco

from metts.dataset.data_collator import MeTTSCollator
from metts.dataset.measure import PitchMeasure, EnergyMeasure, SRMRMeasure, SNRMeasure
from metts.dataset.plotting import plot_batch, plot_batch_meta

if __name__ == "__main__":
    lco.init("config/config.yaml")
    dataset = load_dataset("metts/dataset/dataset.py", "libritts")
    dev_ds = dataset["train"]
    speaker2idx = json.load(open("data/speaker2idx.json"))
    phone2idx = json.load(open("data/phone2idx.json"))
    measure_stats = json.load(open("data/measure_stats.json"))
    collator = MeTTSCollator(
        speaker2idx=speaker2idx,
        phone2idx=phone2idx,
        measure_stats=measure_stats,
        measures=[
            PitchMeasure(),
            EnergyMeasure(),
            SRMRMeasure(),
            SNRMeasure(),
        ],
        include_audio=True,
    )
    dev = DataLoader(
        dataset=dev_ds,
        batch_size=8,
        collate_fn=collator.collate_fn,
        num_workers=0,
    )

    m_dict = {
        k: [] for k in lco["meta"]
    }
    s_dict = {
        k: [] for k in lco["meta"]
    }

    for batch in tqdm(dev):
        for k in ["pitch", "energy", "srmr", "snr", "duration"]:
            if k == "duration":
                array_key = "durations"
            else:
                array_key = k
            fig = plot_batch_meta(batch, k, array_key)
            plt.savefig(f"examples/{k}.png")
        break