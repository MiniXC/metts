import json
import multiprocessing

import torch
from tqdm.auto import tqdm
from datasets import load_dataset
from torch.utils.data import DataLoader
from matplotlib import pyplot as plt
import lco
import numpy as np

from metts.dataset.data_collator import MeTTSCollator, FastSpeechWithConsistencyCollator
from metts.dataset.measure import PitchMeasure, EnergyMeasure, SRMRMeasure, SNRMeasure
from metts.dataset.plotting import plot_batch, plot_batch_meta

if __name__ == "__main__":
    lco.init("config/config.yaml")
    dataset = load_dataset("metts/dataset/dataset.py", "libritts")
    dev_ds = dataset["train"]
    speaker2idx = json.load(open("data/speaker2idx.json"))
    phone2idx = json.load(open("data/phone2idx.json"))
    measure_stats = json.load(open("data/measure_stats.json"))
    # collator = MeTTSCollator(
    #     speaker2idx=speaker2idx,
    #     phone2idx=phone2idx,
    #     measure_stats=measure_stats,
    #     measures=[
    #         PitchMeasure(),
    #         EnergyMeasure(),
    #         SRMRMeasure(),
    #         SNRMeasure(),
    #     ],
    #     keys=["mel", "measures", "durations"],
    # )

    collator = FastSpeechWithConsistencyCollator(
        speaker2idx=speaker2idx,
        phone2idx=phone2idx,
        measure_stats=measure_stats,
        keys="all",
    )
    dev = DataLoader(
        dataset=dev_ds,
        batch_size=8,
        collate_fn=collator.collate_fn,
        num_workers=32,
        shuffle=True,
    )

    for i, batch in enumerate(dev):
        print(batch)
        raise

    m_dict = {
        k: [] for k in lco["meta"]
    }
    s_dict = {
        k: [] for k in lco["meta"]
    }

    mean_dict = {
        k: [] for k in ["pitch", "energy", "srmr", "snr", "duration"]
    }

    std_dict = {
        k: [] for k in ["pitch", "energy", "srmr", "snr", "duration"]
    }

    for i, batch in tqdm(enumerate(dev), total=100_000):
        for k in ["pitch", "energy", "srmr", "snr"]:
            mean_dict[k].append(np.concatenate(batch["measures"][k]).mean())
            std_dict[k].append(np.concatenate(batch["measures"][k]).std())
        mean_dict["duration"].append(np.concatenate(batch["durations"]).mean())
        std_dict["duration"].append(np.concatenate(batch["durations"]).std())
        if i % 10:
            print("==================================")
            for k in mean_dict:
                print(k, np.mean(mean_dict[k]), np.mean(std_dict[k]))