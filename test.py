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
        keys=["mel", "measures", "durations", "phones"],
    )

    # collator = FastSpeechWithConsistencyCollator(
    #     speaker2idx=speaker2idx,
    #     phone2idx=phone2idx,
    #     measure_stats=measure_stats,
    #     keys="all",
    # )
    dev = DataLoader(
        dataset=dev_ds,
        batch_size=8,
        collate_fn=collator.collate_fn,
        num_workers=96,
        shuffle=True,
    )

    m_dict = {
        k: [] for k in lco["meta"]
    }
    s_dict = {
        k: [] for k in lco["meta"]
    }

    n_dict = {
        k: 0 for k in ["pitch", "energy", "srmr", "snr", "duration", "mel"]
    }

    s1_dict = {
        k: 0 for k in ["pitch", "energy", "srmr", "snr", "duration", "mel"]
    }

    s2_dict = {
        k: 0 for k in ["pitch", "energy", "srmr", "snr", "duration", "mel"]
    }

    for i, batch in tqdm(enumerate(dev), total=100_000):
        for k in ["pitch", "energy", "srmr", "snr"]:
            measure_val = batch["measures"][k]
            n_dict[k] += (batch["measures"]["energy"] != 0).sum()
            s1_dict[k] += measure_val.sum()
            s2_dict[k] += (measure_val ** 2).sum()
        # duration
        duration = batch["durations"]
        n_dict["duration"] += (batch["phones"]!=0).sum()
        s1_dict["duration"] += duration.sum()
        s2_dict["duration"] += (duration ** 2).sum()
        # mel
        mel = batch["mel"]
        n_dict["mel"] += (mel!=0).sum()
        s1_dict["mel"] += mel.sum()
        s2_dict["mel"] += (mel ** 2).sum()
        if i % 10:
            print("==================================")
            for k in n_dict:
                print(
                    k,
                    s1_dict[k] / n_dict[k],
                    np.sqrt(s2_dict[k] / n_dict[k] - (s1_dict[k] / n_dict[k]) ** 2),
                )