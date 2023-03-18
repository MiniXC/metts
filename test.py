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
from metts.tts.scaler import GaussianMinMaxScaler

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
        keys=["mel", "measures", "durations", "phones", "dvector"],
    )

    scaler_dict = {
        k: GaussianMinMaxScaler(10**9) for k in ["pitch", "energy", "srmr", "snr", "duration"]
    }

    # collator = FastSpeechWithConsistencyCollator(
    #     speaker2idx=speaker2idx,
    #     phone2idx=phone2idx,
    #     measure_stats=measure_stats,
    #     keys="all",
    # )
    batch_size = 8

    dev = DataLoader(
        dataset=dev_ds,
        batch_size=batch_size,
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
        k: 0 for k in ["pitch", "energy", "srmr", "snr", "duration", "mel", "dvector", "random"]
    }

    s1_dict = {
        k: 0 for k in ["pitch", "energy", "srmr", "snr", "duration", "mel", "dvector", "random"]
    }

    s2_dict = {
        k: 0 for k in ["pitch", "energy", "srmr", "snr", "duration", "mel", "dvector", "random"]
    }

    max_dict = {
        k: float("-inf") for k in ["pitch", "energy", "srmr", "snr", "duration", "mel", "dvector", "random"]
    }

    min_dict = {
        k: float("inf") for k in ["pitch", "energy", "srmr", "snr", "duration", "mel", "dvector", "random"]
    }

    max_dict_t = {
        k: float("-inf") for k in ["pitch", "energy", "srmr", "snr", "duration", "mel", "dvector", "random"]
    }

    min_dict_t = {
        k: float("inf") for k in ["pitch", "energy", "srmr", "snr", "duration", "mel", "dvector", "random"]
    }

    for i, batch in tqdm(enumerate(dev), total=len(dev)):
        pitch_val = batch["measures"]["pitch"]
        # scaler
        for k in ["pitch", "energy", "srmr", "snr"]:
            measure_val = batch["measures"][k].flatten()
            if k == "pitch":
                measure_val = measure_val[pitch_val.flatten() != 0]
            else:
                measure_val = measure_val[measure_val != 0]
            scaler_dict[k].partial_fit(measure_val)
            measure_val = scaler_dict[k].transform(measure_val)

            # plot as kdeplot using seaborn
            import seaborn as sns
            import matplotlib.pyplot as plt
            plt.figure()
            sns.kdeplot(measure_val, fill=True, label=f"{k}")
            # save to file
            plt.savefig(f"examples/{k}.png")
            plt.close()

            max_dict[k] = max(max_dict[k], measure_val.max())
            min_dict[k] = min(min_dict[k], measure_val.min())
            n_dict[k] += len(measure_val)
            s1_dict[k] += measure_val.sum()
            s2_dict[k] += (measure_val ** 2).sum()
        # duration
        duration = batch["durations"]
        duration_val = duration[batch["phones"]!=0].flatten()
        scaler_dict["duration"].partial_fit(duration_val)
        duration = scaler_dict["duration"].transform(duration_val)
        n_dict["duration"] += (batch["phones"]!=0).sum()
        s1_dict["duration"] += len(duration)
        s2_dict["duration"] += (duration ** 2).sum()
        max_dict["duration"] = max(max_dict["duration"], duration.max())
        min_dict["duration"] = min(min_dict["duration"], duration.min())
        # mel
        mel = batch["mel"]
        n_dict["mel"] += (mel!=0).sum()
        s1_dict["mel"] += mel.sum()
        s2_dict["mel"] += (mel ** 2).sum()
        max_dict["mel"] = max(max_dict["mel"], mel.max())
        min_dict["mel"] = min(min_dict["mel"], mel.min())
        # dvector
        dvector = batch["dvector"]
        n_dict["dvector"] += (dvector!=0).sum()
        s1_dict["dvector"] += dvector.sum()
        s2_dict["dvector"] += (dvector ** 2).sum()
        max_dict["dvector"] = max(max_dict["dvector"], dvector.max())
        min_dict["dvector"] = min(min_dict["dvector"], dvector.min())
        # random
        random = torch.normal(0, 1, size=mel.shape)
        n_dict["random"] += (random!=0).sum()
        s1_dict["random"] += random.sum()
        s2_dict["random"] += (random ** 2).sum()
        max_dict["random"] = max(max_dict["random"], random.max())
        min_dict["random"] = min(min_dict["random"], random.min())
        if i % 100 == 0:
            print("==================================")
            for k in n_dict:
                print(
                    k,
                    s1_dict[k] / n_dict[k],
                    np.sqrt(np.abs((s2_dict[k] / n_dict[k]) - ((s1_dict[k] / n_dict[k]) ** 2))),
                    max_dict[k],
                    min_dict[k],
                )