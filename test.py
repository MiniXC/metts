import json
import multiprocessing

import torch
from tqdm.auto import tqdm
from datasets import load_dataset
from torch.utils.data import DataLoader
from matplotlib import pyplot as plt
import lco
import numpy as np

import metts.tts.refinement_models

from metts.dataset.data_collator import MeTTSCollator, FastSpeechWithConsistencyCollator
from metts.dataset.measure import PitchMeasure, EnergyMeasure, SRMRMeasure, SNRMeasure
from metts.dataset.plotting import plot_batch, plot_batch_meta
from metts.tts.consistency_predictor import ConformerConsistencyPredictorWithDVector
from metts.tts.scaler import GaussianMinMaxScaler

if __name__ == "__main__":
    lco.init("config/config.yaml")
    dataset = load_dataset("metts/dataset/dataset.py", "libritts")
    print(dataset.ds.info())
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
        keys=["mel", "measures", "durations", "phones", "dvector", "speaker"],
    )

    batch_size = 8

    dev = DataLoader(
        dataset=dev_ds,
        batch_size=batch_size,
        collate_fn=collator.collate_fn,
        num_workers=96,
        shuffle=False,
    )

    consistency_net = ConformerConsistencyPredictorWithDVector.from_pretrained("pretrained_models/consistency")

    speaker_dvec_dict = {}

    for j, batch in tqdm(enumerate(dev), total=len(dev)):
        consistency_net.eval()
        result = consistency_net(mel=batch["mel"], inference=True)
        dvec = result["logits_dvector"].detach().numpy()
        speaker = batch["speaker"].numpy()
        for i in range(len(speaker)):
            if speaker[i] not in speaker_dvec_dict:
                speaker_dvec_dict[speaker[i]] = []
            speaker_dvec_dict[speaker[i]].append(dvec[i])
        if j > 100:
            break

    # plot PCA
    import matplotlib.pyplot as plt
    from sklearn.decomposition import PCA

    pca = PCA(n_components=2)
    pca.fit(np.concatenate(list(speaker_dvec_dict.values())))
    for speaker, dvec in speaker_dvec_dict.items():
        dvec = pca.transform(dvec)
        plt.scatter(dvec[:, 0], dvec[:, 1], label=speaker)
    plt.legend()
    
    # save figure
    plt.savefig("dvec_pca_con.png")