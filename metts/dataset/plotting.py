import matplotlib.pyplot as plt
from nnAudio.features.mel import MelSpectrogram
import numpy as np
import pandas as pd
import seaborn as sns
import lco
import torch

sns.set_theme(style="white", rc={"axes.facecolor": (0, 0, 0, 0)})


def plot_batch(batch):
    fig = plt.figure(figsize=(20, 10))
    audio_length = batch["audio"].shape[1]
    batch_size = batch["audio"].shape[0]
    df = pd.DataFrame({
        "x": np.tile(np.arange(audio_length), batch_size)/lco["audio"]["sampling_rate"],
        "y": batch["audio"].flatten().numpy(),
        "g": np.repeat(np.arange(batch_size), audio_length),
    })
    # Initialize the FacetGrid object
    pal = sns.cubehelix_palette(10, rot=-.25, light=.7)
    g = sns.FacetGrid(df, row="g", hue="g", aspect=15, height=.7, palette=pal)
    # Draw the densities in a few steps
    g.map(
        sns.lineplot,
        "x",
        "y",
        errorbar=None,
        clip_on=False,
        alpha=1,
        linewidth=1.5
    )
    # passing color=None to refline() uses the hue mapping
    g.refline(y=0, linewidth=2, linestyle="-", color=None, clip_on=False)
    # Define and use a simple function to label the plot in axes coordinates
    def label(x, color, label):
        ax = plt.gca()
        ax.text(0, .65, label, fontweight="bold", color=color,
                ha="left", va="center", transform=ax.transAxes)
    g.map(label, "x")
    # Set the subplots to overlap
    g.figure.subplots_adjust(hspace=-.25)
    # Remove axes details that don't play well with overlap
    g.set_titles("")
    g.set(yticks=[], ylabel="")
    g.despine(bottom=True, left=True)
    plt.xlabel("Time (s)")
    return fig

def drc(x, C=1, clip_val=1e-6, log10=True):
    """Dynamic Range Compression"""
    if log10:
        return torch.log10(torch.clamp(x, min=clip_val) * C)
    else:
        return torch.log(torch.clamp(x, min=clip_val) * C)

def plot_item(item, id2phone):
    mel = MelSpectrogram(
        sr=lco["audio"]["sampling_rate"],
        n_fft=lco["audio"]["n_fft"],
        win_length=lco["audio"]["win_length"],
        hop_length=lco["audio"]["hop_length"],
        n_mels=lco["audio"]["n_mels"],
        htk=True,
        power=2,
    )
    audio = torch.from_numpy(item["audio"]["array"])
    mel = drc(mel(audio))
    fig = plt.figure(figsize=(20, 10))

    audio_len = len(audio) / lco["audio"]["sampling_rate"]

    ax1 = fig.add_subplot(1,1,1)
    ax1.imshow(
        mel.squeeze(0),
        origin="lower",
        aspect="auto",
        extent=[0, audio_len, 0, 80],
        cmap="gray_r",
        alpha=0.8,
    )
    ax1.set_xlim(0, audio_len)
    ax1.set_ylim(0, 80)
    ax1.set_xlabel("Time (s)")
    ax1.set_yticks(range(0, 81, 10))

    # PHONES
    x = 0
    ax2 = ax1.twiny()
    phone_x = []
    phone_l = []
    for phone, duration in zip(item["phones"], item["phone_durations"]):
        phone = id2phone[phone.item()]
        new_x = x * lco["audio"]["hop_length"] / lco["audio"]["sampling_rate"]
        ax1.axline((new_x, 0), (new_x, 80), color="white", alpha=0.3)
        if phone == "[SILENCE]":
            phone = "☐"
        elif phone == "[COMMA]":
            phone = ","
        elif phone == "[FULL STOP]":
            phone = "."
        elif phone == "[QUESTION MARK]":
            phone = "?"
        if duration > 0:
            phone_x.append(new_x + duration * lco["audio"]["hop_length"] / lco["audio"]["sampling_rate"] / 2)
            phone_l.append(phone)
        else:
            if phone == "MASK":
                phone = "☐"
                plt.text(new_x, 80, phone, ha="center", va="center", color="red", fontsize=8)
        x += duration
            
    ax2.set_xlim(0, audio_len)
    ax2.set_xticks(phone_x)
    ax2.set_xticklabels(phone_l)
    ax2.set_xlabel(item["text"])
    ax2.tick_params(axis="x", labelsize=8)
    
    plt.title(item["audio"]["path"])

    return fig
