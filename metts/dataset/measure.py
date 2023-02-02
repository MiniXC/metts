from abc import ABC, abstractmethod
import os

import numpy as np
import pyworld as pw
import librosa
import torch
from srmrpy import SRMR
from scipy import interpolate
import lco

from .snr import SNR

class Measure(ABC):
    def __init__(self, name, description):
        self.name = name
        self.description = description

    @staticmethod
    def interpolate(x):
        def nan_helper(y):
            return np.isnan(y), lambda z: z.nonzero()[0]
        nans, y = nan_helper(x)
        x[nans] = np.interp(y(nans), y(~nans), x[~nans])
        return x

    @abstractmethod
    def compute(self, audio, durations, silence_mask=None):
        pass

    def __call__(self, audio, durations, silence_mask=None, include_prior=False):
        if not lco["measures"]["silence_mask"]:
            silence_mask = None
        measure = self.compute(audio, durations, silence_mask)
        if include_prior:
            return {
                "measure": Measure.interpolate(measure),
                "prior": {
                    "mean": measure[~np.isnan(measure)].mean(),
                    "std": measure[~np.isnan(measure)].std(),
                }
            }
        else:
            return {
                "measure": Measure.interpolate(measure)
            }

    def __eq__(self, other):
        return self.name == other.name

class PitchMeasure(Measure):
    def __init__(
        self,
        name="pitch",
        description="Pitch measure",
        sampling_rate=22050,
        hop_length=256,
        pitch_quality=0.25
    ):
        global pw
        super().__init__(name, description)
        self.sampling_rate = sampling_rate
        self.hop_length = hop_length
        self.dio_speed = int(np.round(1 / pitch_quality))

    def compute(self, audio, durations, silence_mask=None):
        pitch, t = pw.dio(
            audio.astype(np.float64),
            self.sampling_rate,
            frame_period=self.hop_length / self.sampling_rate * 1000,
            speed=self.dio_speed,
        )
        pitch = pw.stonemask(audio.astype(np.float64), pitch, t, self.sampling_rate).astype(np.float32)
        pitch[pitch == 0] = np.nan
        if sum(durations) < len(pitch):
            pitch = pitch[:sum(durations)]
        if silence_mask is not None:
            pitch[silence_mask] = np.nan
        if np.isnan(pitch).all():
            pitch[:] = 1e-6
        return pitch

class EnergyMeasure(Measure):
    def __init__(
        self,
        name="energy",
        description="Energy measure",
        win_length=1024,
        hop_length=256,
    ):
        super().__init__(name, description)
        self.win_length = win_length
        self.hop_length = hop_length

    def compute(self, audio, durations, silence_mask=None):
        energy = librosa.feature.rms(
            y=audio,
            frame_length=self.win_length,
            hop_length=self.hop_length,
            center=False,
        ).astype(np.float32)
        energy = energy.reshape(-1)
        if sum(durations) < len(energy):
            energy = energy[:sum(durations)]
        return energy

class SRMRMeasure(Measure):
    def __init__(
        self,
        name="srmr",
        description="Signal-to-reverberant-ratio measure",
        sampling_rate=22050,
    ):
        super().__init__(name, description)
        self.srmr = SRMR(fs=sampling_rate, faster=True, norm=True)

    def compute(self, audio, durations, silence_mask=None):
        _, frame_srmr = self.srmr.srmr(torch.tensor(audio))
        if len(frame_srmr) == 1:
            srmr = np.repeat(frame_srmr, sum(durations))
        else:
            f = interpolate.interp1d(np.linspace(0,1,len(frame_srmr)), frame_srmr)
            srmr = f(np.linspace(0,1,sum(durations)))
        return srmr

class SNRMeasure(Measure):
    def __init__(
        self,
        name="snr",
        description="Signal-to-noise ratio",
        sampling_rate=22050,
        win_length=1024,
        hop_length=256,
    ):
        super().__init__(name, description)
        self.sampling_rate = sampling_rate
        self.win_length = win_length
        self.hop_length = hop_length

    def compute(self, audio, durations, silence_mask=None):
        snr = SNR(audio.astype(np.float32), self.sampling_rate)
        snr = snr.windowed_wada(window=self.win_length, stride=self.hop_length/self.win_length, use_samples=True)
        if sum(durations) < len(snr):
            snr = snr[:sum(durations)]
        if silence_mask is not None:
            snr[silence_mask] = np.nan
        if all(np.isnan(snr)):
            snr[:] = 1e-6
        return snr