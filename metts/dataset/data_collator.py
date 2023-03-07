from copy import deepcopy
import os
import multiprocessing
import pickle
import json

import numpy as np
import torch
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence
from torch.nn import ConstantPad1d, ConstantPad2d
import torchaudio
from transformers import Wav2Vec2FeatureExtractor, WavLMForXVector
from nnAudio.features.mel import MelSpectrogram
import lco
import torchaudio.transforms as AT
import librosa
from librosa.filters import mel as librosa_mel
from time import time
from .plotting import plot_item

# helper class so we can use "with Timer():"
class Timer():
    def __init__(self, name=None):
        self.name = name

    def __enter__(self):
        self.start = time()

    def __exit__(self, *args):
        self.end = time()
        self.secs = self.end - self.start
        self.msecs = self.secs * 1000
        if self.name:
            print(f"{self.name} took {self.msecs} ms")

class JitWrapper():
    def __init__(self, path):
        super().__init__()
        self.path = path
        self.model = torch.jit.load(path)

    def __getstate__(self):
        self.path 
        return self.path

    def __setstate__(self, d):
        self.path = d
        self.model = torch.jit.load(d)

class MeTTSCollator():
    def __init__(
        self,
        phone2idx,
        speaker2idx,
        measure_stats,
        measures=None,
        pad_to_max_length=True,
        pad_to_multiple_of=None,
        include_audio=False,
        overwrite_max_length=True,
        keys=["vocoder_mel", "vocoder_audio"],
    ):
        self.sampling_rate = lco["audio"]["sampling_rate"]
        self.measures = measures
        self.phone2idx = phone2idx
        self.speaker2idx = speaker2idx
        self.measure_stats = measure_stats
        # find max audio length & max duration
        self.max_frame_length = 0
        self.max_phone_length = 0
        self.max_frame_length = lco["max_lengths"]["frame"]
        self.max_phone_length = lco["max_lengths"]["phone"]
        self.pad_to_max_length = pad_to_max_length
        self.pad_to_multiple_of = pad_to_multiple_of
        self.wav2mel = JitWrapper("data/wav2mel.pt")
        self.dvector = JitWrapper("data/dvector.pt")

        self.num_masked = 0
        self.num_total = 0
        self.percentage_mask_tokens = 0

        self.mel_spectrogram = AT.Spectrogram(
            n_fft=lco["audio"]["n_fft"],
            win_length=lco["audio"]["win_length"],
            hop_length=lco["audio"]["hop_length"],
            pad=0,
            window_fn=torch.hann_window,
            power=2.0,
            normalized=False,
            center=True,
            pad_mode="reflect",
            onesided=True,
        )
        self.mel_basis = librosa_mel(
            sr=self.sampling_rate,
            n_fft=lco["audio"]["n_fft"],
            n_mels=lco["audio"]["n_mels"],
            fmin=0,
            fmax=8000,
        )
        self.mel_basis = torch.from_numpy(self.mel_basis).float()

        self.include_audio = include_audio
        self.overwrite_max_length = overwrite_max_length

        self.keys = keys

    @staticmethod
    def drc(x, C=1, clip_val=1e-7):
        return torch.log(torch.clamp(x, min=clip_val) * C)
        
    def _expand(self, values, durations):
        out = []
        for value, d in zip(values, durations):
            out += [value] * max(0, int(d))
        if isinstance(values, list):
            return np.array(out)
        elif isinstance(values, torch.Tensor):
            return torch.stack(out)
        elif isinstance(values, np.ndarray):
            return np.array(out)
    
    def collate_fn(self, batch):
        result = {}

        #audio_torch, sr = torchaudio.load(batch[0]["audio"]["path"])
        #audio_hf = batch[0]["audio"]["array"]

        # print("audio_torch", audio_torch.shape, audio_torch.dtype, audio_torch.min(), audio_torch.max())
        # print("audio_hf", audio_hf.shape, audio_hf.dtype, audio_hf.min(), audio_hf.max())

        # raise

        for i, row in enumerate(batch):
            phones = row["phones"]
            batch[i]["phones"] = np.array([self.phone2idx[phone.replace("ˌ", "")] for phone in row["phones"]])
            sr = self.sampling_rate
            start = int(sr * row["start"])
            end = int(sr * row["end"])
            audio_path = row["audio"]
            # load audio with librosa
            audio, sr = librosa.load(audio_path, sr=sr, res_type="kaiser_fast")
            audio = audio
            audio = audio[start:end]
            audio = audio / np.abs(audio).max()

            # compute mel spectrogram
            mel = self.mel_spectrogram(torch.tensor(audio).unsqueeze(0))
            mel = torch.sqrt(mel[0])
            mel = torch.matmul(self.mel_basis, mel)
            mel = MeTTSCollator.drc(mel)
            mel = (mel - self.measure_stats["mel"]["mean"][0]) / self.measure_stats["mel"]["std"][0]

            batch[i]["mel"] = mel.T

            durations = np.array(row["phone_durations"])

            max_audio_len = durations.sum() * lco["audio"]["hop_length"]
            if len(audio) < max_audio_len:
                audio = np.pad(audio, (0, max_audio_len - len(audio)))
            elif len(audio) > max_audio_len:
                audio = audio[:max_audio_len]

            
            """
            several options (to test later):
            - remove longest: duration_permutation = np.argsort(durations)
            - remove random: duration_permutation = np.random.permutation(len(durations))
            - random hybrid: duration_permutation = np.argsort(durations+np.random.normal(0, durations.std(), len(durations)))
            """
            
            duration_permutation = np.argsort(durations+np.random.normal(0, durations.std(), len(durations)))
            duration_mask_rm = durations[duration_permutation].cumsum() >= self.max_frame_length
            duration_mask_rm = duration_mask_rm[np.argsort(duration_permutation)]
            batch[i]["phones"][duration_mask_rm] = self.phone2idx["MASK"]
            duration_mask_rm_exp = np.repeat(duration_mask_rm, durations * 256)
            duration_mask_rm_exp_frame = np.repeat(duration_mask_rm, durations)
            self.num_total += 1
            self.num_masked += 1 if sum(duration_mask_rm) > 0 else 0
            self.percentage_mask_tokens += sum(duration_mask_rm_exp) / len(duration_mask_rm_exp)
            durations[duration_mask_rm] = 0
            batch[i]["audio"] = audio[~duration_mask_rm_exp]
            if batch[i]["mel"].shape[0] > duration_mask_rm_exp_frame.shape[0]:
                batch[i]["mel"] = batch[i]["mel"][:duration_mask_rm_exp_frame.shape[0]]
            if batch[i]["mel"].shape[0] < duration_mask_rm_exp_frame.shape[0]:
                batch[i]["mel"] = torch.cat([batch[i]["mel"], torch.zeros(1, batch[i]["mel"].shape[1])])
            
            batch[i]["mel"] = batch[i]["mel"][~duration_mask_rm_exp_frame]
            dur_sum = sum(durations)
            unexpanded_silence_mask = ["[" in p for p in phones]
            silence_mask = self._expand(unexpanded_silence_mask, durations)
            batch[i]["phone_durations"] = durations.copy()
            durations = durations + (np.random.rand(*durations.shape) - 0.5)
            durations = (durations - self.measure_stats["duration"]["mean"][0]) / self.measure_stats["duration"]["std"][0]
            batch[i]["durations"] = durations
            if self.measures is not None:
                measure_paths = {
                    m: audio_path.replace(".wav", "_{m}.pkl")
                    for m in [measure.name for measure in self.measures]
                }
                if all([os.path.exists(path) for path in measure_paths]):
                    measures = {}
                    for measure in self.measures:
                        with open(measure_paths[measure.name], "rb") as f:
                            measures[measure.name] = pickle.load(f)
                else:
                    measure_dict = {
                        measure.name: measure(audio, row["phone_durations"], silence_mask, True)
                        for measure in self.measures
                    }
                    measures = {
                        key: (value["measure"] - self.measure_stats[key]["mean"][0]) / self.measure_stats[key]["std"][0]
                        for key, value in measure_dict.items()
                    }
                    for measure in self.measures:
                        with open(measure_paths[measure.name], "wb") as f:
                            pickle.dump(measures[measure.name], f)
                batch[i]["measures"] = measures
            batch[i]["audio_path"] = audio_path
        max_frame_length = max([sum(x["phone_durations"]) for x in batch])
        max_phone_length = max([len(x["phones"]) for x in batch])
        min_frame_length = min([sum(x["phone_durations"]) for x in batch])
        random_min_frame_length = np.random.randint(0, min_frame_length)
        if self.pad_to_multiple_of is not None:
            max_frame_length = (max_frame_length // self.pad_to_multiple_of + 1) * self.pad_to_multiple_of
            max_phone_length = (max_phone_length // self.pad_to_multiple_of + 1) * self.pad_to_multiple_of
        if self.pad_to_max_length and self.overwrite_max_length:
            max_frame_length = max(self.max_frame_length, max_frame_length)
            max_phone_length = max(self.max_phone_length, max_phone_length)
        max_audio_length = (max_frame_length * lco["audio"]["hop_length"])
        batch[0]["audio"] = ConstantPad1d(
            (0, max_audio_length - len(batch[0]["audio"])), 0
        )(torch.tensor(batch[0]["audio"]))
        batch[0]["mel"] = ConstantPad2d(
            (0, 0, 0, max_frame_length - batch[0]["mel"].shape[0]), 0
        )(batch[0]["mel"])
        batch[0]["phone_durations"] = ConstantPad1d(
            (0, max_phone_length - len(batch[0]["phone_durations"])), 0
        )(torch.tensor(batch[0]["phone_durations"]))
        batch[0]["durations"] = ConstantPad1d(
            (0, max_phone_length - len(batch[0]["durations"])), 0
        )(torch.tensor(batch[0]["durations"]))
        batch[0]["phones"] = ConstantPad1d((0, max_phone_length - len(batch[0]["phones"])), 0
        )(torch.tensor(batch[0]["phones"]))
        if self.measures is not None:
            for measure in self.measures:
                batch[0]["measures"][measure.name] = ConstantPad1d(
                    (0, max_frame_length - len(batch[0]["measures"][measure.name])), 0
                )(torch.tensor(batch[0]["measures"][measure.name]))
        for i in range(1, len(batch)):
            batch[i]["audio"] = torch.tensor(batch[i]["audio"])
            batch[i]["phone_durations"] = torch.tensor(batch[i]["phone_durations"])
            batch[i]["durations"] = torch.tensor(batch[i]["durations"])
            batch[i]["phones"] = torch.tensor(batch[i]["phones"])
            if self.measures is not None:
                for measure in self.measures:
                    batch[i]["measures"][measure.name] = torch.tensor(batch[i]["measures"][measure.name])
        with torch.no_grad():
            if any(not os.path.exists(x["audio_path"].replace(".wav", "_speaker.pt")) for x in batch):
                result["dvector"] = []
                for x in batch:
                    try:
                        embed = self.dvector.model.embed_utterance(self.wav2mel.model(x["audio"].unsqueeze(0), 22050)).squeeze(0)
                    except RuntimeError:
                        embed = torch.zeros(256)
                    result["dvector"].append(embed)
                for i, x in enumerate(batch):
                    torch.save(result["dvector"][i], x["audio_path"].replace(".wav", "_speaker.pt"))
            else:
                result["dvector"] = torch.stack([torch.load(x["audio_path"].replace(".wav", "_speaker.pt")) for x in batch])
            torch.cuda.empty_cache()
        result["audio"] = pad_sequence([x["audio"] for x in batch], batch_first=True)
        result["mel"] = pad_sequence([x["mel"] for x in batch], batch_first=True)
        result["phone_durations"] = pad_sequence([x["phone_durations"] for x in batch], batch_first=True)
        result["durations"] = pad_sequence([x["durations"] for x in batch], batch_first=True)
        result["phones"] = pad_sequence([x["phones"] for x in batch], batch_first=True)
        speakers = [str(x["speaker"]).split("/")[-1] if ("/" in str(x["speaker"])) else x["speaker"] for x in batch]
        # speaker2idx
        result["speaker"] = torch.tensor([self.speaker2idx[x] for x in speakers])
        result["measures"] = {}

        result["vocoder_mask"] = torch.zeros((max_frame_length))
        result["vocoder_mask"][random_min_frame_length:lco["max_lengths"]["vocoder"]+random_min_frame_length] = 1
        result["vocoder_mask"] = result["vocoder_mask"].bool()
        audio_min_idx = random_min_frame_length*lco["audio"]["hop_length"]
        audio_max_idx = (lco["max_lengths"]["vocoder"]+random_min_frame_length)*lco["audio"]["hop_length"]
        result["vocoder_audio"] = result["audio"][:,audio_min_idx:audio_max_idx]
        result["vocoder_mel"] = result["mel"].transpose(1,2)[:, :, result["vocoder_mask"]]

        if not self.include_audio:
            del result["audio"]

        if self.overwrite_max_length:
            MAX_FRAMES = lco["max_lengths"]["frame"]
            MAX_PHONES = lco["max_lengths"]["phone"]
            BATCH_SIZE = len(batch)
            result["phone_durations"][:, -1] = MAX_FRAMES - result["phone_durations"].sum(-1)
            result["val_ind"] = torch.arange(0, MAX_PHONES).repeat(BATCH_SIZE).reshape(BATCH_SIZE, MAX_PHONES)
            result["val_ind"] = result["val_ind"].flatten().repeat_interleave(result["phone_durations"].flatten(), dim=0).reshape(BATCH_SIZE, MAX_FRAMES)

        if self.measures is not None:
            for measure in self.measures:
                result["measures"][measure.name] = pad_sequence([x["measures"][measure.name] for x in batch], batch_first=True)
                # result["means"][measure.name] = torch.tensor([x["measures"][measure.name]["mean"] for x in batch])
                # result["stds"][measure.name] = torch.tensor([x["measures"][measure.name]["std"] for x in batch])
        else:
            result["measures"] = None

        # result["means"]["duration"] = torch.tensor(duration_means)
        # result["stds"]["duration"] = torch.tensor(duration_stds)
        # # audio
        # result["means"]["audio"] = torch.tensor(audio_means)
        # result["stds"]["audio"] = torch.tensor(audio_stds)
        # # mel
        # result["means"]["mel"] = torch.tensor(mel_means)
        # result["stds"]["mel"] = torch.tensor(mel_stds)

        # for k in result["means"]:
        #     result["means"][k] = (result["means"][k] - self.measure_stats[k]["mean"][0]) / self.measure_stats[k]["mean"][1]
        #     result["stds"][k] = (result["stds"][k] - self.measure_stats[k]["std"][0]) / self.measure_stats[k]["std"][1]

        result = {
            k: v
            for k, v in result.items()
            if k in self.keys
        }

        return result

class VocoderCollator():
    def __init__(
        self,
        include_audio=False,
    ):
        self.sampling_rate = lco["audio"]["sampling_rate"]
        self.include_audio = include_audio

    @staticmethod
    def drc(x, C=1, clip_val=1e-5, log10=True):
        """Dynamic Range Compression"""
        if log10:
            return torch.log10(torch.clamp(x, min=clip_val) * C)
        else:
            return torch.log(torch.clamp(x, min=clip_val) * C)
        
    def collate_fn(self, batch):
        result = {}

        chunk_length = lco["max_lengths"]["vocoder"]
        hop_length = lco["audio"]["hop_length"]

        full_audios = []
        audios = []

        for i, row in enumerate(batch):
            sr = lco["audio"]["sampling_rate"]
            start = int(sr * row["start"])
            end = int(sr * row["end"])
            audio, _sr = torchaudio.load(row["audio"])
            # resample
            if _sr != sr:
                if not hasattr(self, "_resampler"):
                    self._resampler = torchaudio.transforms.Resample(_sr, sr)
                audio = self._resampler(audio)
            audio = audio[0][start:end]

            # normalize audio
            audio = (audio - audio.mean()) / audio.std()

            # if self.include_audio:
            #     full_audios.append(audio)
            
            # get random chunk of audio
            audio_length = audio.shape[0]
            if (audio_length/hop_length)-chunk_length <= 0:
                # pad by chunk_length*hop_length-audio_length using torch
                audio = torch.nn.functional.pad(audio, (0, int(chunk_length*hop_length-audio_length)))
            else:
                if (audio_length//hop_length)-chunk_length <= 0:
                    start = 0
                else:
                    start = np.random.randint(0, (audio_length//hop_length)-chunk_length)
                audio = audio[start*hop_length:(start+chunk_length)*hop_length]

            audios.append(audio)

        result["audio"] = torch.tensor(np.stack(audios))

        # if self.include_audio:
        #     result["full_audio"] = torch.tensor(np.stack(full_audios))

        return result