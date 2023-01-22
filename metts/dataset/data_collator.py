from copy import deepcopy
import os
import multiprocessing
import pickle
import json

from tqdm.auto import tqdm
import numpy as np
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.nn import ConstantPad1d
import torchaudio
from transformers import Wav2Vec2FeatureExtractor, WavLMForXVector
import lco

from .measure import PitchMeasure, EnergyMeasure, SRMRMeasure, SNRMeasure
from .plotting import plot_item

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
        measures=None,
        pad_to_max_length=True,
        pad_to_multiple_of=None,
    ):
        self.sampling_rate = lco["audio"]["sampling_rate"]
        self.measures = measures
        self.phone2idx = phone2idx
        self.speaker2idx = speaker2idx
        # find max audio length & max duration
        self.max_frame_length = 0
        self.max_phone_length = 0
        self.max_frame_length = lco["max_lengths"]["frame"]
        self.max_phone_length = lco["max_lengths"]["phone"]
        self.pad_to_max_length = pad_to_max_length
        self.pad_to_multiple_of = pad_to_multiple_of
        self.wav2mel = JitWrapper("data/wav2mel.pt")
        self.dvector = JitWrapper("data/dvector.pt")
        
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
        for i, row in enumerate(batch):
            phones = row["phones"]
            batch[i]["phones"] = np.array([self.phone2idx[phone.replace("ˌ", "")] for phone in row["phones"]])
            sr = self.sampling_rate
            start = int(sr * row["start"])
            end = int(sr * row["end"])
            audio = row["audio"]["array"]
            audio = audio[start:end]
            audio = audio / np.abs(audio).max()
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
            durations[duration_mask_rm] = 0
            batch[i]["phone_durations"] = durations
            batch[i]["audio"]["array"] = audio[~duration_mask_rm_exp]
            dur_sum = sum(durations)
            unexpanded_silence_mask = ["[" in p for p in phones]
            silence_mask = self._expand(unexpanded_silence_mask, durations)
            if self.measures is not None:
                measure_paths = {
                    m: row["audio"]["path"].replace(".wav", "_{m}.pkl")
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
                        key: {
                            "array": (value["measure"] - value["prior"]["mean"]) / (value["prior"]["std"] + 1e-8),
                            "mean": value["prior"]["mean"],
                            "std": value["prior"]["std"],
                        }
                        for key, value in measure_dict.items()
                    }
                    for measure in self.measures:
                        with open(measure_paths[measure.name], "wb") as f:
                            pickle.dump(measures[measure.name], f)
                batch[i]["measures"] = measures
            batch[i]["audio_path"] = row["audio"]["path"]
        max_frame_length = max([sum(x["phone_durations"]) for x in batch])
        max_phone_length = max([len(x["phones"]) for x in batch])
        if self.pad_to_multiple_of is not None:
            max_frame_length = (max_frame_length // self.pad_to_multiple_of + 1) * self.pad_to_multiple_of
            max_phone_length = (max_phone_length // self.pad_to_multiple_of + 1) * self.pad_to_multiple_of
        if self.pad_to_max_length:
            max_frame_length = max(self.max_frame_length, max_frame_length)
            max_phone_length = max(self.max_phone_length, max_phone_length)
        max_audio_length = (max_frame_length * lco["audio"]["hop_length"]) - 1
        batch[0]["audio"]["array"] = ConstantPad1d(
            (0, max_audio_length - len(batch[0]["audio"]["array"])), 0
        )(torch.tensor(batch[0]["audio"]["array"]))
        batch[0]["phone_durations"] = ConstantPad1d(
            (0, max_phone_length - len(batch[0]["phone_durations"])), 0
        )(torch.tensor(batch[0]["phone_durations"]))
        batch[0]["phones"] = ConstantPad1d((0, max_phone_length - len(batch[0]["phones"])), 0
        )(torch.tensor(batch[0]["phones"]))
        for measure in self.measures:
            batch[0]["measures"][measure.name]["array"] = ConstantPad1d(
                (0, max_frame_length - len(batch[0]["measures"][measure.name]["array"])), 0
            )(torch.tensor(batch[0]["measures"][measure.name]["array"]))
        for i in range(1, len(batch)):
            batch[i]["audio"]["array"] = torch.tensor(batch[i]["audio"]["array"])
            batch[i]["phone_durations"] = torch.tensor(batch[i]["phone_durations"])
            batch[i]["phones"] = torch.tensor(batch[i]["phones"])
            for measure in self.measures:
                batch[i]["measures"][measure.name]["array"] = torch.tensor(batch[i]["measures"][measure.name]["array"])
        with torch.no_grad():
            if any(not os.path.exists(x["audio_path"].replace(".wav", "_speaker.pt")) for x in batch):
                result["embeddings"] = []
                for x in batch:
                    try:
                        embed = self.dvector.model.embed_utterance(self.wav2mel.model(x["audio"]["array"].unsqueeze(0), 22050)).squeeze(0)
                    except RuntimeError:
                        embed = torch.zeros(256)
                    result["embeddings"].append(embed)
                for i, x in enumerate(batch):
                    torch.save(result["embeddings"][i], x["audio_path"].replace(".wav", "_speaker.pt"))
            else:
                result["embeddings"] = torch.stack([torch.load(x["audio_path"].replace(".wav", "_speaker.pt")) for x in batch])
            torch.cuda.empty_cache()
        result["audio"] = pad_sequence([x["audio"]["array"] for x in batch], batch_first=True)
        result["phone_durations"] = pad_sequence([x["phone_durations"] for x in batch], batch_first=True)
        result["phones"] = pad_sequence([x["phones"] for x in batch], batch_first=True)
        speakers = [str(x["speaker"]).split("/")[-1] if ("/" in str(x["speaker"])) else x["speaker"] for x in batch]
        # speaker2idx
        result["speaker"] = torch.tensor([self.speaker2idx[x] for x in speakers])
        result["measures"] = {}
        result["measure_means"] = {}
        result["measure_stds"] = {}
        for measure in self.measures:
            result["measures"][measure.name] = pad_sequence([x["measures"][measure.name]["array"] for x in batch], batch_first=True)
            result["measure_means"][measure.name] = torch.tensor([x["measures"][measure.name]["mean"] for x in batch])
            result["measure_stds"][measure.name] = torch.tensor([x["measures"][measure.name]["std"] for x in batch])
        return result