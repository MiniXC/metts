"""MeTTS dataset."""

import os
from pathlib import Path
import hashlib
import pickle

import datasets
import pandas as pd
import numpy as np
from alignments.datasets.libritts import LibrittsDataset
from tqdm.contrib.concurrent import process_map
from tqdm.auto import tqdm
from multiprocessing import cpu_count
from phones.convert import Converter
import torchaudio
import torchaudio.transforms as AT

logger = datasets.logging.get_logger(__name__)

_PHONESET = "arpabet"

_VERBOSE = os.environ.get("METTS_VERBOSE", True)

_MAX_WORKERS = os.environ.get("METTS_MAX_WORKERS", cpu_count())

_VERSION = "1.0.0"

_PATH = os.environ.get("METTS_PATH", os.environ.get("HF_DATASETS_CACHE", None))
if _PATH is not None and not os.path.exists(_PATH):
    os.makedirs(_PATH)

_NO_MEASURES = os.environ.get("METTS_NO_MEASURES", False)

_CITATION = """\
@article{https://doi.org/10.48550/arxiv.2211.16049,
  author = {Minixhofer, Christoph and Klejch, Ondřej and Bell, Peter},
  title = {Evaluating and reducing the distance between synthetic and real speech distributions},
  year = {2022}
}
"""

_DESCRIPTION = """\
Dataset used for loading TTS spectrograms and waveform audio with a number of configurable "measures", which are extracted from the raw audio.
"""

_URL = "https://www.openslr.org/resources/60/"
_URLS = {
    "dev-clean": _URL + "dev-clean.tar.gz",
    "dev-other": _URL + "dev-other.tar.gz",
    "test-clean": _URL + "test-clean.tar.gz",
    "test-other": _URL + "test-other.tar.gz",
    "train-clean-100": _URL + "train-clean-100.tar.gz",
    "train-clean-360": _URL + "train-clean-360.tar.gz",
    "train-other-500": _URL + "train-other-500.tar.gz",
}


class MeTTSConfig(datasets.BuilderConfig):
    """BuilderConfig for MeTTS."""

    def __init__(self, sampling_rate=22050, hop_length=256, win_length=1024, **kwargs):
        """BuilderConfig for MeTTS.

        Args:
          **kwargs: keyword arguments forwarded to super.
        """
        super(MeTTSConfig, self).__init__(**kwargs)

        self.sampling_rate = sampling_rate
        self.hop_length = hop_length
        self.win_length = win_length
        
        if _PATH is None:
            raise ValueError("Please set the environment variable METTS_PATH to point to the MeTTS dataset directory.")
        elif _PATH == os.environ.get("HF_DATASETS_CACHE", None):
            logger.warning("Please set the environment variable METTS_PATH to point to the MeTTS dataset directory. Using HF_DATASETS_CACHE as a fallback.")

class MeTTS(datasets.GeneratorBasedBuilder):
    """MeTTS dataset."""

    BUILDER_CONFIGS = [
        MeTTSConfig(
            name="libritts",
            version=datasets.Version(_VERSION, ""),
        ),
    ]

    def _info(self):
        features = {
            "id": datasets.Value("string"),
            "speaker": datasets.Value("string"),
            "text": datasets.Value("string"),
            "start": datasets.Value("float32"),
            "end": datasets.Value("float32"),
            # phone features
            "phones": datasets.Sequence(datasets.Value("string")),
            "phone_durations": datasets.Sequence(datasets.Value("int32")),
            # audio feature
            "audio": datasets.Audio(sampling_rate=self.config.sampling_rate),
        }

        return datasets.DatasetInfo(
            description=_DESCRIPTION,
            features=datasets.Features(features),
            supervised_keys=None,
            homepage="https://github.com/MiniXC/MeTTS",
            citation=_CITATION,
            task_templates=None,
        )

    def _split_generators(self, dl_manager):
        ds_dict = {}
        for name, url in _URLS.items():
            ds_dict[name] = self._create_alignments_ds(name, url)
        splits = [
            datasets.SplitGenerator(
                name=key.replace("-", "."),
                gen_kwargs={"ds": self._create_data(value)}
            ) 
            for key, value in ds_dict.items()
        ]
        # dataframe with all data
        data_train = self._create_data([ds_dict["train-clean-100"], ds_dict["train-clean-360"], ds_dict["train-other-500"]])
        data_dev = self._create_data([ds_dict["dev-clean"], ds_dict["dev-other"]])
        data_test = self._create_data([ds_dict["test-clean"], ds_dict["test-other"]])
        data_all = pd.concat([data_train, data_dev, data_test])
        splits += [
            datasets.SplitGenerator(
                name="train.all",
                gen_kwargs={
                    "ds": data_all,
                }
            ),
            datasets.SplitGenerator(
                name="dev.all",
                gen_kwargs={
                    "ds": data_dev,
                }
            ),
            datasets.SplitGenerator(
                name="test.all",
                gen_kwargs={
                    "ds": data_test,
                }
            ),
        ]
        # move last row for each speaker from data_all to dev dataframe
        data_dev = data_all.copy()
        data_dev = data_dev.sort_values(by=["speaker", "audio"])
        data_dev = data_dev.groupby("speaker").tail(1)
        data_dev = data_dev.reset_index()
        # remove last row for each speaker from data_all
        data_all = data_all[~data_all["audio"].isin(data_dev["audio"])]
        splits += [
            datasets.SplitGenerator(
                name="train",
                gen_kwargs={
                    "ds": data_all,
                }
            ),
            datasets.SplitGenerator(
                name="dev",
                gen_kwargs={
                    "ds": data_dev,
                }
            ),
        ]
        self.alignments_ds = None
        self.data = None
        return splits

    def _create_alignments_ds(self, name, url):
        self.empty_textgrids = 0
        ds_hash = hashlib.md5(os.path.join(_PATH, f"{name}-alignments").encode()).hexdigest()
        pkl_path = os.path.join(_PATH, f"{ds_hash}.pkl")
        if os.path.exists(pkl_path):
            ds = pickle.load(open(pkl_path, "rb"))
        else:
            tgt_dir = os.path.join(_PATH, f"{name}-alignments")
            src_dir = os.path.join(_PATH, f"{name}-data")
            if os.path.exists(tgt_dir):
                src_dir = None
                url = None
            if os.path.exists(src_dir):
                url = None
            ds = LibrittsDataset(
                target_directory=tgt_dir,
                source_directory=src_dir,
                source_url=url,
                verbose=_VERBOSE,
                tmp_directory=os.path.join(_PATH, f"{name}-tmp"),
                chunk_size=1000,
            )
            pickle.dump(ds, open(pkl_path, "wb"))
        return ds, ds_hash

    def _create_data(self, data):
        entries = []
        self.phone_cache = {}
        self.phone_converter = Converter()
        if not isinstance(data, list):
            data = [data]
        hashes = [ds_hash for ds, ds_hash in data]
        ds = [ds for ds, ds_hash in data]
        self.ds = ds
        del data
        for i, ds in enumerate(ds):
            if os.path.exists(os.path.join(_PATH, f"{hashes[i]}-entries.pkl")):
                add_entries = pickle.load(open(os.path.join(_PATH, f"{hashes[i]}-entries.pkl"), "rb"))
            else:
                add_entries = [
                    entry
                    for entry in process_map(
                        self._create_entry,
                        zip([i] * len(ds), np.arange(len(ds))),
                        chunksize=10_000,
                        max_workers=_MAX_WORKERS,
                        desc=f"processing dataset {hashes[i]}",
                        tqdm_class=tqdm,
                    )
                    if entry is not None
                ]
                pickle.dump(add_entries, open(os.path.join(_PATH, f"{hashes[i]}-entries.pkl"), "wb"))
            entries += add_entries
        if self.empty_textgrids > 0:
            logger.warning(f"Found {self.empty_textgrids} empty textgrids")
        return pd.DataFrame(
            entries,
            columns=[
                "phones",
                "duration",
                "start",
                "end",
                "audio",
                "speaker",
                "text",
                "basename",
            ],
        )
        del self.ds, self.phone_cache, self.phone_converter

    def _create_entry(self, dsi_idx):
        dsi, idx = dsi_idx
        item = self.ds[dsi][idx]
        start, end = item["phones"][0][0], item["phones"][-1][1]

        phones = []
        durations = []

        for i, p in enumerate(item["phones"]):
            s, e, phone = p
            phone.replace("ˌ", "")
            r_phone = phone.replace("0", "").replace("1", "")
            if len(r_phone) > 0:
                phone = r_phone
            if "[" not in phone:
                o_phone = phone
                if o_phone not in self.phone_cache:
                    phone = self.phone_converter(
                        phone, _PHONESET, lang=None
                    )[0]
                    self.phone_cache[o_phone] = phone
                phone = self.phone_cache[o_phone]
            phones.append(phone)
            durations.append(
                int(
                    np.round(e * self.config.sampling_rate / self.config.hop_length)
                    - np.round(s * self.config.sampling_rate / self.config.hop_length)
                )
            )

        if start >= end:
            self.empty_textgrids += 1
            return None

        return (
            phones,
            durations,
            start,
            end,
            item["wav"],
            str(item["speaker"]).split("/")[-1],
            item["transcript"],
            Path(item["wav"]).name,
        )

    def _generate_examples(self, ds):
        j = 0
        for i, row in ds.iterrows():
            # 10kB is the minimum size of a wav file for our purposes
            if Path(row["audio"]).stat().st_size >= 10_000:
                result = {
                    "id": row["basename"],
                    "speaker": row["speaker"],
                    "text": row["text"],
                    "start": row["start"],
                    "end": row["end"],
                    "phones": row["phones"],
                    "phone_durations": row["duration"],
                    "audio": str(row["audio"]),
                }
                yield j, result
                j += 1