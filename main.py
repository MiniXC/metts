from dataclasses import dataclass
import json

from metts.dataset.data_collator import MeTTSCollator
from metts.dataset.dummy import DummyDataset
from metts.tts.model import MeTTS, MeTTSConfig
from metts.dataset.measure import PitchMeasure, EnergyMeasure, SRMRMeasure, SNRMeasure
import lco
from transformers.trainer import Trainer, TrainingArguments
from transformers import HfArgumentParser
from datasets import load_dataset

def main(index):
    training_args = HfArgumentParser(TrainingArguments).parse_json_file("config/trainer.json")[0]

    print(f"Hello TPU core {index}!")
    #dataset = DummyDataset()
    dataset = load_dataset("metts/dataset/dataset.py", "libritts")
    train = dataset["train"]
    dev = dataset["dev"]
    speaker2idx = json.load(open("data/speaker2idx.json"))
    phone2idx = json.load(open("data/phone2idx.json"))
    collator = MeTTSCollator(
        speaker2idx=speaker2idx,
        phone2idx=phone2idx,
        measures=[
            PitchMeasure(),
        ],
    )

    model = MeTTS(MeTTSConfig())

    trainer = Trainer(
        model,
        training_args,
        train_dataset=train,
        eval_dataset=dev,
        data_collator=collator.collate_fn,
    )
    trainer.train()

def _mp_fn(index):
    # For xla_spawn (TPUs)
    lco.init("config/config.yaml")
    main(index)

if __name__ == "__main__":
    main()