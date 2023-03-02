from dataclasses import dataclass
import json

from metts.dataset.data_collator import MeTTSCollator, VocoderCollator
from metts.dataset.dummy import DummyDataset
from metts.tts.diffusion_vocoder import Vocoder, VocoderConfig
from metts.dataset.measure import PitchMeasure, EnergyMeasure, SRMRMeasure, SNRMeasure
from metts.evaluation.metrics import Metrics
# from metts.hf.custom_trainer import Trainer
import lco
from transformers.trainer import TrainingArguments
from transformers import HfArgumentParser, Trainer
from datasets import load_dataset
import torch
import transformers

TRAIN = True

def main(index):
    training_args = HfArgumentParser(TrainingArguments).parse_json_file("config/trainer.json")[0]

    #dataset = DummyDataset()
    dev = load_dataset("metts/dataset/dataset.py", "libritts", split="dev[:1%]", keep_in_memory=True)
    train = load_dataset("metts/dataset/dataset.py", "libritts", split="train", keep_in_memory=True)
    speaker2idx = json.load(open("data/speaker2idx.json"))
    phone2idx = json.load(open("data/phone2idx.json"))
    measure_stats = json.load(open("data/measure_stats.json"))

    speaker2idx = json.load(open("data/speaker2idx.json"))
    phone2idx = json.load(open("data/phone2idx.json"))
    measure_stats = json.load(open("data/measure_stats.json"))

    collator = MeTTSCollator(
        speaker2idx=speaker2idx,
        phone2idx=phone2idx,
        measure_stats=measure_stats,
        measures=[],
    )

    model = Vocoder(VocoderConfig())

    #model = Vocoder.from_pretrained("output/checkpoint-74460")
    #model = model.to(dtype=torch.bfloat16)

    # metrics = Metrics(
    #     dev,
    #     collator,
    #     batch_size=lco["evaluation"]["batch_size"],
    # )

    # adamw = torch.optim.AdamW(model.parameters(), lr=lco["diffusion_vocoder"]["lr"])

    trainer = Trainer(
        model,
        training_args,
        train_dataset=train,
        eval_dataset=dev,
        data_collator=collator.collate_fn,
        # compute_metrics=metrics.compute_metrics,
        # optimizers=(adamw, transformers.get_constant_schedule(adamw)),
    )

    # metrics.set_trainer(trainer)

    if TRAIN:
        trainer.train()
    else:
        trainer.evaluate()

def _mp_fn(index):
    # For xla_spawn (TPUs)
    lco.init("config/config.yaml")
    main(index)

if __name__ == "__main__":
    main()