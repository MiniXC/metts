from dataclasses import dataclass
import json

from metts.dataset.data_collator import MeTTSCollator, VocoderCollator
from metts.dataset.dummy import DummyDataset
from metts.tts.diffusion_vocoder import Vocoder, VocoderConfig
from metts.dataset.measure import PitchMeasure, EnergyMeasure, SRMRMeasure, SNRMeasure
from metts.evaluation.metrics import Metrics
from metts.hf.custom_trainer import Trainer
import lco
from transformers.trainer import TrainingArguments
from transformers import HfArgumentParser
from datasets import load_dataset

TRAIN = True

def main(index):
    training_args = HfArgumentParser(TrainingArguments).parse_json_file("config/trainer.json")[0]

    print(f"Hello TPU core {index}!")
    #dataset = DummyDataset()
    dev  = load_dataset("metts/dataset/dataset.py", "libritts", split="dev[:1%]")
    train = load_dataset("metts/dataset/dataset.py", "libritts", split="train")
    speaker2idx = json.load(open("data/speaker2idx.json"))
    phone2idx = json.load(open("data/phone2idx.json"))
    measure_stats = json.load(open("data/measure_stats.json"))

    collator = VocoderCollator(
        include_audio=lco["evaluation"]["save_audio"],
    )

    model = Vocoder(VocoderConfig())

    #model = Vocoder.from_pretrained("output/checkpoint-74460")
    #model = model.to(dtype=torch.bfloat16)

    metrics = Metrics(
        dev,
        collator,
        save_audio=lco["evaluation"]["save_audio"],
        batch_size=lco["evaluation"]["batch_size"],
    )

    trainer = Trainer(
        model,
        training_args,
        train_dataset=train,
        eval_dataset=dev,
        data_collator=collator.collate_fn,
        compute_metrics=metrics.compute_metrics,
    )

    metrics.set_trainer(trainer)

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