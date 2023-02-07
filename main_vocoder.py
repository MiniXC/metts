from dataclasses import dataclass
import json

from metts.dataset.data_collator import MeTTSCollator
from metts.dataset.dummy import DummyDataset
from metts.tts.diffusion_vocoder import Vocoder, VocoderConfig
from metts.dataset.measure import PitchMeasure, EnergyMeasure, SRMRMeasure, SNRMeasure
from metts.evaluation.metrics import Metrics
import lco
from transformers.trainer import Trainer, TrainingArguments
from transformers import HfArgumentParser
from datasets import load_dataset

def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = predictions[:, 0]
    labels = labels[:, 0]
    return {
        "mean_absolute_error": np.mean(np.abs(predictions - labels)),
        "mean_squared_error": np.mean((predictions - labels) ** 2),
    }

def main(index):
    training_args = HfArgumentParser(TrainingArguments).parse_json_file("config/trainer.json")[0]

    print(f"Hello TPU core {index}!")
    #dataset = DummyDataset()
    dataset = load_dataset("metts/dataset/dataset.py", "libritts")
    train = dataset["train"]
    dev = dataset["dev[:16]"]
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

    metrics = Metrics(
        dev,
        collator,
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

    trainer.train()

def _mp_fn(index):
    # For xla_spawn (TPUs)
    lco.init("config/config.yaml")
    main(index)

if __name__ == "__main__":
    main()