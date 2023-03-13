from dataclasses import dataclass
import json

from metts.dataset.data_collator import MeTTSCollator, VocoderCollator
from metts.dataset.dummy import DummyDataset
from metts.tts.diffusion_vocoder import Vocoder, VocoderConfig
from metts.tts.consistency_predictor import ConformerConsistencyPredictor, ConformerConsistencyPredictorWithDVector, ConsistencyPredictorConfig
from metts.dataset.measure import PitchMeasure, EnergyMeasure, SRMRMeasure, SNRMeasure
from torch.utils.data import DataLoader
from metts.hf.custom_trainer import Trainer
import lco
from transformers.trainer import TrainingArguments
from transformers import HfArgumentParser
from datasets import load_dataset
import torch
import transformers

def compute_metrics():
    model = trainer._wrap_model(trainer.model, training=False)
    device = model.device
    prc_index = trainer.args.process_index
    if prc_index == 0:
        eval_data = dev
        dl = DataLoader(
            eval_data,
            batch_size=lco["evaluation"]["batch_size"],
            collate_fn=dev_collator.collate_fn,
            shuffle=False,
            num_workers=lco["evaluation"]["num_workers"],
        )
        measure_order = ["pitch", "energy", "srmr", "snr"]
        log_dict = []
        for i, item in enumerate(dl):
            result = model(item["mel"].to(device), item["dvector"].to(device))
            log_dict.append(result["loss_dict"])
        log_dict = {k: torch.stack([d[k] for d in log_dict]).mean().item() for k in log_dict[0].keys()}

        return log_dict

def main(index):
    global trainer, dev, dev_collator
    training_args = HfArgumentParser(TrainingArguments).parse_json_file("config/trainer.json")[0]

    #dataset = DummyDataset()
    dev = load_dataset("metts/dataset/dataset.py", "libritts", split="dev[:50%]", keep_in_memory=True)
    train = load_dataset("metts/dataset/dataset.py", "libritts", split="train", keep_in_memory=True)
    speaker2idx = json.load(open("data/speaker2idx.json"))
    phone2idx = json.load(open("data/phone2idx.json"))
    measure_stats = json.load(open("data/measure_stats.json"))

    speaker2idx = json.load(open("data/speaker2idx.json"))
    phone2idx = json.load(open("data/phone2idx.json"))
    measure_stats = json.load(open("data/measure_stats.json"))

    dev_collator = MeTTSCollator(
        speaker2idx=speaker2idx,
        phone2idx=phone2idx,
        measure_stats=measure_stats,
        keys=["mel", "measures", "dvector"],
        measures=[PitchMeasure(), EnergyMeasure(), SRMRMeasure(), SNRMeasure()],
    )

    train_collator = MeTTSCollator(
        speaker2idx=speaker2idx,
        phone2idx=phone2idx,
        measure_stats=measure_stats,
        keys=["mel", "dvector"],
        measures=[],
    )

    teacher_model = ConformerConsistencyPredictor.from_pretrained("models/teacher_consistency")
    teacher_model.eval()
    model = ConformerConsistencyPredictorWithDVector(ConsistencyPredictorConfig())
    model.teacher = teacher_model

    trainer = Trainer(
        model,
        training_args,
        train_dataset=train,
        eval_dataset=dev,
        data_collator=train_collator.collate_fn,
        compute_metrics=compute_metrics,
    )

    trainer.train()

def _mp_fn(index):
    # For xla_spawn (TPUs)
    lco.init("config/config.yaml")
    main(index)

if __name__ == "__main__":
    main()