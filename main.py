from dataclasses import dataclass
import json

from metts.dataset.data_collator import FastSpeechWithConsistencyCollator
from metts.tts.model import FastSpeechWithConsistency, MeTTSConfig
from metts.tts.consistency_predictor import ConformerConsistencyPredictorWithDVector
from metts.dataset.measure import PitchMeasure, EnergyMeasure, SRMRMeasure, SNRMeasure
import lco
from torch.utils.data import DataLoader
from transformers.trainer import TrainingArguments
from metts.hf.custom_trainer import Trainer
from transformers import HfArgumentParser
from datasets import load_dataset
import torch

def compute_metrics():
    model = trainer._wrap_model(trainer.model, training=False)
    device = model.device
    prc_index = trainer.args.process_index
    if prc_index == 0:
        eval_data = dev_global
        dl = DataLoader(
            eval_data,
            batch_size=lco["evaluation"]["batch_size"],
            collate_fn=collator.collate_fn,
            shuffle=False,
            num_workers=lco["evaluation"]["num_workers"],
        )
        log_dict = []
        for i, item in enumerate(dl):
            result = model(
                item["phones"].to(device),
                item["phone_durations"].to(device),
                item["durations"].to(device),
                item["mel"].to(device),
                item["val_ind"].to(device),
                item["speaker"].to(device),
            )
            log_dict.append(result["loss_dict"])
        log_dict = {k: torch.stack([d[k] for d in log_dict]).mean().item() for k in log_dict[0].keys()}

        return log_dict

def main(index):
    global trainer, dev_global, collator
    training_args = HfArgumentParser(TrainingArguments).parse_json_file("config/trainer.json")[0]

    train = load_dataset("metts/dataset/dataset.py", "libritts", split="train")
    dev = load_dataset("metts/dataset/dataset.py", "libritts", split="dev[:1%]")
    dev_global = load_dataset("metts/dataset/dataset.py", "libritts", split="dev[:50%]")
    speaker2idx = json.load(open("data/speaker2idx.json"))
    phone2idx = json.load(open("data/phone2idx.json"))
    measure_stats = json.load(open("data/measure_stats.json"))
    
    collator = FastSpeechWithConsistencyCollator(
        speaker2idx=speaker2idx,
        phone2idx=phone2idx,
        measure_stats=measure_stats,
        keys="all",
    )

    consistency_net = ConformerConsistencyPredictorWithDVector.from_pretrained("models/consistencynet_small")
    model = FastSpeechWithConsistency.from_pretrained("models/baseline_tts_dvec", consistency_net=consistency_net)
    #model = FastSpeechWithConsistency(MeTTSConfig(), consistency_net=consistency_net)

    trainer = Trainer(
        model,
        training_args,
        train_dataset=train,
        eval_dataset=dev,
        data_collator=collator.collate_fn,
        compute_metrics=compute_metrics,
    )
    trainer.train()

def _mp_fn(index):
    # For xla_spawn (TPUs)
    lco.init("config/config.yaml")
    main(index)

if __name__ == "__main__":
    main()