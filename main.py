from dataclasses import dataclass
import json

from metts.dataset.data_collator import FastSpeechWithConsistencyCollator
from metts.tts.model import FastSpeechWithConsistency, MeTTSConfig
from metts.tts.consistency_predictor import ConformerConsistencyPredictorWithDVector
from metts.dataset.measure import PitchMeasure, EnergyMeasure, SRMRMeasure, SNRMeasure
import lco
from torch.utils.data import DataLoader
from transformers.trainer import TrainingArguments, Trainer
#from metts.hf.custom_trainer import Trainer
from transformers import HfArgumentParser
from datasets import load_dataset
import torch
from tqdm.auto import tqdm

class CustomTrainer(Trainer):
    # evaluation loop
    def evaluate(self, eval_dataset=None, ignore_keys=None, metric_key_prefix="eval"):
        # check if on main process
        prc_index = self.args.process_index
        if prc_index == 0:
            losses = []
            dataloader = self.get_eval_dataloader(eval_dataset)
            model = self._wrap_model(self.model, training=False, dataloader=dataloader)
            model.eval()
            for i, inputs in tqdm(enumerate(dataloader), total=len(dataloader), desc="Evaluating"):
                inputs["return_loss"] = True
                loss, _, _ = self.prediction_step(model, inputs, prediction_loss_only=True)
                losses.append(loss)
            log_dict = {
                "eval/loss": torch.stack(losses).mean().item()
            }
            self.log(log_dict)

def main(index):
    global trainer, dev_global, collator
    training_args = HfArgumentParser(TrainingArguments).parse_json_file("config/trainer.json")[0]

    train = load_dataset("metts/dataset/dataset.py", "libritts", split="train")
    dev = load_dataset("metts/dataset/dataset.py", "libritts", split="dev[:10%]")
    speaker2idx = json.load(open("data/speaker2idx.json"))
    phone2idx = json.load(open("data/phone2idx.json"))
    measure_stats = json.load(open("data/measure_stats.json"))
    
    collator = FastSpeechWithConsistencyCollator(
        speaker2idx=speaker2idx,
        phone2idx=phone2idx,
        measure_stats=measure_stats,
        keys="all",
    )

    consistency_net = ConformerConsistencyPredictorWithDVector.from_pretrained("pretrained_models/consistency")
    model = FastSpeechWithConsistency.from_pretrained("output/checkpoint-16000", consistency_net=consistency_net, ignore_mismatched_sizes=True) # (MeTTSConfig(), consistency_net=consistency_net)

    trainer = CustomTrainer(
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