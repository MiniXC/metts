from dataclasses import dataclass

from metts.dataset.data_collator import MeTTSCollator
from metts.dataset.dummy import DummyDataset
from metts.tts.model import MeTTS
from metts.dataset.measure import PitchMeasure, EnergyMeasure, SRMRMeasure, SNRMeasure
import lco
from transformers.trainer import Trainer, TrainingArguments
from transformers import HfArgumentParser

@dataclass
class Args:
    load_data: bool = False

def main(index):
    (args, training_args,) = HfArgumentParser([Args, TrainingArguments]).parse_json_file("config/trainer.json")

    print(f"Hello TPU core {index}! {args}")
    dataset = DummyDataset()

    model = MeTTS()

    trainer = Trainer(model, training_args, train_dataset=dataset, eval_dataset=dataset)
    trainer.train()

def _mp_fn(index):
    # For xla_spawn (TPUs)
    lco.init("config/config.yaml")
    main(index)

if __name__ == "__main__":
    main()