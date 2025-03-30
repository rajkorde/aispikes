import torch
from datasets import load_dataset
from trl import SFTConfig, SFTTrainer


class SFTConfigWithMPS(SFTConfig):
    @property
    def device(self) -> torch.device:
        if torch.backends.mps.is_available():
            return torch.device("mps")
        else:
            return torch.device("cpu")


dataset = load_dataset("stanfordnlp/imdb", split="train")
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

training_args = SFTConfig(max_seq_length=512, output_dir="/tmp")
trainer = SFTTrainer("facebook/opt-350m", train_dataset=dataset, args=training_args)

trainer.train()
