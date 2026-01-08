import os
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForTokenClassification
import pytorch_lightning as pl
import wandb

from ner_labels import LABEL2ID, NUM_LABELS

MODEL_NAME = "dmis-lab/biobert-base-cased-v1.1"


class ClinicalNERDataset(Dataset):
    def __init__(self, samples, tokenizer, max_len=256):
        self.samples = samples
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        tokens = self.samples[idx]["tokens"]
        labels = self.samples[idx]["labels"]

        encoding = self.tokenizer(
            tokens,
            truncation=True,
            padding="max_length",
            max_length=self.max_len,
            is_split_into_words=True,
            return_tensors="pt",
        )

        word_ids = encoding.word_ids(batch_index=0)
        label_ids = []

        previous_word = None
        for word_id in word_ids:
            if word_id is None:
                label_ids.append(-100)
            elif word_id != previous_word:
                label_ids.append(LABEL2ID[labels[word_id]])
            else:
                label_ids.append(LABEL2ID[labels[word_id]])
            previous_word = word_id

        return {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "labels": torch.tensor(label_ids),
        }


class BioBERTNER(pl.LightningModule):
    def __init__(self, lr=2e-5):
        super().__init__()
        self.model = AutoModelForTokenClassification.from_pretrained(
            MODEL_NAME,
            num_labels=NUM_LABELS,
        )
        self.lr = lr

    def forward(self, **inputs):
        return self.model(**inputs)

    def training_step(self, batch, batch_idx):
        loss = self(**batch).loss
        self.log("train_loss", loss)
        return loss

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=self.lr)


if __name__ == "__main__":
    wandb.init(project="clinical-doc-ai", name="biobert-ner")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    # VERY SMALL SAMPLE DATA (EXTEND LATER)
    samples = [
        {
            "tokens": ["Patient", "has", "chest", "pain"],
            "labels": ["O", "O", "B-SYMPTOM", "I-SYMPTOM"],
        },
        {
            "tokens": ["Prescribed", "aspirin"],
            "labels": ["O", "B-MEDICATION"],
        },
    ]

    dataset = ClinicalNERDataset(samples, tokenizer)
    loader = DataLoader(dataset, batch_size=2)

    model = BioBERTNER()

    trainer = pl.Trainer(max_epochs=2)
    trainer.fit(model, loader)

    SAVE_DIR = "services/nlp_inference_service/models"
    os.makedirs(SAVE_DIR, exist_ok=True)

    torch.save(
        model.model.state_dict(),
        os.path.join(SAVE_DIR, "biobert_ner_weights.pt"),
    )

    print("NER model saved")