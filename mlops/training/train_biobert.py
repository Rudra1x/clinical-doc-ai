
import wandb
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import pytorch_lightning as pl
from sklearn.metrics import f1_score, roc_auc_score

wandb.init(project="clinical-doc-ai", name="biobert-icd-training")

from icd_labels import ICD_TO_INDEX, NUM_LABELS

MODEL_NAME = "dmis-lab/biobert-base-cased-v1.1"

class ClinicalDataset(Dataset):
    def __init__(self, samples, tokenizer, max_len=512):
        self.samples = samples
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        text = self.samples[idx]["text"]
        codes = self.samples[idx]["icd_codes"]

        labels = torch.zeros(NUM_LABELS)
        for code in codes:
            if code in ICD_TO_INDEX:
                labels[ICD_TO_INDEX[code]] = 1.0

        encoding = self.tokenizer(
            text,
            truncation=True,
            padding="max_length",
            max_length=self.max_len,
            return_tensors="pt"
        )

        return {
            "input_ids": encoding["input_ids"].squeeze(),
            "attention_mask": encoding["attention_mask"].squeeze(),
            "labels": labels
        }
class BioBERTICDModel(pl.LightningModule):
    def __init__(self, lr=2e-5):
        super().__init__()
        self.model = AutoModelForSequenceClassification.from_pretrained(
            MODEL_NAME,
            num_labels=NUM_LABELS,
            problem_type="multi_label_classification"
        )
        self.lr = lr

    def forward(self, **inputs):
        return self.model(**inputs)

    def training_step(self, batch, batch_idx):
        outputs = self(**batch)
        loss = outputs.loss
        self.log("train_loss", loss, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        outputs = self(**batch)
        logits = outputs.logits
        preds = torch.sigmoid(logits)
        return {
            "preds": preds.detach(),
            "labels": batch["labels"].detach()
        }

    def on_validation_epoch_start(self):
        self.val_preds = []
        self.val_labels = []

    def validation_step(self, batch, batch_idx):
        outputs = self(**batch)
        logits = outputs.logits
        preds = torch.sigmoid(logits)
        self.val_preds.append(preds.detach().cpu())
        self.val_labels.append(batch["labels"].detach().cpu())
        return {}

    def on_validation_epoch_end(self):
        if not hasattr(self, "val_preds") or not hasattr(self, "val_labels"):
            return
        preds = torch.cat(self.val_preds)
        labels = torch.cat(self.val_labels)
        preds_bin = (preds > 0.5).int()
        f1 = f1_score(
            labels.numpy(),
            preds_bin.numpy(),
            average="micro",
            zero_division=0
        )
        try:
            auc = roc_auc_score(labels.numpy(), preds.numpy(), average="micro")
        except:
            auc = 0.0
        self.log("val_f1", f1, prog_bar=True)
        self.log("val_auc", auc, prog_bar=True)

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=self.lr)

if __name__ == "__main__":
    wandb.init(project="clinical-doc-ai", name="biobert-icd-v1")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    # 🔧 Dummy data (replace with real discharge summaries later)
    train_samples = [
        {"text": "Patient admitted with chest pain.", "icd_codes": ["I20.9"]},
        {"text": "History of diabetes and hypertension.", "icd_codes": ["E11.9", "I10"]},
    ]

    val_samples = train_samples

    train_ds = ClinicalDataset(train_samples, tokenizer)
    val_ds = ClinicalDataset(val_samples, tokenizer)

    train_loader = DataLoader(train_ds, batch_size=2, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=2)

    model = BioBERTICDModel()

    trainer = pl.Trainer(
        max_epochs=3,
        accelerator="auto",
        log_every_n_steps=1
    )

    trainer.fit(model, train_loader, val_loader)

    trainer.save_checkpoint(
        "../../services/nlp_inference_service/models/biobert_icd.ckpt"
    )