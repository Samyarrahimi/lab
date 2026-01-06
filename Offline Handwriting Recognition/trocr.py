import os
import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm
from jiwer import cer, wer

import torch
from torch.utils.data import Dataset

from transformers import (
    TrOCRProcessor,
    VisionEncoderDecoderModel,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    default_data_collator,
)

# ================================================================
# --- Hugging Face Patch (avoids 404s for templates) ---
# ================================================================
from transformers.utils import hub
def safe_list_repo_templates(*args, **kwargs):
    return []
hub.list_repo_templates = safe_list_repo_templates
os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
os.environ["TRANSFORMERS_NO_ADDITIONAL_CHAT_TEMPLATES"] = "1"

# ================================================================
# --- Metrics ---
# ================================================================

def compute_metrics(preds, truths):
    c = cer(truths, preds)
    w = wer(truths, preds)

    return {
        "CER": c,
        "1-CER": 1.0 - c,
        "WER": w,
        "1-WER": 1.0 - w,
        "Exact Match ACC": np.mean(
            [p.strip().lower() == t.strip().lower()
             for p, t in zip(preds, truths)]
        ),
    }

# ================================================================
# --- Dataset (your preprocessing preserved) ---
# ================================================================

class HandwritingDataset(Dataset):
    """
    - CSV must have 'FILENAME', 'IDENTITY'
    - Removes NaNs + 'unreadable'
    - Crop top-left 64×256, pad with white
    - Convert grayscale → RGB
    """
    def __init__(self, csv_path, images_dir, processor, max_target_length=32):
        df = pd.read_csv(csv_path)
        df = df.dropna(subset=["FILENAME", "IDENTITY"])
        df["IDENTITY"] = df["IDENTITY"].astype(str)
        df = df[df["IDENTITY"].str.strip().str.lower() != "unreadable"].reset_index(drop=True)
        self.df = df
        self.images_dir = images_dir
        self.processor = processor
        self.crop_h, self.crop_w = 64, 256
        self.max_target_length = max_target_length

    def __len__(self):
        return len(self.df)

    def _crop_pad_top_left(self, img):
        arr = np.array(img.convert("L"), dtype=np.uint8)
        h, w = arr.shape[:2]
        crop = arr[:min(h, self.crop_h), :min(w, self.crop_w)]
        out = np.ones((self.crop_h, self.crop_w), dtype=np.uint8) * 255
        out[:crop.shape[0], :crop.shape[1]] = crop
        return Image.fromarray(out)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = os.path.join(self.images_dir, row["FILENAME"])
        img = Image.open(img_path).convert("RGB")
        gray_crop = self._crop_pad_top_left(img)
        rgb = Image.merge("RGB", [gray_crop, gray_crop, gray_crop])
        text = row["IDENTITY"]

        # Encode both image and text
        encoding = self.processor(images=rgb, return_tensors="pt")
        labels = self.processor.tokenizer(
            text,
            padding="max_length",
            truncation=True,
            max_length=self.max_target_length,
            return_tensors="pt",
        ).input_ids
        labels[labels == self.processor.tokenizer.pad_token_id] = -100

        return {
            "pixel_values": encoding.pixel_values.squeeze(),
            "labels": labels.squeeze(),
        }

# ================================================================
# --- Load Processor + Model ---
# ================================================================

processor = TrOCRProcessor.from_pretrained("microsoft/trocr-small-handwritten")
model = VisionEncoderDecoderModel.from_pretrained("microsoft/trocr-small-handwritten")

model.config.decoder_start_token_id = processor.tokenizer.bos_token_id
model.config.pad_token_id = processor.tokenizer.pad_token_id
model.config.vocab_size = model.config.decoder.vocab_size

# ================================================================
# --- Datasets ---
# ================================================================

train_ds = HandwritingDataset(
    r"C:\Users\samya\Desktop\lab\lab\data\handwriting_recognition\written_name_train_v2.csv",
    r"C:\Users\samya\Desktop\lab\lab\data\handwriting_recognition\train_v2\train",
    processor,
)
#train_ds.df = train_ds.df.sample(frac=0.5, random_state=42).reset_index(drop=True)

val_ds = HandwritingDataset(
    r"C:\Users\samya\Desktop\lab\lab\data\handwriting_recognition\written_name_validation_v2.csv",
    r"C:\Users\samya\Desktop\lab\lab\data\handwriting_recognition\validation_v2\validation",
    processor,
)
#val_ds.df = val_ds.df.sample(frac=0.2, random_state=42).reset_index(drop=True)


test_ds = HandwritingDataset(
    r"C:\Users\samya\Desktop\lab\lab\data\handwriting_recognition\written_name_test_v2.csv",
    r"C:\Users\samya\Desktop\lab\lab\data\handwriting_recognition\test_v2\test",
    processor,
)
#test_ds.df = test_ds.df.sample(frac=0.2, random_state=42).reset_index(drop=True)


# ================================================================
# --- Training Arguments ---
# ================================================================

training_args = Seq2SeqTrainingArguments(
    output_dir="./trocr_finetuned",
    eval_strategy="steps",
    save_strategy="steps",
    logging_strategy="steps",
    save_steps=10000,
    logging_steps=10000,
    learning_rate=1e-4,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=3,
    predict_with_generate=True,
    fp16=True,
    save_total_limit=2,
    load_best_model_at_end=True,
    report_to="none",
    gradient_checkpointing=True,
)

# ================================================================
# --- Trainer ---
# ================================================================


def compute_metrics_hf(eval_pred):
    predictions, labels = eval_pred

    # 1. FIX: Replace -100 in predictions with pad_token_id to prevent OverflowError
    predictions = np.where(predictions != -100, predictions, processor.tokenizer.pad_token_id)
    
    # 2. Decode predictions
    decoded_preds = processor.batch_decode(
        predictions, skip_special_tokens=True
    )

    # 3. FIX: Replace -100 in labels (standard cleanup for calculating metrics)
    labels = np.where(labels != -100, labels, processor.tokenizer.pad_token_id)
    
    # 4. Decode labels
    decoded_labels = processor.batch_decode(
        labels, skip_special_tokens=True
    )

    # 5. Pass the clean strings to your existing helper function
    return compute_metrics(decoded_preds, decoded_labels)
    
trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    train_dataset=train_ds,
    eval_dataset=val_ds,
    processing_class=processor,
    data_collator=default_data_collator,
    compute_metrics=compute_metrics_hf,
)

# ================================================================
# --- Train ---
# ================================================================
trainer.train()
trainer.save_model("./trocr_finetuned")
processor.save_pretrained("./trocr_finetuned")

# ================================================================
# --- Evaluate on Test Set ---
# ================================================================
# ================================================================
# --- Evaluate on Test Set ---
# ================================================================
predictions = trainer.predict(test_ds)

# 1. FIX: Clean predictions before decoding
clean_preds = np.where(predictions.predictions != -100, 
                       predictions.predictions, 
                       processor.tokenizer.pad_token_id)

decoded_preds = processor.batch_decode(clean_preds, skip_special_tokens=True)

# 2. FIX: Clean labels before decoding
clean_labels = np.where(predictions.label_ids != -100, 
                        predictions.label_ids, 
                        processor.tokenizer.pad_token_id)

decoded_labels = processor.batch_decode(clean_labels, skip_special_tokens=True)

# 3. Use your helper function to get the final numbers
metrics = compute_metrics(decoded_preds, decoded_labels)

print("\n=== Test Metrics ===")
for k, v in metrics.items():
    print(f"{k}: {v:.6f}")