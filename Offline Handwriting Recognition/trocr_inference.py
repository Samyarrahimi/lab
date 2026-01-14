import os
import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm
from jiwer import cer, wer

import torch
from torch.utils.data import Dataset, DataLoader

from transformers import (
    TrOCRProcessor,
    VisionEncoderDecoderModel,
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

        # Encode image only (no labels needed for inference)
        encoding = self.processor(images=rgb, return_tensors="pt")

        return {
            "pixel_values": encoding.pixel_values.squeeze(),
            "ground_truth": text,  # Keep for metrics calculation
        }

# ================================================================
# --- Load Pretrained Model + Processor ---
# ================================================================

print("Loading pretrained model and processor from Hugging Face...")
model_name = "microsoft/trocr-small-handwritten"
processor = TrOCRProcessor.from_pretrained(model_name)
model = VisionEncoderDecoderModel.from_pretrained(model_name)

model.config.decoder_start_token_id = processor.tokenizer.bos_token_id
model.config.pad_token_id = processor.tokenizer.pad_token_id
model.config.vocab_size = model.config.decoder.vocab_size

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
model.eval()
print(f"Model loaded on device: {device}")

# ================================================================
# --- Load Test Dataset ---
# ================================================================

test_ds = HandwritingDataset(
    r"C:\Users\samya\Desktop\lab\lab\Offline Handwriting Recognition\data\handwriting_recognition\written_name_test_v2.csv",
    r"C:\Users\samya\Desktop\lab\lab\Offline Handwriting Recognition\data\handwriting_recognition\test_v2\test",
    processor,
)

print(f"Test dataset size: {len(test_ds)}")

# ================================================================
# --- Run Inference ---
# ================================================================

batch_size = 32
test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

all_predictions = []
all_ground_truths = []

print("Running inference on test set...")
with torch.no_grad():
    for batch in tqdm(test_loader, desc="Inferencing"):
        pixel_values = batch["pixel_values"].to(device)

        # Generate predictions
        generated_ids = model.generate(pixel_values)

        # Decode predictions
        generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)

        all_predictions.extend(generated_text)
        all_ground_truths.extend(batch["ground_truth"])

# ================================================================
# --- Calculate Metrics ---
# ================================================================

metrics = compute_metrics(all_predictions, all_ground_truths)

print("\n=== Test Set Inference Results ===")
for k, v in metrics.items():
    print(f"{k}: {v:.6f}")

# ================================================================
# --- Save Predictions (Optional) ---
# ================================================================

results_df = pd.DataFrame({
    "filename": test_ds.df["FILENAME"].values,
    "ground_truth": all_ground_truths,
    "prediction": all_predictions,
})

output_path = "trocr_inference_results.csv"
results_df.to_csv(output_path, index=False)
print(f"\nPredictions saved to: {output_path}")

# ================================================================
# --- Display Sample Predictions ---
# ================================================================

print("\n=== Sample Predictions (first 10) ===")
for i in range(min(10, len(all_predictions))):
    match = "✓" if all_predictions[i].strip().lower() == all_ground_truths[i].strip().lower() else "✗"
    print(f"{match} GT: '{all_ground_truths[i]}' | Pred: '{all_predictions[i]}'")
