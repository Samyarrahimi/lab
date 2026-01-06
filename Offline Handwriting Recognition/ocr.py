from transformers import AutoModel, AutoTokenizer
import torch
import os
import csv

os.environ["CUDA_VISIBLE_DEVICES"] = '0'

model_name = 'deepseek-ai/DeepSeek-OCR'
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
model = AutoModel.from_pretrained(
    model_name,
    _attn_implementation='eager',
    trust_remote_code=True,
    use_safetensors=True
)
model = model.eval().cuda().to(torch.bfloat16)

prompt = """
<image>
Read the image and output ONLY BASIC ASCII LETTERS AND NUMBERS (no accents). What text do you see in the image?
"""

input_folder  = r'C:\Users\samya\Desktop\lab\lab\data\handwriting_recognition\test_v2\test_padded'
output_folder = r'C:\Users\samya\Desktop\lab\lab\data\handwriting_recognition\test_v2\ocr_outputs'
os.makedirs(output_folder, exist_ok=True)

csv_path = os.path.join(output_folder, "ocr_results.csv")
# Create CSV with header if it does not exist
if not os.path.exists(csv_path):
    with open(csv_path, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["FILENAME", "IDENTITY"])

for fname in os.listdir(input_folder)[:10]:
    if not fname.lower().endswith(('.jpg','.jpeg','.png','.bmp')):
        continue
    image_file = os.path.join(input_folder, fname)
    
    res = model.infer(
        tokenizer,
        prompt=prompt,
        image_file=image_file,
        output_path=output_folder,
        base_size=512,
        image_size=512,
        crop_mode=False,
        save_results=False,
        test_compress=False,
        eval_mode=True
    )
    
    if res is None:
        txt_path = os.path.join(output_folder, fname + '.txt')
        if os.path.exists(txt_path):
            with open(txt_path, 'r', encoding='utf-8') as f:
                text = f.read().strip()
        else:
            text = ""
    else:
        if isinstance(res, dict) and 'text' in res:
            text = res['text']
        else:
            text = str(res).strip()
    
    # Filter to ASCII and uppercase
    filtered = ''.join(ch for ch in text if ch.isascii()).upper()
    
    # Save individual file
    out_path = os.path.join(output_folder, fname + '_ocr.txt')
    with open(out_path, 'w', encoding='utf-8') as f:
        f.write(f"IMAGE_FILE: {fname}\n")
        f.write(f"OCR_TEXT: {filtered}\n")
    
    # Append to CSV
    with open(csv_path, 'a', newline='', encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow([fname, filtered])
    
    print(f"{fname} → OCR text saved to {out_path} and appended to CSV.")
