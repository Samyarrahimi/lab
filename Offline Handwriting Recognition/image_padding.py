from PIL import Image
import os

def pad_to_size(input_path, output_path, target_size=(512, 512), fill_color=(255,255,255)):
    """
    Pads an image so that its size becomes target_size (width, height),
    by centering the original image and filling background with fill_color.
    If the original is larger in any dimension, it will resize proportionally
    to fit within target_size first (optional).
    """
    img = Image.open(input_path).convert("RGB")
    orig_w, orig_h = img.size
    tgt_w, tgt_h = target_size

    # Compute scale factor if resizing is needed so it fits within target size
    scale = min(tgt_w/orig_w, tgt_h/orig_h, 1.0)
    if scale < 1.0:
        new_w = int(orig_w * scale)
        new_h = int(orig_h * scale)
        img = img.resize((new_w, new_h), resample=Image.BILINEAR)
        orig_w, orig_h = img.size

    # Create white background canvas
    new_img = Image.new("RGB", (tgt_w, tgt_h), fill_color)

    # Compute top-left coordinates to paste the original image (center it)
    paste_x = (tgt_w - orig_w) // 2
    paste_y = (tgt_h - orig_h) // 2

    new_img.paste(img, (paste_x, paste_y))

    # Save
    new_img.save(output_path)

def process_folder(input_folder, output_folder, target_size=(512,512)):
    os.makedirs(output_folder, exist_ok=True)
    for fname in os.listdir(input_folder):
        if fname.lower().endswith((".png", ".jpg", ".jpeg", ".bmp")):
            input_path = os.path.join(input_folder, fname)
            output_path = os.path.join(output_folder, fname)
            pad_to_size(input_path, output_path, target_size=target_size)
            print(f"Padded {fname} → {target_size}")

if __name__ == "__main__":
    # Example usage:
    in_folder = r"C:\Users\samya\Desktop\lab\lab\data\handwriting_recognition\test_v2\test"
    out_folder = r"C:\Users\samya\Desktop\lab\lab\data\handwriting_recognition\test_v2\test_padded"
    process_folder(in_folder, out_folder, target_size=(512,512))
