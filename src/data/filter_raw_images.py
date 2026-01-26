import os, shutil

raw = "dataset/raw"
target_original = "dataset/original"
target_labels = "dataset/labels"

os.makedirs(target_original, exist_ok=True)
os.makedirs(target_labels, exist_ok=True)

if os.path.isdir(raw):
    for f in os.listdir(raw):
        # Only grab the original images, skipping the '_label' files
        if f.endswith(".bmp") and "_label" not in f:
            shutil.copy(os.path.join(raw, f), os.path.join(target_original, f))
        else:
            shutil.copy(os.path.join(raw, f), os.path.join(target_labels, f.replace("_label", "")))

print(f"Done! All original images are now in '{target_original}'.")
print(f"Done! All label images are now in '{target_labels}'.")