import os
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split

# 1. Config

SOURCE_DIR = "cell_images"
OUTPUT_DIR = "dataset_processed"
TARGET_SIZE = (128, 128)
RANDOM_SEED = 42

CLASSES = ["Parasitized", "Uninfected"]
SPLITS = ["train", "val", "test"]

# 2. All raw files and labels

print("Scanning source folder for raw images")
records = []

for class_name in CLASSES:
    class_folder = os.path.join(SOURCE_DIR, class_name)
    if not os.path.exists(class_folder):
        raise FileNotFoundError(f"Missing folder: {class_folder}.")
        
    for fname in os.listdir(class_folder):
        if fname.lower().endswith((".png", ".jpg", ".jpeg")):
            records.append({
                "filename": fname,
                "raw_path": os.path.join(class_folder, fname),
                "label": class_name
            })

df = pd.DataFrame(records)
print(f"Total raw images located: {len(df)} (Parasitized: {(df['label']=='Parasitized').sum()}, Uninfected: {(df['label']=='Uninfected').sum()})")

# 3. Split (70% Train, 15% Val, 15% Test)

print("Partitioning data into train (70%), val (15%), and test (15%)")

train_df, temp_df = train_test_split(
    df, test_size=0.30, stratify=df["label"], random_state=RANDOM_SEED
)

val_df, test_df = train_test_split(
    temp_df, test_size=0.50, stratify=temp_df["label"], random_state=RANDOM_SEED
)

train_df = train_df.assign(split="train")
val_df = val_df.assign(split="val")
test_df = test_df.assign(split="test")

full_manifest = pd.concat([train_df, val_df, test_df], ignore_index=True)


# 4. Create folders

print(f"Creating output folders in '{OUTPUT_DIR}'")
for split in SPLITS:
    for class_name in CLASSES:
        os.makedirs(os.path.join(OUTPUT_DIR, split, class_name), exist_ok=True)


# 5. Process creat resized and padded images

print(f"Resizing with padding to {TARGET_SIZE} and saving.")

processed_paths = []
total_count = len(full_manifest)

for idx, row in full_manifest.iterrows():
    # Read raw images bytes
    img_raw = tf.io.read_file(row["raw_path"])
    img = tf.image.decode_image(img_raw, channels=3, expand_animations=False)
    
    # Pad and resize
    img_padded = tf.image.resize_with_pad(
        img, 
        target_height=TARGET_SIZE[0], 
        target_width=TARGET_SIZE[1],
        method=tf.image.ResizeMethod.BILINEAR
    )
    img_uint8 = tf.cast(img_padded, tf.uint8)
    
    # Target path
    dest_path = os.path.join(OUTPUT_DIR, row["split"], row["label"], row["filename"])
    
    # Encode and save as PNG
    encoded_png = tf.io.encode_png(img_uint8)
    tf.io.write_file(dest_path, encoded_png)
    processed_paths.append(dest_path)
    
    # Progress feedback, crucial for debugging
    if (idx + 1) % 2500 == 0 or (idx + 1) == total_count:
        print(f"  Processed {idx + 1}/{total_count} images ({(idx + 1)/total_count*100:.1f}%)")

# Save manifest
full_manifest["processed_path"] = processed_paths
manifest_csv_path = os.path.join(OUTPUT_DIR, "dataset_manifest.csv")
full_manifest.to_csv(manifest_csv_path, index=False)

print(f"PREPROCESSING COMPLETE!")

