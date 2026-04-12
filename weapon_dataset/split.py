import os
import random
import shutil

train_img = "train/images"
train_lbl = "train/labels"
val_img = "valid/images"
val_lbl = "valid/labels"

images = os.listdir(train_img)
random.shuffle(images)

split_size = int(len(images) * 0.15)  # 15% validation
val_images = images[:split_size]

for img in val_images:
    # Move image
    shutil.move(os.path.join(train_img, img),
                os.path.join(val_img, img))

    # Move corresponding label
    label = os.path.splitext(img)[0] + ".txt"
    shutil.move(os.path.join(train_lbl, label),
                os.path.join(val_lbl, label))

print("✅ Validation split created successfully!")