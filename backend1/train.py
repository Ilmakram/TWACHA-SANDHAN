# train.py: TensorFlow/Keras CNN training pipeline for multi-class skin disease classification

import os, random, json, argparse
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import applications
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, LearningRateScheduler

# Set random seeds for reproducibility
random.seed(42)
np.random.seed(42)
tf.random.set_seed(42)

# Parse command-line arguments
parser = argparse.ArgumentParser(description="Train CNN on multi-class skin disease dataset")
parser.add_argument("--data_dir", type=str, default="IMG_CLASSES",
                    help="Root directory of images (subfolders per class)")
parser.add_argument("--img_size", type=int, default=224, help="Image height/width (default 224)")
parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
parser.add_argument("--epochs", type=int, default=20, help="Max training epochs")
parser.add_argument("--learning_rate", type=float, default=1e-3, help="Initial learning rate")
parser.add_argument("--model", choices=["ResNet50","MobileNetV2"], default="ResNet50",
                    help="Pretrained model type")
parser.add_argument("--freeze_backbone", action="store_true",
                    help="Freeze base model (feature-extractor mode)")
parser.add_argument("--train_split", type=float, default=0.7, help="Train split fraction")
parser.add_argument("--val_split", type=float, default=0.15, help="Validation split fraction")
parser.add_argument("--test_split", type=float, default=0.15, help="Test split fraction")
parser.add_argument("--use_cuda", action="store_true", help="Use GPU if available")
parser.add_argument("--save_dir", type=str, default="output", help="Directory to save model and outputs")
args = parser.parse_args()

# Enable mixed precision if GPU is used (speeds up training on modern GPUs)【50†L311-L320】
if args.use_cuda and tf.config.list_physical_devices('GPU'):
    keras.mixed_precision.set_global_policy('mixed_float16')

# Create output directory
os.makedirs(args.save_dir, exist_ok=True)

# Gather image file paths and labels
class_names = sorted(os.listdir(args.data_dir))
file_paths = []
labels = []
for label, cls in enumerate(class_names):
    cls_dir = os.path.join(args.data_dir, cls)
    for fname in os.listdir(cls_dir):
        if fname.lower().endswith((".jpg",".png",".jpeg")):
            file_paths.append(os.path.join(cls_dir, fname))
            labels.append(label)
file_paths = np.array(file_paths)
labels = np.array(labels)

# Stratified train/validation/test split using sklearn (maintain class proportions)【3†L719-L727】
from sklearn.model_selection import train_test_split
paths_train, paths_temp, y_train, y_temp = train_test_split(
    file_paths, labels, test_size=(1-args.train_split), random_state=42, stratify=labels)
val_frac = args.val_split / (args.val_split + args.test_split)
paths_val, paths_test, y_val, y_test = train_test_split(
    paths_temp, y_temp, test_size=(args.test_split/(args.val_split+args.test_split)),
    random_state=42, stratify=y_temp)

print(f"Train/Val/Test sizes: {len(paths_train)}/{len(paths_val)}/{len(paths_test)} images")

# Function to load and preprocess images
def load_and_preprocess(path, label, img_size=args.img_size):
    image = tf.io.read_file(path)  # read image

    # decode based on format (important fix)
    image = tf.image.decode_jpeg(image, channels=3)

    image = tf.image.resize(image, [img_size, img_size])  # resize
    image = tf.cast(image, tf.float32) / 255.0  # normalize

    return image, label

# Data augmentation for training: random flips, rotations, color jitter
def augment(image, label):
    image = tf.image.random_flip_left_right(image)
    image = tf.image.random_flip_up_down(image)
    image = tf.image.random_brightness(image, 0.1)
    image = tf.image.random_contrast(image, 0.9, 1.1)
    return image, label

# Build tf.data.Dataset pipelines
train_ds = tf.data.Dataset.from_tensor_slices((paths_train, y_train))
train_ds = train_ds.shuffle(len(paths_train), seed=42).map(
    load_and_preprocess, num_parallel_calls=tf.data.AUTOTUNE).map(
    augment, num_parallel_calls=tf.data.AUTOTUNE).batch(args.batch_size).prefetch(tf.data.AUTOTUNE)
val_ds = tf.data.Dataset.from_tensor_slices((paths_val, y_val)).map(
    load_and_preprocess, num_parallel_calls=tf.data.AUTOTUNE).batch(args.batch_size).prefetch(tf.data.AUTOTUNE)
test_ds = tf.data.Dataset.from_tensor_slices((paths_test, y_test)).map(
    load_and_preprocess, num_parallel_calls=tf.data.AUTOTUNE).batch(args.batch_size).prefetch(tf.data.AUTOTUNE)

# Compute class weights to handle imbalance (inverse of frequency)
class_counts = np.bincount(y_train, minlength=len(class_names))
class_weights = {i: (1.0/count if count>0 else 0.0) for i, count in enumerate(class_counts)}
print("Class weights:", class_weights)

# Instantiate the selected model, excluding top layers【52†L7-L10】【51†L7-L10】
if args.model == "ResNet50":
    base_model = applications.ResNet50(weights='imagenet', include_top=False,
                                       input_shape=(args.img_size, args.img_size, 3))
elif args.model == "MobileNetV2":
    base_model = applications.MobileNetV2(weights='imagenet', include_top=False,
                                          input_shape=(args.img_size, args.img_size, 3))
else:
    raise ValueError("Unsupported model choice")
base_model.trainable = not args.freeze_backbone

# Add classification head for 10 classes
x = base_model.output
x = layers.GlobalAveragePooling2D()(x)
x = layers.Dropout(0.5)(x)
output = layers.Dense(len(class_names), activation='softmax', dtype='float32')(x)
model = keras.Model(inputs=base_model.input, outputs=output)

# Compile model with optimizer, weighted loss, and accuracy
optimizer = keras.optimizers.Adam(learning_rate=args.learning_rate)
loss_fn = keras.losses.SparseCategoricalCrossentropy()
model.compile(optimizer=optimizer, loss=loss_fn, metrics=['accuracy'])

# Callbacks: checkpoint (save best model), early stopping, LR scheduler
checkpoint_cb = ModelCheckpoint(os.path.join(args.save_dir, "best_model.h5"),
                                save_best_only=True, monitor="val_loss")
earlystop_cb = EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True)
def scheduler(epoch, lr):
    return lr * 0.1 if (epoch+1) % 10 == 0 else lr
lr_scheduler_cb = LearningRateScheduler(scheduler)

# Train the model
model.fit(train_ds,
          validation_data=val_ds,
          epochs=args.epochs,
          callbacks=[checkpoint_cb, earlystop_cb, lr_scheduler_cb],
          class_weight=class_weights)  # apply class weights for imbalance

# Evaluate on test set
test_loss, test_acc = model.evaluate(test_ds)
print(f"\nTest Accuracy: {test_acc:.4f}")

# Compute detailed metrics on test set
y_pred = np.concatenate([np.argmax(model.predict(batch[0]), axis=1) for batch in test_ds], axis=0)
y_true = y_test  # labels from splitting
from sklearn.metrics import precision_recall_fscore_support, confusion_matrix
prec, rec, f1, _ = precision_recall_fscore_support(y_true, y_pred, labels=range(len(class_names)))
print("Class-wise Precision/Recall/F1:")
for i, cls in enumerate(class_names):
    print(f"  {cls}: P={prec[i]:.3f}, R={rec[i]:.3f}, F1={f1[i]:.3f}")
conf_mat = confusion_matrix(y_true, y_pred, labels=range(len(class_names)))
print("Confusion Matrix:\n", conf_mat)

# Save final model and label mapping
model.save(os.path.join(args.save_dir, "final_model"))
label_map = {cls: idx for idx, cls in enumerate(class_names)}
with open(os.path.join(args.save_dir, "label_map.json"), "w") as f:
    json.dump(label_map, f)
print(f"Model and label map saved to {args.save_dir}")
