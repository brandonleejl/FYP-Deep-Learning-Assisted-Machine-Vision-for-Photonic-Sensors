# FYP-Deep-Learning-Assisted-Machine-Vision-for-Photonic-Sensors

Hydrogel image pipeline for:
1. Segmenting hydrogel regions from UV images.
2. Predicting continuous pH values from segmented regions using an Enhanced Deep Learning model with Uncertainty Estimation.

## New Features

- **Multi-Task Learning:** Predicts both discrete pH class and continuous pH value simultaneously.
- **Attention Mechanisms:** Uses Convolutional Block Attention Module (CBAM) for better feature extraction.
- **Uncertainty Estimation:** Uses Monte Carlo (MC) Dropout to estimate prediction confidence.
- **Ensemble Learning:** Trains multiple models and averages their predictions for robust results.

## Project Files and Purpose

- `data_preprocessing.py`
Purpose: Dataset preparation utilities.
What it handles: image discovery, resizing/padding, CSV label template generation, mask path resolution, train/validation split, and image/mask loading helpers.

- `data_augmentation.py`
Purpose: Data augmentation only.
What it handles: image+mask augmentation for segmentation and image-only augmentation for regression.

- `model_components.py`
Purpose: Model architecture definitions.
What it handles: `MCDropout` layer, `CBAM` attention blocks, and the `build_enhanced_ph_classifier` function.

- `main.py`
Purpose: End-to-end training and evaluation script.
What it handles: dataset construction, DeepLabV3+ segmentation training, hydrogel region extraction, Ensemble training of pH regressors, and Uncertainty-aware metrics.

## How Files Interact With `main.py`

1. `main.py` imports preprocessing helpers from `data_preprocessing.py`:
- `list_image_files`, `load_labels_csv`, `split_train_val`, `read_image`, `read_mask`

2. `main.py` imports augmentation functions from `data_augmentation.py`:
- `augment_image_and_mask`
- `augment_regression_image`

3. `main.py` imports model components from `model_components.py`:
- `build_enhanced_ph_classifier`, `MCDropout`, `cbam_block`

4. During training:
- Segmentation dataset uses `read_image` + `read_mask` + `augment_image_and_mask`.
- Regression dataset uses masked images and applies `augment_regression_image`.
- An ensemble of 3 models (default) is trained sequentially.

## Function-by-Function Explanation

### `data_preprocessing.py`

- `list_image_files(image_dir)`
Scans `image_dir` for common image extensions (`png/jpg/jpeg`, upper/lower case) and returns a sorted file list.

- `create_labels_template(image_dir, output_csv)`
Creates or updates `output_csv` with columns `filename,ph`.
Automatically writes all image filenames and preserves previously entered pH values.

- `preprocess_images(source_dir, output_dir, target_size=(512, 512))`
Loads each image, converts to float `[0,1]`, resizes with padding (keeps aspect ratio), and saves PNG output to `output_dir`.

- `load_labels_csv(label_csv)`
Reads `labels.csv` into `{filename: ph_float}`.
Skips empty or invalid rows to avoid crashing.

- `get_mask_path(image_path, mask_dir)`
Builds candidate mask names for each image and returns the first existing file.
Supports Labelme JSON masks using the same image stem (e.g., `IMG_6042.json`) and PNG masks.

- `split_train_val(items, val_split=0.2, seed=42)`
Randomly splits items into train/validation sets.
Includes sanity checks so split ratio and sample count are valid.

- `read_image(path, image_size)`
Reads an RGB image tensor, sets shape, normalizes to float, resizes with bilinear interpolation.

- `read_mask(path, image_size)`
Reads mask data from Labelme JSON or image files, resizes with nearest interpolation, thresholds to binary `0/1`.

- `if __name__ == "__main__": ...`
Allows standalone preprocessing run:
- preprocess images
- auto-generate/update `labels_template.csv`

### `data_augmentation.py`

- `augment_image_and_mask(image, mask)`
Applies the same random flips to image and mask (so segmentation labels stay aligned), then applies brightness/contrast changes to image only.
Used in segmentation training.

- `augment_regression_image(image)`
Applies brightness/contrast/saturation jitter to image.
Used in regression training.

### `model_components.py`

- `MCDropout(rate)`
Custom Dropout layer that is active during inference (`training=True`), enabling Monte Carlo sampling.

- `cbam_block(x, ratio=8)`
Applies Channel and Spatial Attention to the input tensor `x`.

- `build_enhanced_ph_classifier(num_classes, input_shape)`
Builds a MobileNetV2-based model with CBAM attention, MC Dropout, and dual outputs (classification logits + regression value).

### `main.py`

- `make_seg_dataset(pairs, training)`
Builds segmentation `tf.data` pipeline from `(image_path, mask_path)` pairs.
Uses preprocessing readers and optional segmentation augmentation.

- `build_deeplabv3plus(input_shape=(512,512,3))`
Constructs a DeepLabV3+ style model (MobileNetV2 backbone + ASPP + decoder) for binary hydrogel segmentation.

- `dice_coef(y_true, y_pred)`
Dice overlap metric for segmentation quality.

- `dice_loss(y_true, y_pred)`
`1 - dice_coef`; encourages better mask overlap.

- `bce_dice_loss(y_true, y_pred)`
Combines binary cross-entropy with Dice loss for stable segmentation training.

- `build_classifier_arrays(image_paths, ph_values, seg_model, mask_threshold=0.5)`
Runs segmentation predictions, thresholds mask, multiplies image by mask to isolate hydrogel, resizes for regression input, and returns `(X, y_dict)` arrays where `y_dict` contains both classification and regression targets.

- `make_cls_dataset(x, y, training)`
Builds regression `tf.data` pipeline with optional augmentation.

- `r2_score(y_true, y_pred)`
Computes R-squared metric for regression performance.

- `main()`
End-to-end orchestration:
1. Load image paths.
2. Match images to masks.
3. Train/evaluate DeepLabV3+ segmentation model.
4. Load pH labels from CSV.
5. Build hydrogel-only regression inputs using segmentation outputs.
6. **Ensemble Training:** Train `NUM_ENSEMBLE` (3) instances of the Enhanced pH Classifier.
7. **Uncertainty Evaluation:** Load all ensemble models, perform `MC_SAMPLES` (10) forward passes per model per sample.
8. Save aggregated predictions (Mean, Uncertainty) to `results/<YYYYMMDD_HHMMSS>_test_predictions.csv`.
9. Save an Excel report with charts to `results/<YYYYMMDD_HHMMSS>_test_predictions.xlsx`.

## Expected Data Layout

```text
project_root/
  images/                     # raw hydrogel images
  masks/                      # optional: masks as Labelme JSON (IMG_xxxx.json) or *_mask.png
  labels.csv                  # columns: filename,ph
  results/                    # auto-saved training/evaluation CSV logs
  data_preprocessing.py
  data_augmentation.py
  model_components.py
  main.py
```

## Typical Usage

1. Run preprocessing and labels template generation:

```bash
python data_preprocessing.py
```

2. Fill pH values in `labels_template.csv`, then save as `labels.csv` (or edit `labels.csv` directly if it exists).

3. Put Labelme JSON files named like `IMG_6042.json` in `masks/`.
If `masks/` does not exist, `main.py` will automatically read JSON masks from `images/`.
By default, `main.py` enforces JSON annotations for every image (`REQUIRE_JSON_MASKS = True`).

4. Run training and evaluation:

```bash
python main.py
```

GPU note:
- `main.py` is configured for GPU-only execution.
- If no GPU is detected, it raises an error and stops (no CPU fallback).

Generated CSV outputs in `results/`:
- `<timestamp>.csv`: run-level summary metrics (including MAE, RMSE, R^2, Mean Uncertainty).
- `<timestamp>_test_predictions.csv`: one row per test/validation sample with:
  `filename`, `actual_ph`, `predicted_ph`, `uncertainty`, `abs_error`, `squared_error`, plus global MAE/RMSE/R^2.
- `<timestamp>_test_predictions.xlsx`: Excel report with plots:
  Actual vs Predicted pH, Residuals by sample, and Absolute Error by sample.

Note:
- Excel export requires `xlsxwriter`. Install with:
  `pip install xlsxwriter`
