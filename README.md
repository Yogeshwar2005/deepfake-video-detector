# Deepfake Video Detection using Deep Learning

An end-to-end deepfake video detection system that classifies uploaded videos as **Real** or **Fake**. The pipeline covers face extraction, model training with systematic ablation studies, video-level inference, and a Django web app with explainability and reporting.

## Results

| Metric | Clean video | Compressed video | Cross-dataset (Celeb-DF-v2) |
|---|---|---|---|
| ROC-AUC | **99.71%** | 99.54% | 79.9% |
| Balanced Accuracy | **97.36%** | **96.93%** | **75.32%** |

Trained on **FaceForensics++**, evaluated on both clean and compressed test splits to reflect real-world video quality, and cross-tested on **Celeb-DF-v2** to check generalization beyond the training distribution - where performance drops significantly, highlighting that FaceForensics++-specific artifacts don't fully transfer to unseen manipulation methods. Note the Celeb-DF-v2 evaluation reuses the decision threshold tuned on FaceForensics++ test data rather than recalibrating on Celeb-DF-v2 itself, so AUC (threshold-independent) is the more reliable number to compare there - balanced accuracy at that threshold is less meaningful out-of-distribution.

## Features

- Deepfake video classification using a fine-tuned **EfficientNet-B0**
- Face extraction via **MTCNN**
- Video-level prediction via configurable frame aggregation (mean / median / max / top-k)
- Django web application for video upload and inference
- Grad-CAM based explainability
- Confidence score visualization
- Prediction history
- PDF report generation
- REST API support
- Automatic upload cleanup


## Setup

```bash
git clone https://github.com/Yogeshwar2005/deepfake-video-detector.git
cd deepfake-video-detector
pip install -r requirements.txt
```

Requires a CUDA-capable GPU for reasonable training/inference speed (the model was trained and evaluated on an NVIDIA RTX 3050, 4GB VRAM). CPU inference is supported but slow.

### Dataset

Download [FaceForensics++](https://github.com/ondyari/FaceForensics) and place raw videos under `data/raw/<manipulation_type>/` (`original`, `Deepfakes`, `Face2Face`, `FaceSwap`, `FaceShifter`, `NeuralTextures`, `DeepFakeDetection`), matching the folder names read by `src/extract_faces.py`. For cross-dataset evaluation, download [Celeb-DF-v2](https://github.com/yuezunli/celeb-deepfakeforensics) separately.

Extract face crops (splits into train/val/test by video identifier automatically):

```bash
cd src
python3 extract_faces.py
```

## Training

Train a single configuration:

```bash
cd src
python3 training.py --epochs 10 --augment --sampler --loss bce --pos-weight 0.157
```

Key flags (see `training.py --help` for the full list):

| Flag | Purpose |
|---|---|
| `--augment` | Enable compression-aware augmentation (Albumentations) |
| `--sampler` | Enable `WeightedRandomSampler` for class imbalance |
| `--loss {bce,focal,topk}` | Loss function |
| `--pos-weight` | Positive class weight for BCE (real:fake imbalance is ~1:6) |
| `--seed` | Random seed |

Reproduce the full ablation sweep or the multi-seed stability check:

```bash
cd scripts
./training_all.sh   # sweeps augment / sampler / pos_weight combinations
./training_one.sh   # trains the best configuration across 5 seeds
```

## Evaluation

Frame-level evaluation (clean and compressed, at the checkpoint's threshold and at 0.5):

```bash
cd scripts
./evaluate_one.sh ../models/efficientnet_b0/results/<checkpoint>.pth
```

Video-level evaluation with a chosen aggregation strategy:

```bash
cd src
python3 evaluate_videos.py --checkpoint <path_to_checkpoint> --aggregation topk --num-frames 16 --test
```

Sweep aggregation method × frame count:

```bash
cd scripts
./evaluate_aggregation.sh
```

Cross-dataset evaluation on Celeb-DF-v2:

```bash
cd src
python3 evaluate_celeb-df-v2.py --checkpoint <path_to_checkpoint>
```

## Model

**EfficientNet-B0**, pretrained on ImageNet, fine-tuned for binary deepfake classification (final layer replaced with a single-logit linear head, trained with `BCEWithLogitsLoss`). Face crops are resized to `224×224` and normalized with ImageNet statistics.

**Why EfficientNet-B0:** EfficientNet-B1 was evaluated and didn't improve accuracy enough to justify its higher training/inference cost. ConvNeXt-Tiny was also tried but required a smaller batch size and much longer training time; the run was cancelled before completion since the time cost wasn't justified given B1 had already failed to beat B0 - B0 gave the best accuracy-to-efficiency tradeoff for this setup.

## Ablation Studies

20+ models were trained to systematically evaluate:

- **Loss function** - BCE vs. focal loss vs. OHEM, validated across 5 seeds to separate real improvement from seed noise (differences were within seed variance; BCE was kept for simplicity)
- **Backbone** - EfficientNet-B0 vs. B1 vs. ConvNeXt-Tiny
- **Augmentation** - compression-aware augmentation (Albumentations) closed most of the clean-to-compressed AUC gap
- **Class imbalance handling** - weighted BCE (`pos_weight`) vs. `WeightedRandomSampler` for the ~1:6 real:fake ratio
- **Video-level aggregation** - mean / median / max / top-k, evaluated at 8/16/32 sampled frames per video; **top-k (k=5) over 16 frames** gave the best stability/speed tradeoff
- **Ensembling** - weighted and logic-gate ensembling across the top two checkpoints (`ensemble.py`); the optimal weighted split found via search was 93% / 7% in favor of the single best model, confirming the second model contributed almost nothing due to high correlation between the two — ensembling was dropped from the final system

## Web Application

```bash
cd webapp
python3 manage.py migrate
python3 manage.py runserver
```

Upload a video and get:
- Real/Fake prediction with confidence score
- Frames analyzed / faces detected
- Grad-CAM heatmaps on the most suspicious frames
- Downloadable PDF report
- Prediction history

The model and MTCNN detector are loaded once at Django startup (module import time in `detector/inference.py`) rather than per-request, avoiding the cost of reloading the model on every upload.

### REST API

```
POST /api/predict/
Content-Type: multipart/form-data
Field: video   (.mp4, .mov, .avi, .mkv)
```

Example response:

```json
{
  "filename": "sample_video.mp4",
  "prediction": "Fake",
  "fake_probability": "XX.XX",
  "frames_analyzed": 16,
  "faces_detected": 16,
  "threshold": "XX.XX",
  "gradcam_results": []
}
```

## Limitations

- Cross-dataset generalization is weak (79.9% AUC on Celeb-DF-v2 vs. 99%+ in-distribution) - the model has learned FaceForensics++-specific manipulation artifacts rather than fully general deepfake cues.
- Not real-time - face detection + CNN inference per frame is the bottleneck for live/streaming use cases.
- No audio-visual or temporal modeling; predictions are purely per-frame, spatial-only.

## Future Work

- Improve cross-dataset generalization (broader training data, domain generalization techniques)
- Transformer-based / temporal video models
- Audio-visual consistency checks
- Real-time inference optimization
- Cloud deployment with authentication and scalable inference
