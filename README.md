# Deepfake Video Detection using Deep Learning

An end-to-end deepfake video detection system that classifies uploaded videos as **Real** or **Fake** using a trained deep learning model.
## Features

* Deepfake video classification using EfficientNet-B0
* Frame extraction and face detection using MTCNN
* Video-level prediction using frame aggregation
* Django web application for video upload and inference
* Confidence score visualization 
* Prediction history 
* Grad-CAM based explainability
* PDF report generation
* REST API support
* Automatic file handling and cleanup

## Model

The detector uses **EfficientNet-B0** pretrained on ImageNet and fine-tuned for binary deepfake classification. Face crops are resized to `224 x 224`, normalized using ImageNet statistics, and passed through the model to produce frame-level predictions.

## Dataset

* **Primary Dataset:** FaceForensics++
* **Cross-Dataset Evaluation:** Celeb-DF-v2

## Ablation Studies

The project evaluated multiple design choices:

* Augmentation vs no augmentation
* BCE loss vs focal loss vs OHEM
* Weighted sampling and class weighting
* EfficientNet-B0 vs EfficientNet-B1 vs ConvNeXt-Tiny
* Mean, median, max, and top-k video aggregation
* 8, 16, and 32 sampled frames per video

The final selected configuration used **EfficientNet-B0**, **16 sampled frames**, and **top-k aggregation with k=5**.

## Web Application

The Django web app allows users to upload a video and receive:

* Real/Fake prediction
* Fake probability
* Frames analyzed
* Faces detected
* Confidence chart
* Grad-CAM visualizations
* Downloadable PDF report

## REST API

Endpoint:

```text
POST /api/predict/
```

Request:

```text
multipart/form-data
key: video
```

Supported formats:

```text
.mp4, .mov, .avi, .mkv
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

## Future Improvements

* Improve cross-dataset generalization
* Add transformer-based video models
* Add audio-visual deepfake detection
* Improve real-time inference speed
* Deploy the application on cloud infrastructure
* Add user authentication and role-based access


