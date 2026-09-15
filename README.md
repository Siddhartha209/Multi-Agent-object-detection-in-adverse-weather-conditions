# Multi-Agent Object Detection — Weather-Adaptive Bounding Box Aggregation

This repository contains the detection aggregation pipeline from a multi-agent object detection system for autonomous driving in adverse weather. 
## Overview

Rather than relying on a single object detection model to handle all weather conditions, the wider project trained four separate YOLOv8 models, each specialised for a specific weather condition (clear, fog, rain streaks, and shadow/sunglare). This pipeline takes their combined output and turns it into one final, de-duplicated prediction per object — the part of the system this repo implements.

## How It Works

1. **Weather Classification** — A ResNet18-based classifier predicts the dominant weather condition present in the input image (clear, fog, rain, or shadow), which determines how much to trust each specialist model's predictions.

2. **Parallel Multi-Model Inference** — All four YOLO weather-specialist models run concurrently (via a thread pool) on the input image, each producing its own set of bounding box predictions.

3. **Confidence Weighting** — Predictions from the model matching the classified weather condition are boosted relative to the other three, reflecting the assumption that a model trained for the detected conditions should be trusted more.

4. **Detection Aggregation** — The weighted predictions from all four models are combined using two different methods, run side by side for comparison:
   - **Non-Maximum Suppression (NMS)** — keeps only the highest-confidence box among overlapping detections, discarding the rest.
   - **Weighted Box Fusion (WBF)** — combines overlapping boxes into a single fused prediction using a confidence-weighted average, retaining information from all contributing models rather than discarding it.

5. **Output** — Annotated images are saved showing the results of each individual weather model, along with the final aggregated NMS and WBF outputs for direct comparison.

## Why Two Aggregation Methods?

NMS is simple and fast but discards potentially useful information from lower-confidence overlapping boxes. WBF is more computationally involved but produces more accurate final bounding boxes by combining information across models rather than picking a single "winner." This pipeline runs both so their outputs can be directly compared.
