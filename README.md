# Multi-Modal Deepfake Detection

A robust deepfake detection system combining CNN-RNN visual analysis with physiological signal verification through remote photoplethysmography (rPPG).

## Overview

This project implements a multi-modal approach to deepfake detection by integrating two complementary techniques:

1. **CNN-RNN Hybrid Model**: Analyzes spatial and temporal inconsistencies in video frames using ResNeXt-50 and LSTM layers
2. **rPPG Physiological Analysis**: Detects inconsistencies in pulse signals extracted from facial videos

The system achieves improved accuracy by combining these approaches through an intelligent fusion mechanism that leverages the strengths of both visual and physiological analysis.

## Features

- **Spatial-Temporal Analysis**: Identifies visual artifacts and inconsistencies across video frames
- **Physiological Signal Verification**: Analyzes heart rate patterns that are difficult for deepfakes to replicate
- **Adaptive Fusion System**: Intelligently combines the results from both models based on confidence levels
- **Comprehensive Visualization**: Provides detailed visualizations of the detection process
- **Support for Video Files and Webcam Input**: Can analyze both pre-recorded videos and live webcam feeds


## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/KrrishK04/Multi-Modal-DeepFake-Detection-using-CNN-RNN-and-Rppg.git
   cd Multi-Modal-DeepFake-Detection-using-CNN-RNN-and-Rppg/rPPg   
   ```

2. Create and activate a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows, use: venv\Scripts\activate
   ```

3. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Download the pre-trained CNN-RNN model from the https://drive.google.com/drive/folders/12TG1ka7p2tIe2BYn2UfRGhgI8Gj97gQn?usp=sharing and place it in the rppg folder.

## Usage

### Process a video file:

```bash
python dfd.py --video_path path/to/video.mp4 --model_path checkpoint.pt
```
or you can simply run dfd.py and manage the video files in the code. Open dfd.py and scroll to the end to change video paths

### Use webcam input:

```bash
python dfd.py --model_path checkpoint.pt
```

### Advanced options:

```bash
python dfd.py --video_path path/to/video.mp4 --model_path checkpoint.pt --no_rppg --no_visualization
```
The Ouput is displayed in the terminal. The code also saves an image file - 'fusion_results.png' as an output in the rppg folder.
## Project Structure

- `dfd.py` - Main script for deepfake detection
- `capture_frames.py` - Handles video frame capture and processing
- `pulse.py` - Extracts blood volume pulse signals from video frames
- `process_mask.py` - Processes facial regions for physiological signal extraction
- `utils.py` - Utility functions for signal processing
- `plot_cont.py` - Visualization utilities for heart rate signals
- `models.py` - CNN-RNN model architecture definition

## Results

The multi-modal approach achieves improved accuracy compared to single-modality approaches:
- CNN-RNN Model: ~98% accuracy on the validation set
- rPPG Analysis: Effective at identifying physiological inconsistencies in deepfake videos
- Combined Approach: Enhanced robustness against diverse deepfake generation techniques
