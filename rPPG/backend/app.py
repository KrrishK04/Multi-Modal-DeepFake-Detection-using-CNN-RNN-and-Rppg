import os
import sys
import cv2
import torch
import numpy as np
from flask import Flask, request, jsonify
from werkzeug.utils import secure_filename
import face_recognition
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
import multiprocessing as mp
import matplotlib.pyplot as plt

# Add the parent directory to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from pulse import Pulse
from utils import moving_avg, scale_pulse
from FaceSeg import FaceSegGPU

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Model definition from dfd.py
class Model(nn.Module):
    def __init__(self, num_classes=2, latent_dim=2048, lstm_layers=1, hidden_dim=2048, bidirectional=False):
        super(Model, self).__init__()
        model = torch.hub.load('pytorch/vision:v0.10.0', 'resnext50_32x4d', pretrained=True)
        self.model = nn.Sequential(*list(model.children())[:-2])
        self.lstm = nn.LSTM(latent_dim, hidden_dim, lstm_layers, bidirectional)
        self.relu = nn.LeakyReLU()
        self.dp = nn.Dropout(0.4)
        self.linear1 = nn.Linear(2048, num_classes)
        self.avgpool = nn.AdaptiveAvgPool2d(1)
    def forward(self, x):
        batch_size, seq_length, c, h, w = x.shape
        x = x.view(batch_size * seq_length, c, h, w)
        fmap = self.model(x)
        x = self.avgpool(fmap)
        x = x.view(batch_size, seq_length, 2048)
        x_lstm, _ = self.lstm(x, None)
        return fmap, self.dp(self.linear1(torch.mean(x_lstm, dim=1)))

class VideoDataset:
    def __init__(self, sequence_length=20, transform=None):
        self.transform = transform
        self.count = sequence_length
        self.original_frames = []  
    def process_video(self, video_path):
        frames = []
        self.original_frames = []  
        vidObj = cv2.VideoCapture(video_path)
        success = 1
        while success and len(frames) < self.count:
            success, image = vidObj.read()
            if success:
                self.original_frames.append(image.copy())
                try:
                    face_locations = face_recognition.face_locations(image)
                    if face_locations:
                        top, right, bottom, left = face_locations[0]
                        face = image[top:bottom, left:right, :]
                        if self.transform:
                            face = self.transform(face)
                        frames.append(face)
                        cv2.rectangle(self.original_frames[-1], (left, top), (right, bottom), (0, 255, 0), 2)
                except Exception as e:
                    print(f"Error in face detection: {e}")
                    if self.transform:
                        frame = self.transform(image)
                    frames.append(frame)
        if len(frames) < self.count:
            last_frame = frames[-1] if frames else None
            if last_frame is not None:
                frames.extend([last_frame] * (self.count - len(frames)))
        vidObj.release()
        if frames:
            frames = torch.stack(frames)
            return frames.unsqueeze(0)  
        else:
            return None
    def get_original_frames(self):
        return self.original_frames

class EnhancedFusionSystem:
    def __init__(self, cnn_rnn_model_path, sequence_length=20, frame_rate=30, batch_size=30, signal_size=270):
        self.sequence_length = sequence_length
        self.frame_rate = frame_rate
        self.batch_size = batch_size
        self.signal_size = signal_size
        self.cnn_rnn_model_path = cnn_rnn_model_path

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = Model(2).to(self.device)

        try:
            checkpoint = torch.load(cnn_rnn_model_path, map_location=self.device)
            if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
                self.model.load_state_dict(checkpoint['model_state_dict'])
            else:
                self.model.load_state_dict(checkpoint)
            print("CNN-RNN model loaded successfully")
        except Exception as e:
            print(f"Error loading model: {e}")
            raise

        self.model.eval()

        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((112, 112)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

        self.rppg_results = {"heart_rates": [], "confidence": 0.0}
        self.deepfake_results = {"prediction": None, "confidence": 0.0}

        self.sm = nn.Softmax(dim=1)

        self.original_frames = []
        self.video_dataset = None

    def predict_deepfake(self, frames):
        with torch.no_grad():
            fmap, logits = self.model(frames.to(self.device))
            probabilities = self.sm(logits)
            _, prediction = torch.max(probabilities, 1)
            confidence = probabilities[0, prediction.item()].item() * 100

            result = {
                "prediction": "REAL" if prediction.item() == 1 else "FAKE",
                "confidence": confidence,
                "logits": logits.cpu().numpy(),
                "probabilities": probabilities.cpu().numpy()
            }
            return result

    def run_rppg(self, source, extended_analysis=True):
        face_seg = FaceSegGPU(bs=self.batch_size)
        pulse = Pulse(
            framerate=self.frame_rate,
            signal_size=self.signal_size,
            batch_size=self.batch_size,
            image_size=128
        )

        cap = cv2.VideoCapture(source)
        mean_rgb_series = []
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            face_coords = face_seg.get_face(frame)
            if face_coords:
                face_frame = frame[face_coords[1]:face_coords[3], face_coords[0]:face_coords[2]]
                if face_frame is not None and face_frame.ndim == 3:
                    mean_rgb = np.mean(face_frame, axis=(0, 1))
                    mean_rgb_series.append(mean_rgb)
        cap.release()

        hrs = []
        if len(mean_rgb_series) > 0:
            mean_rgb_array = np.stack(mean_rgb_series, axis=0)  # shape (N, 3)
            hr_signal = pulse.get_pulse(mean_rgb_array)
            if hr_signal is not None and hr_signal.size > 0:
                # Optionally, extract heart rate from hr_signal using get_rfft_hr
                hr_value = pulse.get_rfft_hr(hr_signal)
                hrs.append(hr_value)

        result = {
            "heart_rates": hrs,
            "average_hr": float(np.mean(hrs)) if len(hrs) > 0 else 0,
            "confidence": 0.0,
            "rhythm_features": {}
        }

        if len(hrs) >= 5:
            result["confidence"] = self._calculate_hr_confidence(hrs)
            if extended_analysis and len(hrs) >= 10:
                smoothed_hrs = moving_avg(hrs, 5)
                hr_diffs = np.diff(smoothed_hrs)
                rmssd = np.sqrt(np.mean(np.square(hr_diffs)))
                pnn50 = 100 * np.sum(np.abs(hr_diffs) > 0.05) / len(hr_diffs)
                low_freq_power = np.sum(np.abs(np.fft.rfft(hr_diffs))[1:5])
                high_freq_power = np.sum(np.abs(np.fft.rfft(hr_diffs))[5:12])
                lf_hf_ratio = low_freq_power / high_freq_power if high_freq_power > 0 else 0
                physio_score = self._calculate_physiological_credibility(hrs, rmssd, pnn50, lf_hf_ratio)
                result["rhythm_features"] = {
                    "rmssd": float(rmssd),
                    "pnn50": float(pnn50),
                    "lf_hf_ratio": float(lf_hf_ratio),
                    "physiological_credibility": float(physio_score)
                }
                result["confidence"] = 0.4 * result["confidence"] + 0.6 * physio_score

        if "rhythm_features" in result and "physiological_credibility" in result["rhythm_features"]:
            physio_score = result["rhythm_features"]["physiological_credibility"]
            result["prediction"] = "REAL" if physio_score > 60 else "FAKE"
        else:
            avg_hr = result["average_hr"]
            hr_stability = np.std(hrs) if len(hrs) >= 3 else 0
            result["prediction"] = "REAL" if (65 <= avg_hr <= 120 and 0.5 <= hr_stability <= 25) else "FAKE"

        self.rppg_results = result

    def _calculate_hr_confidence(self, hrs):
        if len(hrs) < 4:
            return 0.0
        smoothed_hrs = moving_avg(hrs, 5) if len(hrs) >= 5 else hrs
        std_dev = np.std(smoothed_hrs)
        max_expected_std = 20
        confidence = max(0, 100 - (std_dev * 100 / max_expected_std))
        return confidence

    def _calculate_physiological_credibility(self, hrs, rmssd, pnn50, lf_hf_ratio):
        credibility_score = 0
        avg_hr = np.mean(hrs)
        if 65 <= avg_hr <= 120:
            credibility_score += 20
        if 5 <= rmssd <= 50:
            credibility_score += 20
        elif 0.25 <= rmssd <= 80:
            credibility_score += 10
        if pnn50 >= 5:
            credibility_score += 20
        elif pnn50 >= 2:
            credibility_score += 10
        if 0.5 <= lf_hf_ratio <= 2.0:
            credibility_score += 20
        elif 0 <= lf_hf_ratio <= 3.0:
            credibility_score += 10
        hr_stability = np.std(hrs)
        if 1.5 <= hr_stability <= 15:
            credibility_score += 20
        elif (0.5 <= hr_stability < 1.5) or (15 < hr_stability <= 25):
            credibility_score += 10
        return credibility_score

    def process_video(self, video_path, run_rppg=True):
        print(f"Processing video: {video_path}")
        print("Running CNN-RNN deepfake detection...")
        self.video_dataset = VideoDataset(sequence_length=self.sequence_length, transform=self.transform)
        frames = self.video_dataset.process_video(video_path)
        self.original_frames = self.video_dataset.get_original_frames()

        if frames is not None:
            self.deepfake_results = self.predict_deepfake(frames)
            print(f"Deepfake detection result: {self.deepfake_results['prediction']} with confidence {self.deepfake_results['confidence']:.2f}%")
        else:
            print("Could not extract frames for CNN-RNN model")
            self.deepfake_results = {"prediction": "FAKE", "confidence": 0.0}

        if run_rppg:
            print("Running rPPG heart rate analysis...")
            try:
                self.run_rppg(video_path, extended_analysis=True)
                if isinstance(self.rppg_results.get("heart_rates", None), list) and self.rppg_results["heart_rates"]:
                    hrs = self.rppg_results["heart_rates"]
                    avg_hr = np.mean(hrs[-10:]) if len(hrs) > 10 else np.mean(hrs)
                    print(f"Average heart rate: {avg_hr:.1f} BPM with confidence {self.rppg_results['confidence']:.2f}%")
                    print(f"rPPG prediction: {self.rppg_results['prediction']}")
                    if "rhythm_features" in self.rppg_results and "physiological_credibility" in self.rppg_results["rhythm_features"]:
                        physio_score = self.rppg_results["rhythm_features"]["physiological_credibility"]
                        print(f"Physiological credibility score: {physio_score:.1f}/100")
                else:
                    print("No heart rate data extracted")
            except Exception as e:
                print(f"Error in rPPG processing: {e}")
                self.rppg_results = {"heart_rates": [], "average_hr": 0, "confidence": 0.0, "prediction": "FAKE", "rhythm_features": {}}

        fusion_result = self._fuse_results()
        return fusion_result

    def _fuse_results(self):
        deepfake_pred = self.deepfake_results.get("prediction", "Unknown")
        deepfake_confidence = self.deepfake_results.get("confidence", 0)
        heart_rates = self.rppg_results.get("heart_rates", [])
        rppg_confidence = self.rppg_results.get("confidence", 0)
        rppg_pred = self.rppg_results.get("prediction", "Unknown")
        physio_score = 0
        if "rhythm_features" in self.rppg_results and "physiological_credibility" in self.rppg_results["rhythm_features"]:
            physio_score = self.rppg_results["rhythm_features"]["physiological_credibility"]
        hr_is_plausible = False
        avg_hr = 0
        hr_stability = 0

        if isinstance(heart_rates, list) and len(heart_rates) > 4:
            avg_hr = np.mean(heart_rates[-10:]) if len(heart_rates) > 10 else np.mean(heart_rates)
            hr_stability = np.std(heart_rates[-5:]) if len(heart_rates) > 5 else np.std(heart_rates)
            hr_is_plausible = (65 <= avg_hr <= 120) and (0.5 <= hr_stability <= 25)

        if not isinstance(heart_rates, list) or len(heart_rates) < 4 or rppg_confidence < 10:
            if heart_rates and rppg_confidence > 0:
                if avg_hr < 65 or avg_hr > 120 or hr_stability < 0.1:
                    fusion_result = {
                        "prediction": deepfake_pred,
                        "confidence": max(5, deepfake_confidence * 0.95),
                        "method": "CNN-RNN Primary (low rPPG confidence)",
                        "cnn_rnn_prediction": deepfake_pred,
                        "cnn_rnn_confidence": deepfake_confidence,
                        "rppg_prediction": rppg_pred,
                        "rppg_confidence": rppg_confidence
                    }
                else:
                    fusion_result = {
                        "prediction": deepfake_pred,
                        "confidence": min(100, deepfake_confidence * 1.02),
                        "method": "CNN-RNN Primary (low rPPG confidence)",
                        "cnn_rnn_prediction": deepfake_pred,
                        "cnn_rnn_confidence": deepfake_confidence,
                        "rppg_prediction": rppg_pred,
                        "rppg_confidence": rppg_confidence
                    }
            else:
                fusion_result = {
                    "prediction": deepfake_pred,
                    "confidence": deepfake_confidence,
                    "method": "CNN-RNN only (no/unreliable rPPG data)",
                    "cnn_rnn_prediction": deepfake_pred,
                    "cnn_rnn_confidence": deepfake_confidence,
                    "rppg_prediction": "Unknown" if rppg_confidence == 0 else rppg_pred,
                    "rppg_confidence": rppg_confidence
                }
        else:
            total_confidence = deepfake_confidence + rppg_confidence
            if total_confidence == 0:
                cnn_rnn_weight = 1.0
                rppg_weight = 0.0
            else:
                cnn_rnn_weight = 0.6 * (deepfake_confidence / total_confidence)
                rppg_weight = 0.4 * (rppg_confidence / total_confidence)
                weight_sum = cnn_rnn_weight + rppg_weight
                cnn_rnn_weight = cnn_rnn_weight / weight_sum
                rppg_weight = rppg_weight / weight_sum

            rppg_believability = physio_score if (physio_score > 0 and rppg_confidence > 20) else rppg_confidence

            if (deepfake_pred == "FAKE" and rppg_pred == "FAKE") or (deepfake_pred == "REAL" and rppg_pred == "REAL"):
                agreement_bonus = 1.3
                final_confidence = min(100, (deepfake_confidence * cnn_rnn_weight + rppg_believability * rppg_weight) * agreement_bonus)
                fusion_result = {
                    "prediction": deepfake_pred,
                    "confidence": final_confidence,
                    "method": f"Strong Fusion (CNN-RNN and rPPG agree on {deepfake_pred})",
                    "cnn_rnn_prediction": deepfake_pred,
                    "cnn_rnn_confidence": deepfake_confidence,
                    "rppg_prediction": rppg_pred,
                    "rppg_confidence": rppg_confidence
                }
            else:
                cnn_rnn_score = deepfake_confidence
                rppg_score = rppg_believability
                weighted_cnn_rnn = cnn_rnn_score * cnn_rnn_weight
                weighted_rppg = rppg_score * rppg_weight

                if weighted_rppg > weighted_cnn_rnn * 1.5 and rppg_confidence > 30:
                    final_prediction = rppg_pred
                    final_confidence = weighted_rppg
                    method = f"rPPG Override (strong physiological evidence contradicts CNN-RNN)"
                else:
                    if weighted_cnn_rnn >= weighted_rppg or rppg_confidence < 30:
                        final_prediction = deepfake_pred
                        final_confidence = weighted_cnn_rnn * 0.9
                        method = f"CNN-RNN Primary (rPPG disagrees)"
                    else:
                        final_prediction = rppg_pred
                        final_confidence = weighted_rppg * 0.9
                        method = f"rPPG Primary (contradicts CNN-RNN)"

                fusion_result = {
                    "prediction": final_prediction,
                    "confidence": min(100, final_confidence),
                    "method": method,
                    "cnn_rnn_prediction": deepfake_pred,
                    "cnn_rnn_confidence": deepfake_confidence,
                    "rppg_prediction": rppg_pred,
                    "rppg_confidence": rppg_confidence
                }

            fusion_result["heart_rate"] = avg_hr
            fusion_result["hr_plausible"] = hr_is_plausible
            fusion_result["physio_score"] = physio_score
            return fusion_result

fusion_system = EnhancedFusionSystem(
    cnn_rnn_model_path="checkpoint.pt",
    sequence_length=20,
    frame_rate=30,
    batch_size=30,
    signal_size=270
)

@app.route('/analyze', methods=['POST'])
def analyze_video():
    if 'video' not in request.files:
        return jsonify({'error': 'No video file provided'}), 400
    file = request.files['video']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    filename = secure_filename(file.filename)
    video_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(video_path)
    result = fusion_system.process_video(video_path=video_path, run_rppg=True)
    return jsonify(result)

if __name__ == '__main__':
    app.run(debug=True)

    def predict_deepfake(self, frames):
        with torch.no_grad():
            fmap, logits = self.model(frames.to(self.device))
            probabilities = self.sm(logits)
            _, prediction = torch.max(probabilities, 1)
            confidence = probabilities[0, prediction.item()].item() * 100

            result = {
                "prediction": "REAL" if prediction.item() == 1 else "FAKE",
                "confidence": confidence,
                "logits": logits.cpu().numpy(),
                "probabilities": probabilities.cpu().numpy()
            }

            return result

    def run_rppg(self, source, extended_analysis=True):
        face_seg = FaceSegGPU()
        pulse = Pulse(
            fps=self.frame_rate,
            signal_size=self.signal_size,
            batch_size=self.batch_size,
            image_size=128
        )

        cap = cv2.VideoCapture(source)
        hrs = []
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            face_coords = face_seg.get_face(frame)
            if face_coords:
                face_frame = frame[face_coords[1]:face_coords[3], face_coords[0]:face_coords[2]]
                hr = pulse.get_pulse(face_frame)
                if hr:
                    hrs.append(hr)
        cap.release()

        result = {
            "heart_rates": hrs,
            "average_hr": float(np.mean(hrs)) if len(hrs) > 0 else 0,
            "confidence": 0.0,
            "rhythm_features": {}
        }

        if len(hrs) >= 5:
            result["confidence"] = self._calculate_hr_confidence(hrs)
            if extended_analysis and len(hrs) >= 10:
                smoothed_hrs = moving_avg(hrs, 5)
                hr_diffs = np.diff(smoothed_hrs)
                rmssd = np.sqrt(np.mean(np.square(hr_diffs)))
                pnn50 = 100 * np.sum(np.abs(hr_diffs) > 0.05) / len(hr_diffs)
                low_freq_power = np.sum(np.abs(np.fft.rfft(hr_diffs))[1:5])
                high_freq_power = np.sum(np.abs(np.fft.rfft(hr_diffs))[5:12])
                lf_hf_ratio = low_freq_power / high_freq_power if high_freq_power > 0 else 0
                physio_score = self._calculate_physiological_credibility(hrs, rmssd, pnn50, lf_hf_ratio)
                result["rhythm_features"] = {
                    "rmssd": float(rmssd),
                    "pnn50": float(pnn50),
                    "lf_hf_ratio": float(lf_hf_ratio),
                    "physiological_credibility": float(physio_score)
                }
                result["confidence"] = 0.4 * result["confidence"] + 0.6 * physio_score

        if "rhythm_features" in result and "physiological_credibility" in result["rhythm_features"]:
            physio_score = result["rhythm_features"]["physiological_credibility"]
            result["prediction"] = "REAL" if physio_score > 60 else "FAKE"
        else:
            avg_hr = result["average_hr"]
            hr_stability = np.std(hrs) if len(hrs) >= 3 else 0
            result["prediction"] = "REAL" if (65 <= avg_hr <= 120 and 0.5 <= hr_stability <= 25) else "FAKE"

        self.rppg_results = result

    def _calculate_hr_confidence(self, hrs):
        if len(hrs) < 4:
            return 0.0
        smoothed_hrs = moving_avg(hrs, 5) if len(hrs) >= 5 else hrs
        std_dev = np.std(smoothed_hrs)
        max_expected_std = 20
        confidence = max(0, 100 - (std_dev * 100 / max_expected_std))
        return confidence

    def _calculate_physiological_credibility(self, hrs, rmssd, pnn50, lf_hf_ratio):
        credibility_score = 0
        avg_hr = np.mean(hrs)
        if 65 <= avg_hr <= 120:
            credibility_score += 20
        if 5 <= rmssd <= 50:
            credibility_score += 20
        elif 0.25 <= rmssd <= 80:
            credibility_score += 10
        if pnn50 >= 5:
            credibility_score += 20
        elif pnn50 >= 2:
            credibility_score += 10
        if 0.5 <= lf_hf_ratio <= 2.0:
            credibility_score += 20
        elif 0 <= lf_hf_ratio <= 3.0:
            credibility_score += 10
        hr_stability = np.std(hrs)
        if 1.5 <= hr_stability <= 15:
            credibility_score += 20
        elif (0.5 <= hr_stability < 1.5) or (15 < hr_stability <= 25):
            credibility_score += 10
        return credibility_score

    def process_video(self, video_path, run_rppg=True):
        print(f"Processing video: {video_path}")
        print("Running CNN-RNN deepfake detection...")
        self.video_dataset = VideoDataset(sequence_length=self.sequence_length, transform=self.transform)
        frames = self.video_dataset.process_video(video_path)
        self.original_frames = self.video_dataset.get_original_frames()

        if frames is not None:
            self.deepfake_results = self.predict_deepfake(frames)
            print(f"Deepfake detection result: {self.deepfake_results['prediction']} with confidence {self.deepfake_results['confidence']:.2f}%")
        else:
            print("Could not extract frames for CNN-RNN model")
            self.deepfake_results = {"prediction": "Unknown", "confidence": 0.0}

        if run_rppg:
            print("Running rPPG heart rate analysis...")
            try:
                self.run_rppg(video_path, extended_analysis=True)
                if isinstance(self.rppg_results.get("heart_rates", None), list) and self.rppg_results["heart_rates"]:
                    hrs = self.rppg_results["heart_rates"]
                    avg_hr = np.mean(hrs[-10:]) if len(hrs) > 10 else np.mean(hrs)
                    print(f"Average heart rate: {avg_hr:.1f} BPM with confidence {self.rppg_results['confidence']:.2f}%")
                    print(f"rPPG prediction: {self.rppg_results['prediction']}")
                    if "rhythm_features" in self.rppg_results and "physiological_credibility" in self.rppg_results["rhythm_features"]:
                        physio_score = self.rppg_results["rhythm_features"]["physiological_credibility"]
                        print(f"Physiological credibility score: {physio_score:.1f}/100")
                else:
                    print("No heart rate data extracted")
            except Exception as e:
                print(f"Error in rPPG processing: {e}")
                self.rppg_results = {"heart_rates": [], "average_hr": 0, "confidence": 0.0, "prediction": "Unknown", "rhythm_features": {}}

        fusion_result = self._fuse_results()
        return fusion_result

    def _fuse_results(self):
        deepfake_pred = self.deepfake_results.get("prediction", "Unknown")
        deepfake_confidence = self.deepfake_results.get("confidence", 0)
        heart_rates = self.rppg_results.get("heart_rates", [])
        rppg_confidence = self.rppg_results.get("confidence", 0)
        rppg_pred = self.rppg_results.get("prediction", "Unknown")
        physio_score = 0
        if "rhythm_features" in self.rppg_results and "physiological_credibility" in self.rppg_results["rhythm_features"]:
            physio_score = self.rppg_results["rhythm_features"]["physiological_credibility"]
        hr_is_plausible = False
        avg_hr = 0
        hr_stability = 0

        if isinstance(heart_rates, list) and len(heart_rates) > 4:
            avg_hr = np.mean(heart_rates[-10:]) if len(heart_rates) > 10 else np.mean(heart_rates)
            hr_stability = np.std(heart_rates[-5:]) if len(heart_rates) > 5 else np.std(heart_rates)
            hr_is_plausible = (65 <= avg_hr <= 120) and (0.5 <= hr_stability <= 25)

        if not isinstance(heart_rates, list) or len(heart_rates) < 4 or rppg_confidence < 10:
            if heart_rates and rppg_confidence > 0:
                if avg_hr < 65 or avg_hr > 120 or hr_stability < 0.1:
                    fusion_result = {
                        "prediction": deepfake_pred,
                        "confidence": max(5, deepfake_confidence * 0.95),
                        "method": "CNN-RNN Primary (low rPPG confidence)",
                        "cnn_rnn_prediction": deepfake_pred,
                        "cnn_rnn_confidence": deepfake_confidence,
                        "rppg_prediction": rppg_pred,
                        "rppg_confidence": rppg_confidence
                    }
                else:
                    fusion_result = {
                        "prediction": deepfake_pred,
                        "confidence": min(100, deepfake_confidence * 1.02),
                        "method": "CNN-RNN Primary (low rPPG confidence)",
                        "cnn_rnn_prediction": deepfake_pred,
                        "cnn_rnn_confidence": deepfake_confidence,
                        "rppg_prediction": rppg_pred,
                        "rppg_confidence": rppg_confidence
                    }
            else:
                fusion_result = {
                    "prediction": deepfake_pred,
                    "confidence": deepfake_confidence,
                    "method": "CNN-RNN only (no/unreliable rPPG data)",
                    "cnn_rnn_prediction": deepfake_pred,
                    "cnn_rnn_confidence": deepfake_confidence,
                    "rppg_prediction": "Unknown" if rppg_confidence == 0 else rppg_pred,
                    "rppg_confidence": rppg_confidence
                }
        else:
            total_confidence = deepfake_confidence + rppg_confidence
            if total_confidence == 0:
                cnn_rnn_weight = 1.0
                rppg_weight = 0.0
            else:
                cnn_rnn_weight = 0.6 * (deepfake_confidence / total_confidence)
                rppg_weight = 0.4 * (rppg_confidence / total_confidence)
                weight_sum = cnn_rnn_weight + rppg_weight
                cnn_rnn_weight = cnn_rnn_weight / weight_sum
                rppg_weight = rppg_weight / weight_sum

            rppg_believability = physio_score if (physio_score > 0 and rppg_confidence > 20) else rppg_confidence

            if (deepfake_pred == "FAKE" and rppg_pred == "FAKE") or (deepfake_pred == "REAL" and rppg_pred == "REAL"):
                agreement_bonus = 1.3
                final_confidence = min(100, (deepfake_confidence * cnn_rnn_weight + rppg_believability * rppg_weight) * agreement_bonus)
                fusion_result = {
                    "prediction": deepfake_pred,
                    "confidence": final_confidence,
                    "method": f"Strong Fusion (CNN-RNN and rPPG agree on {deepfake_pred})",
                    "cnn_rnn_prediction": deepfake_pred,
                    "cnn_rnn_confidence": deepfake_confidence,
                    "rppg_prediction": rppg_pred,
                    "rppg_confidence": rppg_confidence
                }
            else:
                cnn_rnn_score = deepfake_confidence
                rppg_score = rppg_believability
                weighted_cnn_rnn = cnn_rnn_score * cnn_rnn_weight
                weighted_rppg = rppg_score * rppg_weight

                if weighted_rppg > weighted_cnn_rnn * 1.5 and rppg_confidence > 30:
                    final_prediction = rppg_pred
                    final_confidence = weighted_rppg
                    method = f"rPPG Override (strong physiological evidence contradicts CNN-RNN)"
                else:
                    if weighted_cnn_rnn >= weighted_rppg or rppg_confidence < 30:
                        final_prediction = deepfake_pred
                        final_confidence = weighted_cnn_rnn * 0.9
                        method = f"CNN-RNN Primary (rPPG disagrees)"
                    else:
                        final_prediction = rppg_pred
                        final_confidence = weighted_rppg * 0.9
                        method = f"rPPG Primary (contradicts CNN-RNN)"

                fusion_result = {
                    "prediction": final_prediction,
                    "confidence": min(100, final_confidence),
                    "method": method,
                    "cnn_rnn_prediction": deepfake_pred,
                    "cnn_rnn_confidence": deepfake_confidence,
                    "rppg_prediction": rppg_pred,
                    "rppg_confidence": rppg_confidence
                }

        fusion_result["heart_rate"] = avg_hr
        fusion_result["hr_plausible"] = hr_is_plausible
        fusion_result["physio_score"] = physio_score
        return fusion_result

fusion_system = EnhancedFusionSystem(
    cnn_rnn_model_path="..\\checkpoint.pt",
    sequence_length=20,
    frame_rate=30,
    batch_size=30,
    signal_size=270
)

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({"error": "No file part"}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400
    if file:
        filename = secure_filename(file.filename)
        video_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(video_path)

        result = fusion_system.process_video(video_path=video_path, run_rppg=True)

        return jsonify(result)

if __name__ == '__main__':
    app.run(debug=True)
