import torch
from torch.autograd import Variable
import time
import os
import sys
import numpy as np
import cv2
import matplotlib.pyplot as plt
import face_recognition
import pandas as pd
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
import multiprocessing as mp
from threading import Thread
from optparse import OptionParser

from pulse import Pulse
from capture_frames import CaptureFrames
from process_mask import ProcessMasks
from plot_cont import DynamicPlot
from utils import moving_avg, scale_pulse

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
        """Predict if the video is real or fake using CNN-RNN model"""
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
    
    def run_rppg(self, source, plot=True, extended_analysis=True):
        """Run the rPPG pipeline to extract heart rate with enhanced analysis"""
        # Setup pipes for communication between processes
        mask_process_pipe, child_process_pipe = mp.Pipe()
        plot_pipe = None
        plotter_pipe = None
        
        if plot:
            plot_pipe, plotter_pipe = mp.Pipe()
            plotter = DynamicPlot(self.signal_size, self.batch_size)
            plot_process = mp.Process(target=plotter, args=(plotter_pipe,), daemon=True)
            plot_process.start()
        
        # Start mask processing
        process_mask = ProcessMasks(self.signal_size, self.frame_rate, self.batch_size)
        mask_processer = mp.Process(target=process_mask, args=(child_process_pipe, plot_pipe, source), daemon=True)
        mask_processer.start()
        
        # Start frame capture
        capture = CaptureFrames(self.batch_size, source, show_mask=True)
        capture(mask_process_pipe, source)
        
        # Wait for processes to finish
        mask_processer.join()
        if plot:
            plot_process.join()
        
        # Get results
        try:
            hrs = np.load('hrs.npy')
            
            # Enhanced analysis of heart rate data
            result = {
                "heart_rates": hrs,
                "average_hr": float(np.mean(hrs)) if len(hrs) > 0 else 0,
                "confidence": 0.0,
                "rhythm_features": {}
            }
            
            if len(hrs) >= 5:
                # Calculate basic statistical features
                result["confidence"] = self._calculate_hr_confidence(hrs)
                
                # Extended analysis if requested
                if extended_analysis and len(hrs) >= 10:
                    # Calculate heart rate variability (HRV)
                    smoothed_hrs = moving_avg(hrs, 5)
                    hr_diffs = np.diff(smoothed_hrs)
                    
                    # RMSSD - Root Mean Square of Successive Differences
                    rmssd = np.sqrt(np.mean(np.square(hr_diffs)))
                    
                    # pNN50 - percentage of successive RR intervals that differ by more than 50 ms
                    pnn50 = 100 * np.sum(np.abs(hr_diffs) > 0.05) / len(hr_diffs)
                    
                    # Frequency domain analysis (simplified)
                    low_freq_power = np.sum(np.abs(np.fft.rfft(hr_diffs))[1:5])
                    high_freq_power = np.sum(np.abs(np.fft.rfft(hr_diffs))[5:12])
                    lf_hf_ratio = low_freq_power / high_freq_power if high_freq_power > 0 else 0
                    
                    # Physiological credibility score (0-100)
                    physio_score = self._calculate_physiological_credibility(hrs, rmssd, pnn50, lf_hf_ratio)
                    
                    # Store extended analysis results
                    result["rhythm_features"] = {
                        "rmssd": float(rmssd),
                        "pnn50": float(pnn50),
                        "lf_hf_ratio": float(lf_hf_ratio),
                        "physiological_credibility": float(physio_score)
                    }
                    
                    # Update confidence with physiological credibility
                    result["confidence"] = 0.4 * result["confidence"] + 0.6 * physio_score
            
            # Add rPPG prediction based on physiological credibility
            if "rhythm_features" in result and "physiological_credibility" in result["rhythm_features"]:
                physio_score = result["rhythm_features"]["physiological_credibility"]
                # High physiological score suggests REAL, low suggests FAKE
                result["prediction"] = "REAL" if physio_score > 60 else "FAKE"
            else:
                # Fallback if we don't have physiological credibility
                avg_hr = result["average_hr"]
                hr_stability = np.std(hrs) if len(hrs) >= 3 else 0
                
                # Basic checks for plausibility
                result["prediction"] = "REAL" if (65 <= avg_hr <= 120 and 0.5 <= hr_stability <= 25) else "FAKE"
                
            self.rppg_results = result
        except Exception as e:
            print(f"Error loading heart rate results: {e}")
            self.rppg_results = {
                "heart_rates": np.array([]),
                "average_hr": 0,
                "confidence": 0.0,
                "prediction": "Unknown",
                "rhythm_features": {}
            }
    
    def _calculate_hr_confidence(self, hrs):
        """Calculate confidence in heart rate measurement based on stability"""
        if len(hrs) < 4:
            return 0.0
        
        # Use moving average to smooth the heart rate values
        smoothed_hrs = moving_avg(hrs, 5) if len(hrs) >= 5 else hrs
        
        # Calculate the standard deviation as a measure of stability
        std_dev = np.std(smoothed_hrs)
        
        # Convert to confidence - lower std_dev means higher confidence
        # Normalize to range 0-100
        max_expected_std = 20  
        confidence = max(0, 100 - (std_dev * 100 / max_expected_std))
        
        return confidence
    
    def _calculate_physiological_credibility(self, hrs, rmssd, pnn50, lf_hf_ratio):
        """Calculate a physiological credibility score for the heart rate signal"""
        credibility_score = 0
        
        # 1. Check if average HR is in normal human range (65-120 BPM)
        avg_hr = np.mean(hrs)
        if 65 <= avg_hr <= 120:
            credibility_score += 20
        
        # 2. Check heart rate variability (HRV) measures
        if 5 <= rmssd <= 50:
            credibility_score += 20
        elif 0.25 <= rmssd <= 80:
            credibility_score += 10
        
        # Normal pNN50 is typically >10%
        if pnn50 >= 5:
            credibility_score += 20
        elif pnn50 >= 2:
            credibility_score += 10

        
        # Normal LF/HF ratio typically ranges from 0.5 to 2.0
        if 0.5 <= lf_hf_ratio <= 2.0:
            credibility_score += 20
        elif 0 <= lf_hf_ratio <= 3.0:
            credibility_score += 10
        
        # 3. Check signal stability - not too stable (synthetic) or too unstable (noisy)
        hr_stability = np.std(hrs)
        if 1.5 <= hr_stability <= 15:
            credibility_score += 20
        elif (0.5 <= hr_stability < 1.5) or (15 < hr_stability <= 25):
            credibility_score += 10
        
        return credibility_score
    
    def process_video(self, video_path, run_rppg=True, show_visualization=True):
        """Process a video file with both CNN-RNN and rPPG pipelines"""
        print(f"Processing video: {video_path}")
        
        # Process with CNN-RNN model
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
        
        # Process with rPPG if requested
        if run_rppg:
            print("Running rPPG heart rate analysis...")
            try:
                self.run_rppg(video_path, plot=show_visualization, extended_analysis=True)
                if isinstance(self.rppg_results.get("heart_rates", None), np.ndarray) and self.rppg_results["heart_rates"].size > 0:
                    hrs = self.rppg_results["heart_rates"]
                    avg_hr = np.mean(hrs[-10:]) if hrs.size > 10 else np.mean(hrs)
                    print(f"Average heart rate: {avg_hr:.1f} BPM with confidence {self.rppg_results['confidence']:.2f}%")
                    print(f"rPPG prediction: {self.rppg_results['prediction']}")
                    
                    # Print physiological credibility if available
                    if "rhythm_features" in self.rppg_results and "physiological_credibility" in self.rppg_results["rhythm_features"]:
                        physio_score = self.rppg_results["rhythm_features"]["physiological_credibility"]
                        print(f"Physiological credibility score: {physio_score:.1f}/100")
                else:
                    print("No heart rate data extracted")
            except Exception as e:
                print(f"Error in rPPG processing: {e}")
                self.rppg_results = {"heart_rates": np.array([]), "average_hr": 0, "confidence": 0.0, "prediction": "Unknown", "rhythm_features": {}}
        
        # Combine results
        fusion_result = self._fuse_results()
        
        # Visualization if requested
        if show_visualization:
            self._visualize_results(frames)
        
        return fusion_result
    
    def process_webcam(self, camera_id=0, duration=30, show_visualization=True):
        """Process webcam feed with both CNN-RNN and rPPG pipelines"""
        print(f"Processing webcam feed from camera {camera_id}")
        
        # For CNN-RNN, we need to capture a sequence of frames first
        print("Capturing frames for CNN-RNN model...")
        self.video_dataset = VideoDataset(sequence_length=self.sequence_length, transform=self.transform)
        
        cap = cv2.VideoCapture(camera_id)
        if not cap.isOpened():
            print("Error: Could not open camera.")
            return None
        
        collected_frames = []
        frame_count = 0
        
        while frame_count < self.sequence_length:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Store original frame
            self.original_frames.append(frame.copy())
            
            # Process frame (detect face)
            try:
                face_locations = face_recognition.face_locations(frame)
                if face_locations:
                    top, right, bottom, left = face_locations[0]
                    face = frame[top:bottom, left:right, :]
                    transformed_face = self.transform(face)
                    collected_frames.append(transformed_face)
                    frame_count += 1
                    
                    # Draw rectangle on original frame for visualization
                    cv2.rectangle(self.original_frames[-1], (left, top), (right, bottom), (0, 255, 0), 2)
                
                # Display the frame
                cv2.imshow('Collecting Frames', frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            except Exception as e:
                print(f"Error processing frame: {e}")
        
        cap.release()
        cv2.destroyAllWindows()
        
        # Process with CNN-RNN model
        if len(collected_frames) == self.sequence_length:
            frames = torch.stack(collected_frames).unsqueeze(0)  # Add batch dimension
            self.deepfake_results = self.predict_deepfake(frames)
            print(f"Deepfake detection result: {self.deepfake_results['prediction']} with confidence {self.deepfake_results['confidence']:.2f}%")
        else:
            print("Could not collect enough frames for CNN-RNN model")
            self.deepfake_results = {"prediction": "Unknown", "confidence": 0.0}
        
        # Process with rPPG
        print("Running rPPG heart rate analysis...")
        try:
            self.run_rppg(camera_id, plot=show_visualization, extended_analysis=True)
            if isinstance(self.rppg_results.get("heart_rates", None), np.ndarray) and self.rppg_results["heart_rates"].size > 0:
                hrs = self.rppg_results["heart_rates"]
                avg_hr = np.mean(hrs[-10:]) if hrs.size > 10 else np.mean(hrs)
                print(f"Average heart rate: {avg_hr:.1f} BPM with confidence {self.rppg_results['confidence']:.2f}%")
                print(f"rPPG prediction: {self.rppg_results['prediction']}")
                
                # Print physiological credibility if available
                if "rhythm_features" in self.rppg_results and "physiological_credibility" in self.rppg_results["rhythm_features"]:
                    physio_score = self.rppg_results["rhythm_features"]["physiological_credibility"]
                    print(f"Physiological credibility score: {physio_score:.1f}/100")
            else:
                print("No heart rate data extracted")
        except Exception as e:
            print(f"Error in rPPG processing: {e}")
            self.rppg_results = {"heart_rates": np.array([]), "average_hr": 0, "confidence": 0.0, "prediction": "Unknown", "rhythm_features": {}}
        
        # Combine results
        fusion_result = self._fuse_results()
        
        # Visualization if requested
        if show_visualization and len(collected_frames) == self.sequence_length:
            self._visualize_results(frames)
        
        return fusion_result
    
    def _fuse_results(self):
        """Combine results with stronger influence from rPPG data but respecting confidence levels"""
        deepfake_pred = self.deepfake_results.get("prediction", "Unknown")
        deepfake_confidence = self.deepfake_results.get("confidence", 0)
        
        # Extract rPPG data
        heart_rates = self.rppg_results.get("heart_rates", np.array([]))
        rppg_confidence = self.rppg_results.get("confidence", 0)
        rppg_pred = self.rppg_results.get("prediction", "Unknown")
        
        # Get physiological credibility score if available
        physio_score = 0
        if "rhythm_features" in self.rppg_results and "physiological_credibility" in self.rppg_results["rhythm_features"]:
            physio_score = self.rppg_results["rhythm_features"]["physiological_credibility"]
        
        # Calculate heart rate plausibility
        hr_is_plausible = False
        avg_hr = 0
        hr_stability = 0
        
        if isinstance(heart_rates, np.ndarray) and heart_rates.size > 4:
            avg_hr = np.mean(heart_rates[-10:]) if heart_rates.size > 10 else np.mean(heart_rates)
            hr_stability = np.std(heart_rates[-5:]) if heart_rates.size > 5 else np.std(heart_rates)
            
            # Basic physiological plausibility checks
            hr_is_plausible = (65 <= avg_hr <= 120) and (0.5 <= hr_stability <= 25)
        
        
        # If rPPG confidence is very low (< 10%) or we don't have sufficient data, rely almost entirely on CNN-RNN
        if isinstance(heart_rates, np.ndarray) and (heart_rates.size < 4 or rppg_confidence < 10):
            # Even with limited data, we can still use it for minor influence if confidence isn't zero
            if heart_rates.size > 0 and rppg_confidence > 0:
                # Only make minor adjustments to CNN-RNN prediction based on heart rate plausibility
                if avg_hr < 65 or avg_hr > 120 or hr_stability < 0.1:  # Implausible heart rate
                    fusion_result = {
                        "prediction": deepfake_pred,  # Keep CNN-RNN prediction
                        "confidence": max(5, deepfake_confidence * 0.95),  # Slightly reduce confidence
                        "method": "CNN-RNN Primary (low rPPG confidence)",
                        "cnn_rnn_prediction": deepfake_pred,
                        "cnn_rnn_confidence": deepfake_confidence,
                        "rppg_prediction": rppg_pred,
                        "rppg_confidence": rppg_confidence
                    }
                else:
                    # If heart rate seems plausible, very slightly boost confidence in CNN-RNN result
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
                # With zero rPPG confidence or no heart rate data, rely solely on CNN-RNN
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
            # Strong rPPG fusion only when rPPG has meaningful confidence
            
            # Weight factors based on relative confidence levels
            # Scale weights by confidence levels - higher confidence gets more weight
            total_confidence = deepfake_confidence + rppg_confidence
            if total_confidence == 0:  # Avoid division by zero
                cnn_rnn_weight = 1.0
                rppg_weight = 0.0
            else:
                # Base weights - adjusted by relative confidence
                cnn_rnn_weight = 0.6 * (deepfake_confidence / total_confidence)  
                rppg_weight = 0.4 * (rppg_confidence / total_confidence)
                
                # Normalize to ensure weights sum to 1
                weight_sum = cnn_rnn_weight + rppg_weight
                cnn_rnn_weight = cnn_rnn_weight / weight_sum
                rppg_weight = rppg_weight / weight_sum
            
            # Calculate physiological believability score (0-100)
            # Only use physio_score if there's reasonable confidence
            rppg_believability = physio_score if (physio_score > 0 and rppg_confidence > 20) else rppg_confidence
            
            # If CNN-RNN and rPPG agree, strengthen the confidence
            if (deepfake_pred == "FAKE" and rppg_pred == "FAKE") or (deepfake_pred == "REAL" and rppg_pred == "REAL"):
                # Agreement strengthens confidence
                agreement_bonus = 1.3  # Significant bonus for agreement
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
                    # rPPG strongly suggests the opposite with good confidence
                    final_prediction = rppg_pred
                    final_confidence = weighted_rppg
                    method = f"rPPG Override (strong physiological evidence contradicts CNN-RNN)"
                else:
                    # Default to CNN-RNN if there's disagreement and no strong confidence from rPPG
                    if weighted_cnn_rnn >= weighted_rppg or rppg_confidence < 30:
                        final_prediction = deepfake_pred
                        # Reduce confidence slightly due to disagreement
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
        
        # Add heart rate info to result
        fusion_result["heart_rate"] = avg_hr
        fusion_result["hr_plausible"] = hr_is_plausible
        fusion_result["physio_score"] = physio_score
        
        return fusion_result
    
    def _visualize_results(self, frames=None):
        """Create enhanced visualization of the results with clear display of predictions"""
        plt.figure(figsize=(15, 10))
        
        # Plot 1: Heart Rate over time
        plt.subplot(2, 2, 1)
        plt.title('Heart Rate Over Time')
        heart_rates = self.rppg_results.get("heart_rates", np.array([]))
        if isinstance(heart_rates, np.ndarray) and heart_rates.size > 0:
            hrs = heart_rates
            
            # Plot raw heart rates in light color
            plt.plot(hrs, 'lightblue', alpha=0.5, label='Raw HR')
            
            # Plot smoothed heart rates in darker color
            if heart_rates.size >= 5:
                smoothed = moving_avg(hrs, 5)
                plt.plot(range(2, len(smoothed)+2), smoothed, 'blue', linewidth=2, label='Smoothed HR')
            
            plt.ylim(65, 120)
            plt.xlabel('Frames')
            plt.ylabel('BPM')
            plt.legend()
            
            # Add plausibility region
            plt.axhspan(65, 120, alpha=0.2, color='green', label='Normal HR Range')
            
        else:
            plt.text(0.5, 0.5, 'No heart rate data available', ha='center', va='center')
        
        # Plot 2: Deepfake Detection Results
        plt.subplot(2, 2, 2)
        plt.title('CNN-RNN Deepfake Detection')
        labels = ['FAKE', 'REAL']
        if self.deepfake_results.get("probabilities") is not None:
            probs = self.deepfake_results["probabilities"][0]
            plt.bar(labels, probs, color=['red', 'green'])
            plt.ylim(0, 1)
            for i, v in enumerate(probs):
                plt.text(i, v + 0.05, f'{v:.2f}', ha='center')
            
            # Add text showing the prediction and confidence
            prediction = self.deepfake_results["prediction"]
            confidence = self.deepfake_results["confidence"]
            plt.text(0.5, -0.15, f'Prediction: {prediction}\nConfidence: {confidence:.2f}%', 
                    transform=plt.gca().transAxes, ha='center', 
                    bbox=dict(facecolor='yellow', alpha=0.2))
        else:
            plt.text(0.5, 0.5, 'No deepfake detection results available', ha='center', va='center')
        
        # Plot 3: Display a frame used by CNN-RNN with face detection
        plt.subplot(2, 2, 3)
        plt.title('CNN-RNN Input Frame')
        
        if self.original_frames and len(self.original_frames) > 0:
            # Get a middle frame to display
            frame_idx = min(len(self.original_frames) // 2, len(self.original_frames) - 1)
            frame = self.original_frames[frame_idx]
            
            # Convert BGR to RGB for display
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Display frame
            plt.imshow(frame_rgb)
            plt.axis('off')
        else:
            # If no original frames, show the sample frame if available
            if frames is not None:
                sample_frame = frames[0, 0].cpu().numpy().transpose(1, 2, 0)
                # Denormalize
                sample_frame = (sample_frame - sample_frame.min()) / (sample_frame.max() - sample_frame.min())
                plt.imshow(sample_frame)
                plt.axis('off')
            else:
                plt.text(0.5, 0.5, 'No frames available', ha='center', va='center')
        
        # Plot 4: Overall Fusion Result
        plt.subplot(2, 2, 4)
        plt.title('FUSION RESULT')
        plt.axis('off')
        fusion_result = self._fuse_results()
        
        # Format the prediction with color
        prediction_color = 'green' if fusion_result['prediction'] == 'REAL' else 'red'
        
        # Add a colored box for the overall prediction
        plt.text(0.5, 0.9, f"FINAL PREDICTION: {fusion_result['prediction']}", 
                fontsize=16, fontweight='bold', 
                color='white', ha='center',
                bbox=dict(facecolor=prediction_color, alpha=0.8))
        
        # Display the model predictions side by side
        cnn_color = 'green' if fusion_result['cnn_rnn_prediction'] == 'REAL' else 'red'
        rppg_color = 'green' if fusion_result['rppg_prediction'] == 'REAL' else 'red'
        
        # CNN-RNN prediction
        plt.text(0.25, 0.75, "CNN-RNN", fontsize=14, fontweight='bold', ha='center')
        plt.text(0.25, 0.7, f"Prediction: {fusion_result['cnn_rnn_prediction']}", 
                fontsize=12, color=cnn_color, ha='center', fontweight='bold')
        plt.text(0.25, 0.65, f"Confidence: {fusion_result['cnn_rnn_confidence']:.2f}%", 
                fontsize=12, ha='center')
        
        # rPPG prediction
        plt.text(0.75, 0.75, "rPPG", fontsize=14, fontweight='bold', ha='center')
        plt.text(0.75, 0.7, f"Prediction: {fusion_result['rppg_prediction']}", 
                fontsize=12, color=rppg_color, ha='center', fontweight='bold')
        plt.text(0.75, 0.65, f"Confidence: {fusion_result['rppg_confidence']:.2f}%", 
                fontsize=12, ha='center')
        
        # Fusion method and confidence
        plt.text(0.5, 0.5, f"Fusion Method: {fusion_result['method']}", 
                fontsize=12, ha='center')
        plt.text(0.5, 0.45, f"Overall Confidence: {fusion_result['confidence']:.2f}%", 
                fontsize=12, ha='center', fontweight='bold')
        
        # Heart rate info if available
        if 'heart_rate' in fusion_result and fusion_result['heart_rate'] > 0:
            plt.text(0.5, 0.35, f"Heart Rate: {fusion_result['heart_rate']:.1f} BPM", 
                    fontsize=12, ha='center')
            plt.text(0.5, 0.3, f"Physiologically Plausible: {fusion_result['hr_plausible']}", 
                    fontsize=12, ha='center')
            
            if 'physio_score' in fusion_result and fusion_result['physio_score'] > 0:
                plt.text(0.5, 0.25, f"Physiological Score: {fusion_result['physio_score']:.1f}/100", 
                        fontsize=12, ha='center')
        
        plt.tight_layout()
        plt.savefig('fusion_results.png')
        plt.show()
# Main execution function
def run_deepfake_detection(
    video_path=None,                     
    model_path="checkpoint.pt",          
    run_rppg=True,                      
    show_visualization=True,            
    camera_id=0,                        
    sequence_length=20,               
    frame_rate=30,                       
    batch_size=30,                       
    signal_size=270                      
):
 
    fusion_system = EnhancedFusionSystem(
        cnn_rnn_model_path=model_path,
        sequence_length=sequence_length,
        frame_rate=frame_rate,
        batch_size=batch_size,
        signal_size=signal_size
    )
    
    # Process input
    if video_path:
        # Process video file
        result = fusion_system.process_video(
            video_path=video_path,
            run_rppg=run_rppg,
            show_visualization=show_visualization
        )
    else:
        # Process webcam feed
        result = fusion_system.process_webcam(
            camera_id=camera_id,
            show_visualization=show_visualization
        )
    
    if result:
        print("\n" + "="*50)
        print("DEEPFAKE DETECTION ANALYSIS")
        print("="*50)
        
        print("CNN-RNN MODEL PREDICTION:")
        print(f"  Result: {result['cnn_rnn_prediction']}")
        print(f"  Confidence: {result['cnn_rnn_confidence']:.2f}%")
        
        print("\nrPPG PHYSIOLOGICAL ANALYSIS:")
        print(f"  Result: {result['rppg_prediction']}")
        print(f"  Confidence: {result['rppg_confidence']:.2f}%")
        
        if 'heart_rate' in result and result['heart_rate'] > 0:
            print(f"  Average Heart Rate: {result['heart_rate']:.1f} BPM")
            print(f"  Physiologically Plausible: {result['hr_plausible']}")
            
            if 'physio_score' in result and result['physio_score'] > 0:
                print(f"  Physiological Score: {result['physio_score']:.1f}/100")
        
        print("\nFUSION RESULT:")
        print(f"  FINAL PREDICTION: {result['prediction']}")
        print(f"  Overall Confidence: {result['confidence']:.2f}%")
        print(f"  Fusion Method: {result['method']}")
        
        print("="*50)
        
        if fusion_system.original_frames and len(fusion_system.original_frames) > 0:
            frame_idx = min(len(fusion_system.original_frames) // 2, len(fusion_system.original_frames) - 1)
            sample_frame = fusion_system.original_frames[frame_idx]
            cv2.imwrite('sample_frame.png', sample_frame)
            print("A sample frame from the video has been saved as 'sample_frame.png'")
        
        print("Detailed analysis visualization has been saved as 'fusion_results.png'")
    
    return result

if __name__ == "__main__":
    
    # VIDEO_PATH = r"C:\Users\KK\Desktop\dfd\Celeb-DF\Celeb-synthesis\id1_id3_0002.mp4"
    # VIDEO_PATH = r"C:\Users\KK\Desktop\dfd\Celeb-DF\Celeb-synthesis\id0_id6_0007.mp4"
    # VIDEO_PATH = r"C:\Users\KK\Desktop\dfd\Celeb-DF\Celeb-synthesis\id0_id1_0005.mp4"
    # VIDEO_PATH=r"C:\Users\KK\Downloads\WhatsApp Video 2025-04-06 at 14.38.53_1dc15000.mp4"
    # VIDEO_PATH = r"C:\Users\KK\Desktop\dfd\Celeb-DF\Celeb-real\id1_0002.mp4"
    # VIDEO_PATH = r"C:\Users\KK\Desktop\dfd\Celeb-DF\Celeb-real\id2_0007.mp4"
    # VIDEO_PATH = r"C:\Users\KK\Desktop\dfd\Celeb-DF\Celeb-real\id3_0000.mp4"
    VIDEO_PATH = r"C:\Users\KK\Downloads\test.mp4"
    # VIDEO_PATH=""
    # VIDEO_PATH = r"C:\Users\KK\Desktop\dfd\Celeb-DF\Celeb-synthesis\id0_id2_0003.mp4"
    # VIDEO_PATH = r"C:\Users\KK\Desktop\dfd\Celeb-DF\Celeb-synthesis\id0_id2_0005.mp4"
    # VIDEO_PATH = r"C:\Users\KK\Desktop\dfd\Celeb-DF\Celeb-synthesis\id0_id1_0003.mp4"

    MODEL_PATH = "checkpoint.pt"   
    
    RUN_RPPG = True                
    SHOW_VISUALIZATION = True      
    CAMERA_ID = 0                 
    
    SEQUENCE_LENGTH = 20          
    FRAME_RATE = 30                
    BATCH_SIZE = 30                
    SIGNAL_SIZE = 270             

    result = run_deepfake_detection(
        video_path=VIDEO_PATH,
        model_path=MODEL_PATH,
        run_rppg=RUN_RPPG,
        show_visualization=SHOW_VISUALIZATION,
        camera_id=CAMERA_ID,
        sequence_length=SEQUENCE_LENGTH,
        frame_rate=FRAME_RATE,
        batch_size=BATCH_SIZE,
        signal_size=SIGNAL_SIZE
    )
