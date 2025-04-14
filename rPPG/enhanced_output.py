import matplotlib.pyplot as plt
from PIL import Image
import torch
import numpy as np
import cv2
from enhanced_output import display_enhanced_results


def display_enhanced_results(fusion_system, frames=None):
    """
    Create an enhanced visualization showing:
    1. Overall fusion prediction and confidence
    2. CNN-RNN model prediction and confidence
    3. rPPG analysis prediction and confidence
    4. Sample frame used by CNN-RNN model
    
    Parameters:
    -----------
    fusion_system : EnhancedFusionSystem
        The fusion system with populated results
    frames : torch.Tensor, optional
        CNN-RNN input frames, if available
    """
    # Prepare results data
    fusion_result = fusion_system._fuse_results()
    deepfake_results = fusion_system.deepfake_results
    rppg_results = fusion_system.rppg_results
    
    # Create figure
    plt.figure(figsize=(15, 10))
    
    # 1. Sample Frame - Top left
    plt.subplot(2, 2, 1)
    plt.title('Sample Frame Used for CNN-RNN')
    
    if frames is not None:
        # Extract and denormalize the first frame
        sample_frame = frames[0, 0].cpu().numpy().transpose(1, 2, 0)
        
        # Denormalize using the same values used in normalization
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        sample_frame = (sample_frame * std) + mean
        
        # Clip to [0, 1] range
        sample_frame = np.clip(sample_frame, 0, 1)
        
        plt.imshow(sample_frame)
    else:
        plt.text(0.5, 0.5, 'No frame available', ha='center', va='center')
    
    # 2. CNN-RNN Results - Top right
    plt.subplot(2, 2, 2)
    plt.title('CNN-RNN Deepfake Detection')
    
    labels = ['FAKE', 'REAL']
    colors = ['red', 'green']
    
    if deepfake_results.get("probabilities") is not None:
        probs = deepfake_results["probabilities"][0]
        bars = plt.bar(labels, probs, color=colors)
        plt.ylim(0, 1)
        
        # Add value labels on top of bars
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height + 0.05,
                    f'{height:.2f}', ha='center', va='bottom')
        
        # Add prediction text
        pred_color = 'green' if deepfake_results["prediction"] == 'REAL' else 'red'
        plt.text(0.5, 0.85, f"Prediction: {deepfake_results['prediction']}", 
                 transform=plt.gca().transAxes, ha='center',
                 bbox=dict(facecolor=pred_color, alpha=0.6, boxstyle='round'),
                 color='white', fontweight='bold', fontsize=12)
        
        plt.text(0.5, 0.75, f"Confidence: {deepfake_results['confidence']:.2f}%", 
                 transform=plt.gca().transAxes, ha='center')
    else:
        plt.text(0.5, 0.5, 'No CNN-RNN results available', ha='center', va='center')
    
    # 3. rPPG Results - Bottom left
    plt.subplot(2, 2, 3)
    plt.title('rPPG Heart Rate Analysis')
    
    heart_rates = rppg_results.get("heart_rates", np.array([]))
    
    if isinstance(heart_rates, np.ndarray) and heart_rates.size > 0:
        # Calculate basic metrics
        avg_hr = np.mean(heart_rates)
        hr_stability = np.std(heart_rates)
        
        # Determine if HR is physiologically plausible
        hr_is_plausible = (40 <= avg_hr <= 180) and (0.5 <= hr_stability <= 25)
        
        # Plot heart rate data
        plt.plot(heart_rates, 'lightblue', alpha=0.5, label='Raw HR')
        
        # Plot smoothed heart rates if enough data
        if heart_rates.size >= 5:
            from scipy.signal import savgol_filter
            or_try_moving_avg = lambda x, w: np.convolve(x, np.ones(w)/w, mode='valid')
            
            try:
                smoothed = savgol_filter(heart_rates, min(5, len(heart_rates) - 2), 2)
                plt.plot(smoothed, 'blue', linewidth=2, label='Smoothed HR')
            except:
                if len(heart_rates) >= 5:
                    smoothed = or_try_moving_avg(heart_rates, 5)
                    plt.plot(range(2, len(smoothed)+2), smoothed, 'blue', linewidth=2, label='Smoothed HR')
        
        plt.ylim(40, 180)
        plt.axhspan(40, 180, alpha=0.2, color='green', label='Normal HR Range')
        plt.legend(loc='upper right')
        
        # Add rPPG prediction result box
        rppg_pred = "REAL" if hr_is_plausible else "FAKE"
        pred_color = 'green' if rppg_pred == 'REAL' else 'red'
        
        plt.text(0.5, 0.2, f"rPPG Prediction: {rppg_pred}", 
                transform=plt.gca().transAxes, ha='center',
                bbox=dict(facecolor=pred_color, alpha=0.6, boxstyle='round'),
                color='white', fontweight='bold', fontsize=12)
        
        # Add key metrics
        plt.text(0.5, 0.1, f"Avg HR: {avg_hr:.1f} BPM | Confidence: {rppg_results.get('confidence', 0):.1f}%", 
                transform=plt.gca().transAxes, ha='center')
    else:
        plt.text(0.5, 0.5, 'No rPPG heart rate data available', ha='center', va='center')
    
    # 4. Final Fusion Results - Bottom right
    plt.subplot(2, 2, 4)
    plt.title('Enhanced Fusion Result')
    plt.axis('off')
    
    # Format the prediction with color box
    prediction_color = 'green' if fusion_result['prediction'] == 'REAL' else 'red'
    
    # Create a colored box for the overall prediction
    result_text = plt.text(0.5, 0.9, f"OVERALL PREDICTION: {fusion_result['prediction']}", 
                  fontsize=16, fontweight='bold', ha='center',
                  color='white', 
                  bbox=dict(facecolor=prediction_color, alpha=0.8, boxstyle='round'))
    
    # Display detailed results
    details = [
        f"Overall Confidence: {fusion_result['confidence']:.2f}%",
        f"Fusion Method: {fusion_result['method']}",
        f"",
        f"CNN-RNN Prediction: {deepfake_results['prediction']}",
        f"CNN-RNN Confidence: {deepfake_results['confidence']:.2f}%",
        f"",
        f"rPPG Prediction: {'REAL' if fusion_result.get('hr_plausible', False) else 'FAKE'}",
        f"rPPG Confidence: {fusion_result.get('rppg_confidence', 0):.1f}%",
        f"Heart Rate: {fusion_result.get('heart_rate', 0):.1f} BPM"
    ]
    
    # Add physiological score if available
    if "physio_score" in fusion_result and fusion_result["physio_score"] > 0:
        details.append(f"Physiological Score: {fusion_result['physio_score']:.1f}/100")
    
    plt.text(0.5, 0.7, '\n'.join(details), 
             fontsize=12, va='top', ha='center')
    
    # Show the relative influence in decision
    if fusion_result.get('method', '').startswith(('Weighted', 'Strong')):
        # Create axes for influence weights visualization
        ax = plt.axes([0.35, 0.25, 0.5, 0.08], frameon=True)
        
        # Approximate weights (modify these based on your fusion implementation)
        cnn_rnn_weight = 0.4
        rppg_weight = 0.6
        
        # Create horizontal stacked bar
        ax.barh(['Influence'], [cnn_rnn_weight], color='blue', alpha=0.6, label='CNN-RNN')
        ax.barh(['Influence'], [rppg_weight], left=[cnn_rnn_weight], color='green', alpha=0.6, label='rPPG')
        
        ax.set_xlim(0, 1)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.2), ncol=2)
        ax.set_title('Decision Influence Weights', fontsize=10)
    
    plt.tight_layout()
    plt.savefig('enhanced_results.png')
    plt.show()

# This can be added to the EnhancedFusionSystem class or used standalone
# Example usage:
# display_enhanced_results(fusion_system, frames)



 # VIDEO_PATH = r"C:\Users\KK\Desktop\dfd\Celeb-DF\Celeb-real\id8_0006.mp4" # Change this to your video path
    # VIDEO_PATH = r"C:\Users\KK\Desktop\dfd\Celeb-DF\Celeb-synthesis\id0_id1_0003.mp4"
    # VIDEO_PATH = r"C:\Users\KK\Desktop\dfd\Celeb-DF\Celeb-synthesis\id0_id4_0001.mp4"
    # VIDEO_PATH = r"C:\Users\KK\Desktop\dfd\Celeb-DF\Celeb-synthesis\id0_id6_0007.mp4"
    # VIDEO_PATH = r"C:\Users\KK\Desktop\dfd\Celeb-DF\Celeb-synthesis\id0_id16_0002.mp4"
    # VIDEO_PATH = r"C:\Users\KK\Desktop\dfd\Celeb-DF\Celeb-synthesis\id1_id3_0002.mp4"
    # VIDEO_PATH = r"C:\Users\KK\Desktop\dfd\Celeb-DF\Celeb-synthesis\id1_id9_0000.mp4"
    # VIDEO_PATH = r"C:\Users\KK\Desktop\dfd\Celeb-DF\Celeb-synthesis\id2_id4_0007.mp4"
    # VIDEO_PATH=r"C:\Users\KK\Pictures\Camera Roll\WIN_20250406_14_33_21_Pro.mp4"
    # VIDEO_PATH=r"C:\Users\KK\Downloads\WhatsApp Video 2025-04-06 at 14.38.53_1dc15000.mp4"