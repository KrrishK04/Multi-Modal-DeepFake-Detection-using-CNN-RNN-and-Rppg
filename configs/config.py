import os
import platform
import torch
from utils.device import get_device, get_optimal_workers, get_pin_memory


class Config:
    """
    Central configuration for the DeepFake Detection training pipeline.

    Everything adapts automatically based on the detected accelerator:
      • Windows + NVIDIA  →  CUDA   (cuDNN benchmark, pinned memory, 4 workers)
      • macOS + M4 Pro    →  MPS    (unified-memory tuning, 0 workers, no pin)
      • Anything else     →  CPU
    """

    # ──────────────────────────── Device ────────────────────────────
    DEVICE = get_device()

    # ──────────────────────────── Paths ─────────────────────────────
    PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    if platform.system() == "Windows":
        _DEFAULT_DATA_DIR = r"C:\Users\KK\Desktop\dfd\Data"
    else:
        _DEFAULT_DATA_DIR = PROJECT_ROOT

    BASE_DATA_DIR     = os.environ.get("DFD_DATA_DIR", _DEFAULT_DATA_DIR)
    CROPPED_FACES_DIR = os.path.join(BASE_DATA_DIR, "CroppedFaces")
    LABELS_CSV        = os.path.join(BASE_DATA_DIR, "Gobal_metadata.csv")
    
    # ──────────────────────────── Model/Experiment Name ────────────────
    # Change this for each new model configuration to prevent overwriting
    # Previous model data will be preserved in its own directory
    MODEL_NAME = "30Frames"  # subfolder under checkpoints/ for save/load; must match where checkpoint.pt lives
    
    CHECKPOINT_DIR    = os.path.join(PROJECT_ROOT, "checkpoints", MODEL_NAME)

    # ──────────────────────────── DataLoader ────────────────────────
    NUM_WORKERS = get_optimal_workers(DEVICE)
    PIN_MEMORY  = get_pin_memory(DEVICE)

    # ──────────────────────────── Training ──────────────────────────
    NUM_EPOCHS      = 20
    LEARNING_RATE   = 1e-4      # higher LR for transformer (AdamW + cosine)
    WEIGHT_DECAY    = 0.05      # standard for ViT fine-tuning

    if DEVICE.type == "cuda":
        BATCH_SIZE = 8          # ViT-S + 16 frames × 224² is heavier
    elif DEVICE.type == "mps":
        BATCH_SIZE = 8
    else:
        BATCH_SIZE = 2

    # ──────────────────────────── Backbone ──────────────────────────
    BACKBONE = "vit_small"      # "vit_small" | "resnext50" (legacy)

    # ──────────────────────────── Image ─────────────────────────────
    IM_SIZE = 224               # ViT-S/16 expects 224×224
    MEAN    = [0.485, 0.456, 0.406]
    STD     = [0.229, 0.224, 0.225]

    # ──────────────────────────── Clip / Sequence ───────────────────
    CLIP_LEN        = 30        # fixed number of frames per clip
    SEQUENCE_LENGTH = CLIP_LEN  # alias for backward compat

    # ──────────────────────────── ViT Temporal Transformer ─────────
    TEMPORAL_LAYERS   = 4
    TEMPORAL_HEADS    = 6
    TEMPORAL_FF       = 1536
    TEMPORAL_DROPOUT  = 0.1
    T_MAX             = 64      # max temporal positions supported
    USE_CLS_TOKEN     = True

    # ──────────────────────────── Model (legacy ResNeXt+LSTM) ──────
    NUM_CLASSES   = 2
    LATENT_DIM    = 2048
    LSTM_LAYERS   = 1
    HIDDEN_DIM    = 2048
    BIDIRECTIONAL = False

    @classmethod
    def print_config(cls):
        print("=" * 60)
        print("  Configuration")
        print("=" * 60)
        print(f"  Platform         : {platform.system()} {platform.machine()}")
        print(f"  Device           : {cls.DEVICE}")
        print(f"  Backbone         : {cls.BACKBONE}")
        print(f"  Batch size       : {cls.BATCH_SIZE}")
        print(f"  Clip length      : {cls.CLIP_LEN}")
        print(f"  Num workers      : {cls.NUM_WORKERS}")
        print(f"  Pin memory       : {cls.PIN_MEMORY}")
        print(f"  Data dir         : {cls.BASE_DATA_DIR}")
        print(f"  Model name       : {cls.MODEL_NAME}")
        print(f"  Checkpoint dir   : {cls.CHECKPOINT_DIR}")
        print(f"  Epochs           : {cls.NUM_EPOCHS}")
        print(f"  Learning rate    : {cls.LEARNING_RATE}")
        print(f"  Weight decay     : {cls.WEIGHT_DECAY}")
        print(f"  Image size       : {cls.IM_SIZE}")
        print(f"  Num classes      : {cls.NUM_CLASSES}")
        if cls.BACKBONE == "vit_small":
            print(f"  Temporal layers  : {cls.TEMPORAL_LAYERS}")
            print(f"  Temporal heads   : {cls.TEMPORAL_HEADS}")
            print(f"  CLS token        : {cls.USE_CLS_TOKEN}")
        print("=" * 60)
