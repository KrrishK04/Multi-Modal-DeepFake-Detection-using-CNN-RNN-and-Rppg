import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import os

from configs.config import Config
from models.transppgnet import TransPPGNet
from utils.data_loader_enhanced import get_transppgnet_loader
from utils.device import get_device

def train_epoch(model, loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    pbar = tqdm(loader, desc="Training")
    for batch_idx, (video_clips, rppg_signals, labels) in enumerate(pbar):
        # Move to device
        video_clips = video_clips.to(device)
        rppg_signals = rppg_signals.to(device)
        labels = labels.to(device)
        
        optimizer.zero_grad()
        
        # Forward pass
        # returns logits, attn_weights
        outputs, _ = model(video_clips, rppg_signals)
        
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        # Stats
        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
        
        pbar.set_postfix({'loss': running_loss/(batch_idx+1), 'acc': 100.*correct/total})
        
    return running_loss / len(loader), 100. * correct / total

def validate(model, loader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for video_clips, rppg_signals, labels in tqdm(loader, desc="Validating"):
            video_clips = video_clips.to(device)
            rppg_signals = rppg_signals.to(device)
            labels = labels.to(device)
            
            outputs, _ = model(video_clips, rppg_signals)
            loss = criterion(outputs, labels)
            
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
    return running_loss / len(loader), 100. * correct / total

def main():
    Config.print_config()
    device = Config.DEVICE
    
    # 1. Initialize Model
    print("Initializing TransPPGNet...")
    model = TransPPGNet(num_classes=2, pretrained=True)
    model = model.to(device)
    
    # 2. Data Loaders
    print("Loading Data...")
    # UPDATE THIS PATH to your actual data location
    data_dir = Config.BASE_DATA_DIR 
    train_loader = get_transppgnet_loader(data_dir, batch_size=Config.BATCH_SIZE, split='train')
    val_loader = get_transppgnet_loader(data_dir, batch_size=Config.BATCH_SIZE, split='val')
    
    # 3. Setup Training
    criterion = nn.CrossEntropyLoss().to(device)
    optimizer = optim.AdamW(model.parameters(), lr=Config.LEARNING_RATE, weight_decay=Config.WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10)
    
    num_epochs = 10
    best_acc = 0.0
    
    # 4. Training Loop
    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch+1}/{num_epochs}")
        
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc = validate(model, val_loader, criterion, device)
        
        scheduler.step()
        
        print(f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%")
        print(f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}%")
        
        # Checkpoint
        if val_acc > best_acc:
            best_acc = val_acc
            save_path = os.path.join(Config.CHECKPOINT_DIR, "transppgnet_best.pth")
            os.makedirs(Config.CHECKPOINT_DIR, exist_ok=True)
            torch.save(model.state_dict(), save_path)
            print(f"Saved Best Model to {save_path}")

if __name__ == "__main__":
    main()
