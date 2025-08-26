"""
Training script for RTV models
"""

import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as transforms
from torch.utils.tensorboard import SummaryWriter
import argparse
import yaml
import logging
from tqdm import tqdm
import numpy as np
from PIL import Image
import json
from typing import Dict, List, Optional, Tuple

from rtv_models import RTVNet, RTVLoss, preprocess_image, postprocess_image

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class RTVDataset(Dataset):
    """Dataset for RTV training"""
    
    def __init__(self, 
                 data_root: str, 
                 split: str = 'train',
                 image_size: Tuple[int, int] = (512, 512),
                 garment_type: Optional[str] = None):
        
        self.data_root = data_root
        self.split = split
        self.image_size = image_size
        self.garment_type = garment_type
        
        # Load pairs
        pairs_file = os.path.join(data_root, 'pairs', f'{split}.txt')
        if not os.path.exists(pairs_file):
            raise FileNotFoundError(f"Pairs file not found: {pairs_file}")
            
        self.pairs = self._load_pairs(pairs_file)
        
        # Transforms
        self.transform = transforms.Compose([
            transforms.Resize(image_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])
        
        logger.info(f"Loaded {len(self.pairs)} pairs for {split}")
    
    def _load_pairs(self, pairs_file: str) -> List[Dict]:
        """Load training pairs from file"""
        pairs = []
        with open(pairs_file, 'r') as f:
            for line in f:
                line = line.strip()
                if line:
                    parts = line.split('\t')
                    if len(parts) >= 2:
                        pairs.append({
                            'person_id': parts[0],
                            'garment_id': parts[1]
                        })
        return pairs
    
    def _load_pose_keypoints(self, person_id: str) -> Optional[np.ndarray]:
        """Load pose keypoints for person"""
        pose_file = os.path.join(self.data_root, 'people', person_id, 'poses', 'keypoints.json')
        if os.path.exists(pose_file):
            with open(pose_file, 'r') as f:
                pose_data = json.load(f)
                return np.array(pose_data['keypoints']).reshape(-1, 3)
        return None
    
    def _load_garment_mask(self, garment_id: str) -> Optional[np.ndarray]:
        """Load garment segmentation mask"""
        mask_file = os.path.join(self.data_root, 'garments', garment_id, 'masks', 'mask.png')
        if os.path.exists(mask_file):
            mask = Image.open(mask_file).convert('L')
            return np.array(mask)
        return None
    
    def __len__(self):
        return len(self.pairs)
    
    def __getitem__(self, idx):
        pair = self.pairs[idx]
        person_id = pair['person_id']
        garment_id = pair['garment_id']
        
        # Load person image
        person_img_path = os.path.join(self.data_root, 'people', person_id, 'images', 'image.jpg')
        if not os.path.exists(person_img_path):
            person_img_path = person_img_path.replace('.jpg', '.png')
        
        person_img = Image.open(person_img_path).convert('RGB')
        
        # Load garment image  
        garment_img_path = os.path.join(self.data_root, 'garments', garment_id, 'images', 'image.jpg')
        if not os.path.exists(garment_img_path):
            garment_img_path = garment_img_path.replace('.jpg', '.png')
            
        garment_img = Image.open(garment_img_path).convert('RGB')
        
        # Load target (person wearing garment) - in practice this would be the ground truth
        # For now, we'll use the person image as target (identity preservation)
        target_img = person_img.copy()
        
        # Apply transforms
        person_tensor = self.transform(person_img)
        garment_tensor = self.transform(garment_img)
        target_tensor = self.transform(target_img)
        
        # Load pose keypoints
        pose_keypoints = self._load_pose_keypoints(person_id)
        if pose_keypoints is None:
            # Create dummy keypoints
            pose_keypoints = np.zeros((17, 3))
        
        # Load garment mask
        garment_mask = self._load_garment_mask(garment_id)
        if garment_mask is None:
            # Create dummy mask
            garment_mask = np.zeros(self.image_size)
        
        return {
            'person_image': person_tensor,
            'garment_image': garment_tensor,
            'target_image': target_tensor,
            'pose_keypoints': torch.FloatTensor(pose_keypoints),
            'garment_mask': torch.LongTensor(garment_mask),
            'person_id': person_id,
            'garment_id': garment_id
        }

class RTVTrainer:
    """Trainer class for RTV models"""
    
    def __init__(self, config: Dict):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Create directories
        self.checkpoint_dir = config['training']['checkpoint_dir']
        self.log_dir = config['training']['log_dir']
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        os.makedirs(self.log_dir, exist_ok=True)
        
        # Initialize model
        self.model = RTVNet(
            backbone_type=config['model']['backbone'],
            num_joints=config['model']['num_joints'],
            num_garment_classes=config['model']['num_garment_classes']
        ).to(self.device)
        
        # Loss function
        self.criterion = RTVLoss(
            perceptual_weight=config['loss']['perceptual_weight'],
            adversarial_weight=config['loss']['adversarial_weight'],
            identity_weight=config['loss']['identity_weight'],
            pose_weight=config['loss']['pose_weight'],
            segmentation_weight=config['loss']['segmentation_weight']
        ).to(self.device)
        
        # Optimizer
        self.optimizer = optim.Adam(
            self.model.parameters(),
            lr=config['training']['learning_rate'],
            weight_decay=config['training']['weight_decay']
        )
        
        # Scheduler
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=config['training']['epochs']
        )
        
        # Data loaders
        self.train_loader = self._create_dataloader('train')
        self.val_loader = self._create_dataloader('val')
        
        # Tensorboard
        self.writer = SummaryWriter(self.log_dir)
        
        # Training state
        self.epoch = 0
        self.best_val_loss = float('inf')
        
    def _create_dataloader(self, split: str) -> DataLoader:
        """Create data loader for given split"""
        dataset = RTVDataset(
            data_root=self.config['dataset']['data_root'],
            split=split,
            image_size=tuple(self.config['dataset']['image_size']),
            garment_type=self.config['dataset'].get('garment_type')
        )
        
        return DataLoader(
            dataset,
            batch_size=self.config['training']['batch_size'],
            shuffle=(split == 'train'),
            num_workers=self.config['training']['num_workers'],
            pin_memory=True
        )
    
    def train_epoch(self):
        """Train for one epoch"""
        self.model.train()
        total_loss = 0
        loss_components = {}
        
        pbar = tqdm(self.train_loader, desc=f'Epoch {self.epoch}')
        
        for batch_idx, batch in enumerate(pbar):
            # Move to device
            for key in batch:
                if isinstance(batch[key], torch.Tensor):
                    batch[key] = batch[key].to(self.device)
            
            # Forward pass
            self.optimizer.zero_grad()
            
            outputs = self.model(
                batch['person_image'], 
                batch['garment_image']
            )
            
            # Prepare targets
            targets = {
                'target_image': batch['target_image'],
                'pose_keypoints': batch['pose_keypoints'],
                'garment_masks': batch['garment_mask']
            }
            
            # Compute loss
            losses = self.criterion(outputs, targets)
            
            # Backward pass
            losses['total'].backward()
            self.optimizer.step()
            
            # Update statistics
            total_loss += losses['total'].item()
            
            for key, value in losses.items():
                if key not in loss_components:
                    loss_components[key] = 0
                loss_components[key] += value.item()
            
            # Update progress bar
            pbar.set_postfix({
                'loss': losses['total'].item(),
                'avg_loss': total_loss / (batch_idx + 1)
            })
            
            # Log to tensorboard
            global_step = self.epoch * len(self.train_loader) + batch_idx
            self.writer.add_scalar('train/batch_loss', losses['total'].item(), global_step)
        
        # Average losses
        avg_loss = total_loss / len(self.train_loader)
        avg_components = {k: v / len(self.train_loader) for k, v in loss_components.items()}
        
        return avg_loss, avg_components
    
    def validate(self):
        """Validate model"""
        self.model.eval()
        total_loss = 0
        loss_components = {}
        
        with torch.no_grad():
            for batch in tqdm(self.val_loader, desc='Validation'):
                # Move to device
                for key in batch:
                    if isinstance(batch[key], torch.Tensor):
                        batch[key] = batch[key].to(self.device)
                
                # Forward pass
                outputs = self.model(
                    batch['person_image'],
                    batch['garment_image']
                )
                
                # Prepare targets
                targets = {
                    'target_image': batch['target_image'],
                    'pose_keypoints': batch['pose_keypoints'],
                    'garment_masks': batch['garment_mask']
                }
                
                # Compute loss
                losses = self.criterion(outputs, targets)
                
                # Update statistics
                total_loss += losses['total'].item()
                
                for key, value in losses.items():
                    if key not in loss_components:
                        loss_components[key] = 0
                    loss_components[key] += value.item()
        
        # Average losses
        avg_loss = total_loss / len(self.val_loader)
        avg_components = {k: v / len(self.val_loader) for k, v in loss_components.items()}
        
        return avg_loss, avg_components
    
    def save_checkpoint(self, is_best: bool = False):
        """Save model checkpoint"""
        checkpoint = {
            'epoch': self.epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'best_val_loss': self.best_val_loss,
            'config': self.config
        }
        
        # Save latest checkpoint
        checkpoint_path = os.path.join(self.checkpoint_dir, 'latest.pth')
        torch.save(checkpoint, checkpoint_path)
        
        # Save best checkpoint
        if is_best:
            best_path = os.path.join(self.checkpoint_dir, 'best.pth')
            torch.save(checkpoint, best_path)
            logger.info(f"Best model saved at epoch {self.epoch}")
    
    def load_checkpoint(self, checkpoint_path: str):
        """Load model checkpoint"""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        self.epoch = checkpoint['epoch']
        self.best_val_loss = checkpoint['best_val_loss']
        
        logger.info(f"Checkpoint loaded from {checkpoint_path}")
    
    def train(self):
        """Main training loop"""
        logger.info("Starting training...")
        
        for epoch in range(self.epoch, self.config['training']['epochs']):
            self.epoch = epoch
            
            # Train
            train_loss, train_components = self.train_epoch()
            
            # Validate  
            val_loss, val_components = self.validate()
            
            # Update learning rate
            self.scheduler.step()
            
            # Log to tensorboard
            self.writer.add_scalar('train/epoch_loss', train_loss, epoch)
            self.writer.add_scalar('val/epoch_loss', val_loss, epoch)
            self.writer.add_scalar('train/learning_rate', self.optimizer.param_groups[0]['lr'], epoch)
            
            for key, value in train_components.items():
                self.writer.add_scalar(f'train/{key}', value, epoch)
            
            for key, value in val_components.items():
                self.writer.add_scalar(f'val/{key}', value, epoch)
            
            # Save checkpoint
            is_best = val_loss < self.best_val_loss
            if is_best:
                self.best_val_loss = val_loss
            
            self.save_checkpoint(is_best)
            
            # Log progress
            logger.info(
                f"Epoch {epoch}: "
                f"train_loss={train_loss:.4f}, "
                f"val_loss={val_loss:.4f}, "
                f"best_val_loss={self.best_val_loss:.4f}"
            )
        
        logger.info("Training completed!")
        self.writer.close()

def load_config(config_path: str) -> Dict:
    """Load training configuration"""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config

def main():
    parser = argparse.ArgumentParser(description='Train RTV model')
    parser.add_argument('--config', type=str, required=True, help='Path to config file')
    parser.add_argument('--resume', type=str, default=None, help='Path to checkpoint to resume')
    parser.add_argument('--garment', type=str, default=None, help='Specific garment type to train')
    
    args = parser.parse_args()
    
    # Load config
    config = load_config(args.config)
    
    # Override garment type if specified
    if args.garment:
        config['dataset']['garment_type'] = args.garment
        config['training']['checkpoint_dir'] = os.path.join(
            config['training']['checkpoint_dir'], args.garment
        )
        config['training']['log_dir'] = os.path.join(
            config['training']['log_dir'], args.garment
        )
    
    # Create trainer
    trainer = RTVTrainer(config)
    
    # Resume if checkpoint provided
    if args.resume:
        trainer.load_checkpoint(args.resume)
    
    # Start training
    trainer.train()

if __name__ == "__main__":
    main()