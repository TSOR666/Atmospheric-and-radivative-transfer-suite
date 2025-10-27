"""Training helpers for the MDNO model."""
from collections import defaultdict
from typing import DefaultDict, Dict, List

import numpy as np
import torch
from torch.cuda.amp import GradScaler, autocast

from .config import MDNOConfig
from .model import EnhancedMDNO_v53_Complete
from ._logging import LOGGER as logger

class MDNOTrainer:
    """Comprehensive training system"""
    
    def __init__(self, model: EnhancedMDNO_v53_Complete, config: MDNOConfig):
        self.model = model
        self.config = config
        self.device = config.get_device()
        
        self.optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay
        )
        
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            self.optimizer, T_0=50, T_mult=2
        )
        
        # AUDIT FIX: Guard GradScaler for CPU compatibility
        self.scaler = GradScaler(enabled=(config.use_mixed_precision and self.device.type == "cuda"))
        self.loss_history: DefaultDict[str, List[float]] = defaultdict(list)
        self.best_loss = float('inf')
        
        logger.info("Trainer initialized")
    
    def train_step(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        self.model.train()
        self.optimizer.zero_grad()
        
        batch = {k: v.to(self.device) if torch.is_tensor(v) else v 
                for k, v in batch.items()}
        
        if self.scaler.is_enabled():
            with autocast():
                outputs = self.model(batch['inputs'])
                losses = self.model.compute_physics_loss(
                    outputs, batch['targets'], batch['inputs']
                )
                total_loss = losses['total']
        else:
            outputs = self.model(batch['inputs'])
            losses = self.model.compute_physics_loss(
                outputs, batch['targets'], batch['inputs']
            )
            total_loss = losses['total']
        
        if self.scaler.is_enabled():
            self.scaler.scale(total_loss).backward()
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.gradient_clip)
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.gradient_clip)
            self.optimizer.step()
        
        # AUDIT FIX: Step scheduler
        if self.scheduler is not None:
            self.scheduler.step()
        
        loss_dict = {k: v.item() if torch.is_tensor(v) else v 
                    for k, v in losses.items()}
        
        return loss_dict
    
    def validate(self, val_loader) -> Dict[str, float]:
        self.model.eval()
        val_losses = defaultdict(list)
        
        with torch.no_grad():
            for batch in val_loader:
                batch = {k: v.to(self.device) if torch.is_tensor(v) else v 
                        for k, v in batch.items()}
                
                outputs = self.model(batch['inputs'])
                losses = self.model.compute_physics_loss(
                    outputs, batch['targets'], batch['inputs']
                )
                
                for k, v in losses.items():
                    val_losses[k].append(v.item() if torch.is_tensor(v) else v)
        
        avg_losses = {k: np.mean(v) for k, v in val_losses.items()}
        
        if avg_losses['total'] < self.best_loss:
            self.best_loss = avg_losses['total']
            self.save_checkpoint('best_model.pt')
        
        return avg_losses
    
    def save_checkpoint(self, filename: str):
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'config': self.config,
            'best_loss': self.best_loss,
            'loss_history': dict(self.loss_history)
        }
        
        if self.scaler.is_enabled():
            checkpoint['scaler_state_dict'] = self.scaler.state_dict()
        
        torch.save(checkpoint, filename)
        logger.info(f"Checkpoint saved to {filename}")
    
    def load_checkpoint(self, filename: str):
        checkpoint = torch.load(filename, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        self.best_loss = checkpoint.get('best_loss', float('inf'))
        self.loss_history = defaultdict(list, checkpoint.get('loss_history', {}))
        
        if self.scaler.is_enabled() and 'scaler_state_dict' in checkpoint:
            self.scaler.load_state_dict(checkpoint['scaler_state_dict'])
        
        logger.info(f"Checkpoint loaded from {filename}")

# ============================================================================
# TESTING (AUDIT FIX)
# ============================================================================

