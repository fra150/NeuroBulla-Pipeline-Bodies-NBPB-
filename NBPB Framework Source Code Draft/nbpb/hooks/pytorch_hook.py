import torch
import torch.nn as nn
from typing import Dict, Any, Optional, Callable
import numpy as np
import time
from ..nucleus import NBPBNucleus
from ..types import InterventionType
from ..utils.logging import NBPBLogger

class PyTorchHook:
    """
    Universal hook to integrate NBPB with PyTorch pipelines
    Supports:
    - Standard PyTorch training loops
    - HuggingFace Trainer (easy extension)
    - Custom training loops
    """
    
    def __init__(self, nucleus: Optional[NBPBNucleus] = None, config: Optional[Dict[str, Any]] = None):
        self.nucleus: Optional[NBPBNucleus] = nucleus
        self.optimizer_ref: Optional[torch.optim.Optimizer] = None
        self.scheduler_ref: Optional[torch.optim.lr_scheduler._LRScheduler] = None
        self.model_ref: Optional[nn.Module] = None
        
        self.hooks_handles = []
        self.training_active = False
        self.logger = NBPBLogger("PyTorchHook")
        
        if self.nucleus:
            self.logger.info("PyTorchHook initialized with NBPBNucleus instance.")

    def set_nucleus(self, nucleus: NBPBNucleus):
        """Sets the Nucleus and registers callbacks if not already done."""
        self.nucleus = nucleus
        self.logger.info(f"NBPBNucleus instance set for PyTorchHook.")

    def set_optimizer(self, optimizer: torch.optim.Optimizer):
        """Sets the optimizer reference."""
        self.optimizer_ref = optimizer
        self.logger.info("Optimizer reference set.")

    def set_scheduler(self, scheduler: torch.optim.lr_scheduler._LRScheduler):
        """Sets the scheduler reference."""
        self.scheduler_ref = scheduler
        self.logger.info("Scheduler reference set.")
    
    def attach_to_model(self, model: nn.Module):
        """Attaches hooks to model layers for monitoring"""
        self.model_ref = model
        
        def forward_hook(module, input, output):
            if self.training_active and self.nucleus and self.nucleus.is_active:
                if 'watcher' in self.nucleus.organs:
                    watcher = self.nucleus.organs['watcher']
                    watcher.process(output, stage=f"{module.__class__.__name__}_forward")
                    
        def backward_hook(module, grad_input, grad_output):
            if self.training_active and self.nucleus and self.nucleus.is_active:
                if grad_output[0] is not None and 'watcher' in self.nucleus.organs:
                    watcher = self.nucleus.organs['watcher']
                    watcher.process(grad_output[0], stage=f"{module.__class__.__name__}_backward")
        
        for name, module in model.named_modules():
            if isinstance(module, (nn.Linear, nn.Conv2d, nn.LSTM, nn.Transformer)):
                handle_fwd = module.register_forward_hook(forward_hook)
                handle_bwd = module.register_full_backward_hook(backward_hook)
                self.hooks_handles.extend([handle_fwd, handle_bwd])
        
        self.logger.info(f"Attached monitoring hooks to {len(self.hooks_handles)//2} model layers")
    
    def monitor_data(self, features: torch.Tensor, target: torch.Tensor = None, feature_names: list = None, stage: str = "data_processing") -> bool:
        """
        Monitors data through NBPB organs
        Returns:
            bool: True if safe to continue, False if training should stop
        """
        if not self.nucleus or not self.nucleus.is_active:
            return True
        continue_training = True 
        if 'watcher' in self.nucleus.organs:
            watcher_ok = self.nucleus.organs['watcher'].process(features, stage=stage)
            continue_training = continue_training and watcher_ok
            
        if 'immuno_guard' in self.nucleus.organs and target is not None:
            immuno_ok = self.nucleus.organs['immuno_guard'].process(
                features, target, feature_names, stage=stage
            )
            continue_training = continue_training and immuno_ok       
        return continue_training
    
    def _adjust_pytorch_parameters(self, params_to_adjust: Dict[str, Any]) -> bool:
        """Callback for Nucleus to modify PyTorch parameters."""
        if not self.training_active:
            self.logger.warning("Parameter adjustment requested while training is not active.")
            return False

        self.logger.info(f"Attempting to adjust PyTorch parameters: {params_to_adjust}")
        adjusted_something = False
        
        if 'learning_rate' in params_to_adjust and self.optimizer_ref:
            new_lr = float(params_to_adjust['learning_rate'])
            for param_group in self.optimizer_ref.param_groups:
                param_group['lr'] = new_lr
            self.logger.info(f"Optimizer learning rate adjusted to {new_lr}")
            adjusted_something = True
            
        return adjusted_something

    def _modify_pytorch_data(self, data_batch: Any) -> Any:
        """Callback for Nucleus to modify batch data."""
        self.logger.info(f"Data modification requested for batch of type {type(data_batch)}")
        return data_batch
    
    def training_step_wrapper(self, original_step_fn: Callable) -> Callable:
        """
        Wrapper for training step that integrates NBPB monitoring
        Args:
            original_step_fn: Original training step function
        Returns:
            Wrapped function with NBPB integration
        """
        def wrapped_step(*args, **kwargs):
            self.training_active = True
            if not self.nucleus or not self.nucleus.is_active:
                return original_step_fn(*args, **kwargs) 
            try:
                result = original_step_fn(*args, **kwargs)
                if hasattr(result, 'loss') and result.loss is not None:
                    if 'watcher' in self.nucleus.organs:
                        loss_tensor = result.loss if torch.is_tensor(result.loss) else torch.tensor(result.loss)
                        self.nucleus.organs['watcher'].process(loss_tensor, stage="loss_monitoring")
                return result

            except Exception as e:
                self.nucleus.receive_event({
                    "timestamp": time.time(),
                    "organ": "PyTorchHook",
                    "event_type": "training_step_error",
                    "severity": "critical",
                    "data": {"error": str(e), "error_type": type(e).__name__}
                })
                raise               
        return wrapped_step
    
    def start_training(self):
        """Starts training - activates monitoring"""
        self.training_active = True
        if self.nucleus:
            self.nucleus.activate()
            
    def stop_training(self):
        """Ends training - deactivates monitoring"""
        self.training_active = False
        if self.nucleus:
            self.nucleus.deactivate()
        
    def cleanup(self):
        """Removes all registered hooks"""
        for hook in self.hooks_handles:
            hook.remove()
        self.hooks_handles.clear()
    
    def end_of_epoch(self, epoch: int) -> None:
        """
        Called at the end of each epoch to allow organs to perform epoch-level adjustments.
        Args:
            epoch: Current epoch number
        """
        if not self.nucleus or not self.nucleus.is_active:
            return 
        self.logger.info(f"End of epoch {epoch} - triggering organ adjustments")
        
        # Notify HormoneCtrl for learning rate adjustments
        if 'hormone_ctrl' in self.nucleus.organs:
            hormone_ctrl = self.nucleus.organs['hormone_ctrl']
            if hasattr(hormone_ctrl, 'end_of_epoch'):
                hormone_ctrl.end_of_epoch(epoch, self.optimizer_ref, self.model_ref)
        
        # Notify other organs that might need epoch-level processing
        for organ_name, organ in self.nucleus.organs.items():
            if hasattr(organ, 'end_of_epoch') and organ_name != 'hormone_ctrl':
                try:
                    organ.end_of_epoch(epoch)
                except Exception as e:
                    self.logger.warning(f"Organ {organ_name} failed during end_of_epoch: {e}")
    
    def get_training_report(self) -> Dict[str, Any]:
        """Generates complete training report"""
        if not self.nucleus:
            return {}  
        return {
            "nbpb_health_report": self.nucleus.get_health_report(),
            "watcher_summary": self.nucleus.organs['watcher'].get_data_health_summary() if 'watcher' in self.nucleus.organs else {},
            "immuno_correlation_report": self.nucleus.organs['immuno_guard'].get_correlation_report() if 'immuno_guard' in self.nucleus.organs else {}
        }