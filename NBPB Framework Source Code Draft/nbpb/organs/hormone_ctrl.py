from typing import Dict, Any, Optional, Callable
from .base_organ import BaseOrgan
from ..types import EventSeverity


class HormoneCtrlOrgan(BaseOrgan):
    """
    Organo Hormone-Ctrl - Regola learning-rate, scheduler, freezing layer (omeostasi)
    SCAFFOLD - Implementazione completa in milestone 0.2
    This organ will directly interact with the ML pipeline's training parameters.
    """
    
    def __init__(self, 
                 config: Any,
                 name: str,
                 nucleus_callback: Callable[[Dict[str, Any]], bool],
                 logger_parent_name: Optional[str] = None):
        """
        Initialize the HormoneCtrl organ.

        Args:
            config: The specific configuration for this organ
            name: The organ name (provided by Nucleus, e.g. "hormone_ctrl")
            nucleus_callback: Callback to send events to Nucleus
            logger_parent_name: Parent logger name for hierarchical logging
        """
        super().__init__(config=config,
                        name=name,
                        nucleus_callback=nucleus_callback,
                        logger_parent_name=logger_parent_name)

        self.logger.info(f"HormoneCtrlOrgan '{self.name}' initialized (SCAFFOLD - full implementation in v0.2).")

    def process(self,
                current_metrics: Optional[Dict[str, float]] = None,
                current_epoch: Optional[int] = None,
                current_step: Optional[int] = None,
                pipeline_state: Optional[Any] = None,
                **kwargs) -> Any:
        """
        Placeholder for Hormone-Ctrl processing logic.
        In the complete version, this method will analyze training metrics
        and suggest/apply adjustments to pipeline parameters.

        Args:
            current_metrics: Current training loop metrics
            current_epoch: Current epoch
            current_step: Current step
            pipeline_state: Object or dictionary representing modifiable pipeline state
            **kwargs: Additional arguments

        Returns:
            bool: True if pipeline can continue (result of Nucleus callback)
        """
        if not self.is_active:
            self.logger.debug(f"HormoneCtrlOrgan '{self.name}' is inactive, skipping process.")
            return True

        event_message = (
            f"HormoneCtrlOrgan '{self.name}' active (SCAFFOLD). "
            f"Full implementation in v0.2. "
            f"Received metrics: {current_metrics}, epoch: {current_epoch}, step: {current_step}."
        )
        
        event_data = {
            "message": "Hormone-Ctrl organ active - full implementation in v0.2",
            "planned_features": [
                "Dynamic learning rate adjustment based on validation loss plateaus",
                "Adaptive scheduler tuning (e.g., adjusting patience for ReduceLROnPlateau)",
                "Progressive layer unfreezing strategies for transfer learning", 
                "Early stopping suggestions based on homeostasis metrics",
                "Resource-aware training adjustments (e.g., batch size based on memory)"
            ],
            "current_input_metrics": current_metrics,
            "current_epoch": current_epoch,
            "current_step": current_step
        }

        should_continue = self._send_event_to_nucleus(
            event_type="hormone_ctrl_status_update",
            severity=EventSeverity.INFO,
            message=event_message,
            data=event_data
        )
        
        self.logger.debug(f"HormoneCtrlOrgan '{self.name}' process complete. Pipeline continue: {should_continue}")
        return should_continue

    def activate(self) -> None:
        """Activate the HormoneCtrl organ."""
        super().activate()

    def deactivate(self) -> None:
        """Deactivate the HormoneCtrl organ."""
        super().deactivate()
        
    def end_of_epoch(self, epoch: int, optimizer=None, model=None) -> None:
        """
        Perform end-of-epoch adjustments to learning rate and layer freezing.
        Args:
            epoch: Current epoch number
            optimizer: PyTorch optimizer instance
            model: PyTorch model instance
        """
        if not self.is_active:
            return
            
        self.logger.info(f"HormoneCtrl processing end of epoch {epoch}")
        
        # Get configuration parameters with defaults
        lr_min = getattr(self.config, 'lr_min', 1e-6)
        lr_max = getattr(self.config, 'lr_max', 1e-2)
        freeze_after_epoch = getattr(self.config, 'freeze_after_epoch', None)
        
        # Adjust learning rate if optimizer is provided
        if optimizer:
            current_lr = optimizer.param_groups[0]['lr']
            
            # Calculate new learning rate (simple decay strategy)
            decay_factor = 0.8  # 20% reduction each time
            new_lr = max(current_lr * decay_factor, lr_min)
            
            if new_lr != current_lr:
                for param_group in optimizer.param_groups:
                    param_group['lr'] = new_lr
                    
                self.logger.info(f"HormoneCtrl: learning rate adjusted from {current_lr:.6f} to {new_lr:.6f}")
                
                # Send event to nucleus
                self._send_event_to_nucleus(
                    event_type="learning_rate_adjusted",
                    severity=EventSeverity.INFO,
                    message=f"HormoneCtrl: learning rate adjusted to {new_lr:.6f}",
                    data={
                        "previous_lr": current_lr,
                        "new_lr": new_lr,
                        "epoch": epoch
                    }
                )
        
        # Freeze layers if model is provided and we've reached the freeze epoch
        if model and freeze_after_epoch is not None and epoch >= freeze_after_epoch:
            frozen_layer = None
            
            # Simple example: freeze first layer if it's a Sequential model
            if hasattr(model, 'children'):
                for i, layer in enumerate(model.children()):
                    if i == 0:  # Freeze only the first layer for this example
                        for param in layer.parameters():
                            param.requires_grad = False
                        frozen_layer = f"Layer {i}"
                        break
            
            if frozen_layer:
                self.logger.info(f"HormoneCtrl: {frozen_layer} frozen after epoch {epoch}")
                
                # Send event to nucleus
                self._send_event_to_nucleus(
                    event_type="layer_frozen",
                    severity=EventSeverity.INFO,
                    message=f"HormoneCtrl: {frozen_layer} frozen after epoch {epoch}",
                    data={
                        "frozen_layer": frozen_layer,
                        "epoch": epoch
                    }
                )
