from contextlib import suppress

import torch
from torch import nn


class _BaseCAM:
    """
    Base class for gradient-based Class Activation Mapping methods.
    Safely manages forward/backward hooks and GPU memory cleanup.
    """

    def __init__(
        self,
        model: nn.Module,
        target_layer: nn.Module,
    ):
        self.model = model
        self.target_layer = target_layer

        self.activations: torch.Tensor | None = None
        self.gradients: torch.Tensor | None = None

        self.forward_handle = self.target_layer.register_forward_hook(
            self._save_activations
        )
        self.backward_handle = self.target_layer.register_full_backward_hook(
            self._save_gradients
        )

    # ---------------------------------------------------------
    # Hooks
    # ---------------------------------------------------------

    def _save_activations(self, module, inputs, output):
        self.activations = output.detach()

    def _save_gradients(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()

    # ---------------------------------------------------------
    # Weight Computation (Implemented by Subclasses)
    # ---------------------------------------------------------

    def _compute_weights(
        self,
        activations: torch.Tensor,
        gradients: torch.Tensor,
    ) -> torch.Tensor:
        raise NotImplementedError

    # ---------------------------------------------------------
    # Generate CAM
    # ---------------------------------------------------------

    def generate(
        self,
        input_tensor: torch.Tensor,
        target_class: int | None = None,
    ):
        """
        Generate a normalized CAM heatmap.

        Args:
            input_tensor: Input tensor with shape [1, C, H, W].
            target_class: Class index to explain. If None, uses predicted class.

        Returns:
            cam: Normalized 2D heatmap tensor [H', W'].
            predicted_class: Model's predicted class index (int).
            probabilities: Softmax probabilities tensor [num_classes].
        """
        self.activations = None
        self.gradients = None
        self.model.zero_grad(set_to_none=True)

        # Ensure gradient tracking works even if all model parameters are frozen
        if not input_tensor.requires_grad:
            input_tensor = input_tensor.detach().requires_grad_(True)

        with torch.enable_grad():
            outputs = self.model(input_tensor)
            probabilities = torch.softmax(outputs, dim=1)
            predicted_class = outputs.argmax(dim=1).item()

            if target_class is None:
                target_class = predicted_class

            target_score = outputs[0, target_class]
            target_score.backward()

        activations = self.activations
        gradients = self.gradients

        if activations is None:
            raise RuntimeError("Activations were not captured.")

        if gradients is None:
            raise RuntimeError("Gradients were not captured.")

        weights = self._compute_weights(activations, gradients)

        # Weighted combination of feature maps + ReLU
        cam = (weights * activations).sum(dim=1)
        cam = torch.relu(cam)[0]

        # Min-max normalization to [0, 1]
        cam_min = cam.min()
        cam_max = cam.max()
        cam = cam - cam_min

        if cam_max > cam_min:
            cam = cam / (cam_max - cam_min)

        # Release references to avoid retaining VRAM between calls
        self.activations = None
        self.gradients = None
        self.model.zero_grad(set_to_none=True)

        return (
            cam.detach(),
            predicted_class,
            probabilities[0].detach(),
        )

    # ---------------------------------------------------------
    # Cleanup & Context Manager
    # ---------------------------------------------------------

    def remove_hooks(self):
        """Remove registered forward and backward hooks."""
        if self.forward_handle is not None:
            self.forward_handle.remove()
            self.forward_handle = None

        if self.backward_handle is not None:
            self.backward_handle.remove()
            self.backward_handle = None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.remove_hooks()
        return False

    def __del__(self):
        with suppress(AttributeError, RuntimeError, ReferenceError):
            self.remove_hooks()


class GradCAM(_BaseCAM):
    """
    Standard Grad-CAM implementation.
    Works with CNN-based classification models.
    """

    def _compute_weights(
        self,
        activations: torch.Tensor,
        gradients: torch.Tensor,
    ) -> torch.Tensor:
        return gradients.mean(dim=(2, 3), keepdim=True)


class GradCAMPlusPlus(_BaseCAM):
    """
    Grad-CAM++ implementation (Chattopadhay et al., WACV 2018).
    Provides sharper localization when multiple small lesions are present.
    """

    def _compute_weights(
        self,
        activations: torch.Tensor,
        gradients: torch.Tensor,
    ) -> torch.Tensor:
        grads_power_2 = gradients.pow(2)
        grads_power_3 = grads_power_2 * gradients
        sum_activations = activations.sum(dim=(2, 3), keepdim=True)

        eps = 1e-7
        denominator = 2.0 * grads_power_2 + sum_activations * grads_power_3 + eps
        alpha = grads_power_2 / denominator
        alpha = torch.where(gradients != 0, alpha, torch.zeros_like(alpha))

        positive_gradients = torch.relu(gradients)
        weights = (alpha * positive_gradients).sum(dim=(2, 3), keepdim=True)
        return weights
