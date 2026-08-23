import torch
from torch import nn


class GradCAM:
    """
    Generic Grad-CAM implementation.

    Works with CNN-based classification models.
    """

    def __init__(
        self,
        model: nn.Module,
        target_layer: nn.Module
    ):
        self.model = model
        self.target_layer = target_layer

        self.activations = None
        self.gradients = None

        self.forward_handle = (
            self.target_layer.register_forward_hook(
                self._save_activations
            )
        )

        self.backward_handle = (
            self.target_layer.register_full_backward_hook(
                self._save_gradients
            )
        )

    # ---------------------------------------------------------
    # Hooks
    # ---------------------------------------------------------

    def _save_activations(
        self,
        module,
        inputs,
        output
    ):
        self.activations = output

    def _save_gradients(
        self,
        module,
        grad_input,
        grad_output
    ):
        self.gradients = grad_output[0]

    # ---------------------------------------------------------
    # Generate Grad-CAM
    # ---------------------------------------------------------

    def generate(
        self,
        input_tensor: torch.Tensor,
        target_class: int | None = None
    ):
        """
        Generate a Grad-CAM heatmap.

        Args:
            input_tensor:
                Input tensor with shape [1, C, H, W].

            target_class:
                Class index to explain.
                If None, the predicted class is used.

        Returns:
            cam:
                Normalized Grad-CAM heatmap as a tensor.

            predicted_class:
                Model's predicted class index.

            probabilities:
                Softmax probabilities for all classes.
        """

        self.model.zero_grad()

        # Forward pass
        outputs = self.model(input_tensor)

        probabilities = torch.softmax(
            outputs,
            dim=1
        )

        predicted_class = (
            outputs.argmax(dim=1).item()
        )

        # Use predicted class if no target is specified
        if target_class is None:
            target_class = predicted_class

        # Score for target class
        target_score = outputs[
            0,
            target_class
        ]

        # Backward pass
        target_score.backward()

        # Feature maps
        activations = self.activations

        # Gradients
        gradients = self.gradients

        if activations is None:
            raise RuntimeError(
                "Activations were not captured."
            )

        if gradients is None:
            raise RuntimeError(
                "Gradients were not captured."
            )

        # -----------------------------------------------------
        # Global average pooling of gradients
        # -----------------------------------------------------

        weights = gradients.mean(
            dim=(2, 3),
            keepdim=True
        )

        # -----------------------------------------------------
        # Weighted feature maps
        # -----------------------------------------------------

        cam = (
            weights * activations
        ).sum(dim=1)

        # Remove negative influence
        cam = torch.relu(cam)

        # Remove batch dimension
        cam = cam[0]

        # -----------------------------------------------------
        # Normalize
        # -----------------------------------------------------

        cam_min = cam.min()
        cam_max = cam.max()

        cam = cam - cam_min

        if cam_max > cam_min:
            cam = cam / (
                cam_max - cam_min
            )

        return (
            cam.detach(),
            predicted_class,
            probabilities[
                0
            ].detach()
        )

    # ---------------------------------------------------------
    # Cleanup
    # ---------------------------------------------------------

    def remove_hooks(self):
        """
        Remove registered forward and backward hooks.
        """

        self.forward_handle.remove()
        self.backward_handle.remove()

    def __del__(self):
        """
        Attempt to remove hooks when the object is destroyed.
        """

        try:
            self.remove_hooks()
        except Exception:
            pass