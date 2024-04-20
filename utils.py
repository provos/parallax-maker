# (c) 2024 Niels Provos
# 

import io
import base64
import torch


def pil_to_data_url(pil_image):
    """Converts a PIL image to a data URL."""
    buffered = io.BytesIO()
    pil_image.save(buffered, format="PNG")
    img_str = base64.b64encode(buffered.getvalue()).decode('utf-8')
    return f"data:image/png;base64,{img_str}"


def torch_get_device():
    """
    Returns the appropriate torch device based on the availability of CUDA or MPS.

    Returns:
        torch.device: The torch device (cuda, mps, or cpu) based on availability.
    """
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")