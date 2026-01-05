import torch
from PIL import Image
import torchvision.transforms as T

def load_clip(device=None):
    """Load OpenAI CLIP (ViT-B/32) and return model + preprocess.

    Tries the `clip` package first (recommended)."""
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    try:
        import clip
        model, preprocess = clip.load("ViT-B/32", device=device)
        return model.eval(), preprocess, device
    except Exception:
        # Provide helpful error message; user can install via requirements.txt
        raise RuntimeError(
            "Failed to import CLIP. Install requirements (see requirements.txt) or pip install git+https://github.com/openai/CLIP.git"
        )

def pil_to_clip(image: Image.Image, preprocess):
    """Apply CLIP preprocess to a PIL image and return a tensor on CPU (batch dim added).
    """
    return preprocess(image).unsqueeze(0)
