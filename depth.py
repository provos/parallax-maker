# (c) 2024 Niels Provos

import torch
from utils import torch_get_device

class DepthEstimationModel:
    def __init__(self, model="midas"):
        assert model in ["midas", "zoedepth"]
        self.model_type = model
        self.model = None
        self.transforms = None
        
    def __eq__(self, other):
        if not isinstance(other, DepthEstimationModel):
            return False
        return self.model_type == other.model_type
        
    def load_model(self, progress_callback=None):
        load_pipeline = {
            "midas": create_medias_pipeline,
            "zoedepth": create_zoedepth_pipeline
        }

        result = load_pipeline[self.model_type](
            progress_callback=progress_callback)

        if self.model_type == "midas":
            self.model, self.transforms = result
        elif self.model_type == "zoedepth":
            self.model = result

    def depth_map(self, image, progress_callback=None):
        if self.model is None:
            self.load_model()

        run_pipeline = {
            "midas": lambda img, cb: run_medias_pipeline(img, self.model, self.transforms, progress_callback=cb),
            "zoedepth": lambda img, cb: run_zoedepth_pipeline(img, self.model, progress_callback=cb)
        }

        return run_pipeline[self.model_type](image, progress_callback)

def create_medias_pipeline(progress_callback=None):
    """
    Creates a media pipeline using the MiDaS model for depth estimation.

    Args:
        progress_callback (callable, optional): A callback function to report progress. Defaults to None.

    Returns:
        tuple: A tuple containing the MiDaS model and the transformation pipeline.

    """
    # Load the MiDaS v2.1 model
    model_type = "DPT_Large"
    midas = torch.hub.load("intel-isl/MiDaS", model_type, skip_validation=True)

    if progress_callback:
        progress_callback(30, 100)

    # Set the model to evaluation mode
    midas.eval()

    # Define the transformation pipeline
    midas_transforms = torch.hub.load(
        "intel-isl/MiDaS", "transforms", skip_validation=True)
    if model_type == "DPT_Large" or model_type == "DPT_Hybrid":
        transforms = midas_transforms.dpt_transform
    else:
        transforms = midas_transforms.small_transform

    if progress_callback:
        progress_callback(50, 100)

    # Set the device (CPU or GPU)
    midas.to(torch_get_device())

    return midas, transforms


def run_medias_pipeline(image, midas, transforms, progress_callback=None):
    """
    Runs the media pipeline for segmentation.

    Args:
        image (numpy.ndarray): The input image.
        midas (torch.nn.Module): The MIDAS model.
        transforms (torchvision.transforms.Compose): The image transforms.
        progress_callback (callable, optional): A callback function to report progress.

    Returns:
        numpy.ndarray: The predicted segmentation mask.
    """
    input_batch = transforms(image).to(torch_get_device())
    with torch.no_grad():
        prediction = midas(input_batch)

        prediction = torch.nn.functional.interpolate(
            prediction.unsqueeze(1),
            size=image.shape[:2],
            mode="bicubic",
            align_corners=False,
        ).squeeze()

    if progress_callback:
        progress_callback(90, 100)

    return prediction.cpu().numpy()


def midas_depth_map(image, progress_callback=None):
    if progress_callback:
        progress_callback(0, 100)

    midas, transforms = create_medias_pipeline(
        progress_callback=progress_callback)

    depth_map = run_medias_pipeline(
        image, midas, transforms, progress_callback=progress_callback)

    if progress_callback:
        progress_callback(100, 100)

    return depth_map


def create_zoedepth_pipeline(progress_callback=None):
    # Triggers fresh download of MiDaS repo
    torch.hub.help("intel-isl/MiDaS", "DPT_BEiT_L_384", force_reload=True)

    # Zoe_NK
    model_zoe_nk = torch.hub.load(
        "isl-org/ZoeDepth", "ZoeD_NK", pretrained=True, skip_validation=True)

    # Set the device (CPU or GPU)
    device = torch_get_device()
    model_zoe_nk.to(device)
    
    if progress_callback:
        progress_callback(50, 100)

    return model_zoe_nk


def run_zoedepth_pipeline(image, model_zoe_nk, progress_callback=None):
    depth_map = model_zoe_nk.infer_pil(image)  # as numpy

    # invert the depth map since we are expecting the farthest objects to be black
    depth_map = 255 - depth_map

    if progress_callback:
        progress_callback(100, 100)

    return depth_map


def zoedepth_depth_map(image, progress_callback=None):
    model_zoe_nk = create_zoedepth_pipeline(
        progress_callback=progress_callback)

    return run_zoedepth_pipeline(image, model_zoe_nk, progress_callback=progress_callback)
