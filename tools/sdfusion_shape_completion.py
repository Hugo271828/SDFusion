"""Run shape completion with trained SDFusion + VQ-VAE checkpoints.

All configuration such as checkpoint locations, dataset paths, sampling
hyper-parameters, and masking ranges are intentionally hard-coded to make the
script self-contained without relying on command-line arguments.
"""

from pathlib import Path
from typing import Dict, Tuple

import mcubes
import torch
import torch.nn.functional as F
import trimesh

from datasets.base_dataset import CreateDataset
from models.base_model import create_model
from models.networks.diffusion_networks.samplers.ddim import DDIMSampler
from utils.demo_util import SDFusionOpt
from utils.demo_util import get_partial_shape


# -----------------------------------------------------------------------------
# Hard-coded configuration
# -----------------------------------------------------------------------------

# Device & gpu setup ---------------------------------------------------------
USE_GPU = torch.cuda.is_available()
GPU_IDS = [0] if USE_GPU else []
DEVICE_STR = "cuda:0" if USE_GPU else "cpu"

# Paths ----------------------------------------------------------------------
DATAROOT = "/workspace/data"  # Root directory that contains the ShapeNet SDFs
SDFUSION_CKPT = (
    "saved_ckpt/sdfusion-snet-chair.pth"
)  # Path to the trained diffusion checkpoint
VQVAE_CKPT = (
    "saved_ckpt/vqvae-snet-chair.pth"
)  # Path to the trained VQ-VAE checkpoint
OUTPUT_DIR = Path("completion_results")

# Dataset & sampling settings ------------------------------------------------
CATEGORY = "chair"
RESOLUTION = 128
TRUNC_THRESHOLD = 0.2
SAMPLE_INDEX = 0

# Diffusion sampling hyper-parameters ---------------------------------------
DDIM_STEPS = 100
DDIM_ETA = 0.0
UNCOND_SCALE = None  # use model default

# Spatial window defining the visible (known) region of the partial shape ----
XYZ_DICT: Dict[str, Tuple[float, float]] = {
    "x": (-1.0, 1.0),
    "y": (-1.0, -0.1),
    "z": (-1.0, 1.0),
}

# Surface extraction ---------------------------------------------------------
ISO_LEVEL = 0.02


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------

def _export_sdf_as_ply(sdf_tensor: torch.Tensor, out_path: Path, level: float = ISO_LEVEL) -> None:
    """Convert a single SDF volume into a mesh and write it as a PLY file."""

    volume = sdf_tensor.detach().cpu().squeeze().numpy()

    if volume.ndim != 3:
        raise ValueError(f"Expected a 3D volume, received shape {volume.shape}.")

    try:
        verts, faces = mcubes.marching_cubes(volume, level)
    except ValueError as exc:  # pragma: no cover - mcubes raises ValueError on degenerate fields
        raise RuntimeError(f"Marching cubes failed for {out_path}: {exc}") from exc

    # Normalize vertices to [-0.5, 0.5] for compatibility with existing utilities.
    n = volume.shape[0]
    verts = verts / n - 0.5

    mesh = trimesh.Trimesh(vertices=verts, faces=faces, process=False)
    mesh.export(out_path)


def _prepare_options() -> SDFusionOpt:
    """Instantiate and populate an SDFusion option object."""

    opt = SDFusionOpt(gpu_ids=0)
    opt.gpu_ids = GPU_IDS
    opt.device = DEVICE_STR
    opt.trunc_thres = TRUNC_THRESHOLD

    opt.init_dset_args(
        dataroot=DATAROOT,
        dataset_mode="snet",
        cat=CATEGORY,
        res=RESOLUTION,
        cached_dir=None,
    )

    opt.init_model_args(
        ckpt_path=SDFUSION_CKPT,
        vq_ckpt_path=VQVAE_CKPT,
    )

    return opt


def main() -> None:
    torch.set_grad_enabled(False)

    opt = _prepare_options()

    # Create model and dataset ------------------------------------------------
    model = create_model(opt)
    model.eval()
    if hasattr(model, "df"):
        model.df.eval()
    if hasattr(model, "vqvae"):
        model.vqvae.eval()

    _, test_dataset = CreateDataset(opt)
    sample = test_dataset[SAMPLE_INDEX]

    sdf = sample["sdf"].unsqueeze(0).to(DEVICE_STR)
    sdf = torch.clamp(sdf, min=-TRUNC_THRESHOLD, max=TRUNC_THRESHOLD)

    # Run shape completion ----------------------------------------------------
    # Encode into the latent space -------------------------------------------------
    z = model.vqvae(sdf, forward_no_quant=True, encode_only=True)
    if model.latent_scaling != 1.0:
        z = z * model.latent_scaling

    partial_info = get_partial_shape(sdf, xyz_dict=XYZ_DICT, z=z)
    z_mask = partial_info["z_mask"].to(z.device)

    expected_spatial = model.z_shape[1:]
    if tuple(z_mask.shape[-3:]) != expected_spatial:
        z_mask = F.interpolate(z_mask.float(), size=expected_spatial, mode="nearest")
    else:
        z_mask = z_mask.float()

    z_mask = z_mask.to(z.dtype)

    ddim_sampler = DDIMSampler(model)
    ddim_steps = DDIM_STEPS or model.ddim_steps
    guidance_scale = UNCOND_SCALE if UNCOND_SCALE is not None else model.scale

    samples, _ = ddim_sampler.sample(
        S=ddim_steps,
        batch_size=1,
        shape=model.z_shape,
        conditioning=None,
        verbose=False,
        x0=z,
        mask=z_mask,
        unconditional_guidance_scale=guidance_scale,
        eta=DDIM_ETA,
    )

    completed = model.decode_latents(samples)

    completed = torch.clamp(completed, min=-TRUNC_THRESHOLD, max=TRUNC_THRESHOLD)

    model.x_part = partial_info["shape_part"]
    model.x_missing = partial_info["shape_missing"]
    partial = torch.clamp(model.x_part, min=-TRUNC_THRESHOLD, max=TRUNC_THRESHOLD)
    ground_truth = sdf

    model.gen_df = completed

    # Export meshes -----------------------------------------------------------
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    _export_sdf_as_ply(ground_truth[0], OUTPUT_DIR / "sample_ground_truth.ply")
    _export_sdf_as_ply(partial[0], OUTPUT_DIR / "sample_partial.ply")
    _export_sdf_as_ply(completed[0], OUTPUT_DIR / "sample_completion.ply")

    print(f"Meshes written to: {OUTPUT_DIR.resolve()}")


if __name__ == "__main__":
    main()

