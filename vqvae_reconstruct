#!/usr/bin/env python3
"""Utility script to inspect VQVAE reconstructions on ShapeNet SDF samples."""

import argparse
from pathlib import Path
from types import SimpleNamespace

import torch
from omegaconf import OmegaConf
import trimesh

from datasets.snet_dataset import ShapeNetDataset
from models.networks.vqvae_networks.network import VQVAE
from utils.util_3d import sdf_to_mesh


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Reconstruct a ShapeNet sample with a trained VQVAE")
    parser.add_argument("--vq-config", required=True, help="Path to the VQVAE config yaml used for training")
    parser.add_argument("--checkpoint", required=True, help="Path to the trained VQVAE checkpoint (.pth)")
    parser.add_argument("--dataroot", required=True, help="Root directory that contains the ShapeNet SDF_v1 data")
    parser.add_argument("--cat", default="all", help="Category name defined in info-shapenet.json")
    parser.add_argument("--phase", default="train", choices=["train", "test"], help="Dataset split to sample from")
    parser.add_argument("--resolution", type=int, default=64, help="SDF resolution used during preprocessing")
    parser.add_argument("--index", type=int, default=0, help="Index of the sample to reconstruct")
    parser.add_argument("--output-dir", default="outputs/vqvae_recon", help="Directory to store the exported meshes")
    parser.add_argument(
        "--trunc-thres",
        type=float,
        default=0.2,
        help=(
            "Truncation threshold applied to the SDF values. "
            "Match the value that was used during VQVAE training (default 0.2)."
        ),
    )
    parser.add_argument("--iso-level", type=float, default=0.02, help="Iso-surface level for marching cubes")
    parser.add_argument(
        "--max-dataset-size",
        type=int,
        default=2147483648,
        help="Optional cap on dataset size (defaults to the repository training option)",
    )
    parser.add_argument("--device", default="cuda", choices=["cuda", "cpu"], help="Preferred device for inference")
    return parser.parse_args()


def build_dataset(args: argparse.Namespace) -> ShapeNetDataset:
    opt = SimpleNamespace(
        dataroot=args.dataroot,
        max_dataset_size=args.max_dataset_size,
        trunc_thres=args.trunc_thres,
    )
    dataset = ShapeNetDataset()
    dataset.initialize(opt, phase=args.phase, cat=args.cat, res=args.resolution)
    return dataset


def load_vqvae(args: argparse.Namespace, device: torch.device) -> VQVAE:
    config = OmegaConf.load(args.vq_config)
    model_cfg = config.model.params
    vqvae = VQVAE(model_cfg.ddconfig, model_cfg.n_embed, model_cfg.embed_dim)
    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    state_dict = checkpoint["vqvae"] if "vqvae" in checkpoint else checkpoint
    vqvae.load_state_dict(state_dict)
    vqvae.to(device)
    vqvae.eval()
    return vqvae


def sdf_to_trimesh(meshes: "Meshes", output_path: Path) -> None:
    if meshes is None:
        raise RuntimeError("Failed to extract a mesh from the supplied SDF")
    verts_list = meshes.verts_list()
    faces_list = meshes.faces_list()
    if not verts_list:
        raise RuntimeError("Mesh extraction returned an empty vertex list")
    mesh = trimesh.Trimesh(
        vertices=verts_list[0].detach().cpu().numpy(),
        faces=faces_list[0].detach().cpu().numpy(),
        process=False,
    )
    mesh.export(output_path)


def main() -> None:
    args = parse_args()
    device = torch.device(args.device if args.device == "cpu" or torch.cuda.is_available() else "cpu")
    if args.device == "cuda" and device.type != "cuda":
        print("CUDA is not available, falling back to CPU inference.")

    dataset = build_dataset(args)
    if len(dataset) == 0:
        raise RuntimeError("Dataset is empty – please check dataroot, category, and split settings")
    if args.index >= len(dataset):
        raise IndexError(f"Requested index {args.index} but dataset only has {len(dataset)} samples")

    sample = dataset[args.index]
    print(f"Loaded sample from {sample['path']}")
    sdf = sample["sdf"].unsqueeze(0).to(device)  # (1, 1, res, res, res)
    if args.trunc_thres > 0:
        sdf = torch.clamp(sdf, min=-args.trunc_thres, max=args.trunc_thres)

    vqvae = load_vqvae(args, device)
    with torch.no_grad():
        recon, _ = vqvae(sdf, verbose=False)
        if args.trunc_thres > 0:
            recon = torch.clamp(recon, min=-args.trunc_thres, max=args.trunc_thres)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    input_mesh_path = output_dir / "input_mesh.ply"
    recon_mesh_path = output_dir / "reconstruction_mesh.ply"

    input_mesh = sdf_to_mesh(sdf, level=args.iso_level, render_all=True)
    recon_mesh = sdf_to_mesh(recon, level=args.iso_level, render_all=True)

    sdf_to_trimesh(input_mesh, input_mesh_path)
    sdf_to_trimesh(recon_mesh, recon_mesh_path)

    print(f"Saved input mesh to {input_mesh_path}")
    print(f"Saved reconstructed mesh to {recon_mesh_path}")


if __name__ == "__main__":
    main()
