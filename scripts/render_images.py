"""Render a novel view from 2–4 images of the same scene."""

import argparse
import importlib.util
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
DVLT_REF = "git+https://github.com/nv-tlabs/dvlt.git@f59bb18259ab446a7585c4f47731a77cfdefa426"


def ensure_dvlt():
    if importlib.util.find_spec("dvlt") is None:
        subprocess.run([sys.executable, "-m", "pip", "install", "--no-deps", DVLT_REF], check=True)


def load_image(path: Path, size: int):
    import numpy as np
    import torch
    from PIL import Image

    image = Image.open(path).convert("RGB")
    width, height = image.size
    side = min(width, height)
    left, top = (width - side) // 2, (height - side) // 2
    image = image.crop((left, top, left + side, top + side)).resize(
        (size, size), Image.Resampling.LANCZOS
    )
    array = np.asarray(image, dtype=np.float32) / 255.0
    return torch.from_numpy(array).permute(2, 0, 1)


def interpolate_camera_path(cameras, frame_count: int):
    import numpy as np
    import torch
    from scipy.spatial.transform import Rotation, Slerp

    c2w = cameras.extrinsics.float()
    intrinsics = cameras.intrinsics.float()
    start_rotation = c2w[0, 0, :3, :3].detach().cpu().numpy()
    end_rotation = c2w[0, -1, :3, :3].detach().cpu().numpy()
    times = np.linspace(0.0, 1.0, frame_count, dtype=np.float64)
    rotations = Slerp(
        [0.0, 1.0], Rotation.from_matrix(np.stack([start_rotation, end_rotation]))
    )(times).as_matrix()
    rotations = torch.as_tensor(rotations, device=c2w.device, dtype=c2w.dtype)
    amount = torch.as_tensor(times, device=c2w.device, dtype=c2w.dtype)
    translations = torch.lerp(
        c2w[:, 0, :3, 3], c2w[:, -1, :3, 3], amount[:, None]
    )
    path = torch.eye(4, device=c2w.device, dtype=c2w.dtype)[None].repeat(frame_count, 1, 1)
    path[:, :3, :3] = rotations
    path[:, :3, 3] = translations
    path = path[None]
    k = torch.lerp(
        intrinsics[:, :1], intrinsics[:, -1:], amount[None, :, None, None]
    )
    return path, k


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("images", nargs="+", type=Path, help="2–4 images of the same scene")
    parser.add_argument("--output", type=Path, default=Path("novel_view.mp4"))
    parser.add_argument("--checkpoint", default="hf://Knowing/RLSplat/re10k-24view/model.safetensors")
    parser.add_argument(
        "--budget-per-image", type=int, default=4096,
        help="Gaussian budget per input image (default: 4096)",
    )
    parser.add_argument("--frames", type=int, default=24, help="number of video frames")
    parser.add_argument("--fps", type=int, default=12, help="output video frame rate")
    parser.add_argument("--render-batch-size", type=int, default=4)
    parser.add_argument("--input-size", type=int, default=504)
    parser.add_argument("--render-size", type=int, default=384)
    args = parser.parse_args()

    import numpy as np
    import torch
    from hydra import compose, initialize_config_dir
    from omegaconf import OmegaConf
    from PIL import Image

    if not 2 <= len(args.images) <= 4:
        parser.error("provide 2–4 images of the same scene")
    if args.budget_per_image < 1:
        parser.error("--budget-per-image must be positive")
    if args.frames < 2 or args.fps < 1 or args.render_batch_size < 1:
        parser.error("--frames must be at least 2; --fps and --render-batch-size must be positive")
    for path in args.images:
        if not path.is_file():
            parser.error(f"image not found: {path}")
    if not torch.cuda.is_available():
        parser.error("CUDA is required for RLSplat inference and rendering")

    ensure_dvlt()
    from src.config import load_typed_root_config
    from src.evaluation.checkpoint import load_evaluation_checkpoint
    from src.global_cfg import set_cfg
    from src.model.decoder import get_decoder
    from src.model.encoder import get_encoder

    with initialize_config_dir(version_base=None, config_dir=str(ROOT / "config")):
        cfg_dict = compose(config_name="main")
    OmegaConf.update(cfg_dict, "checkpointing.load", args.checkpoint, force_add=True)
    cfg = load_typed_root_config(cfg_dict)
    set_cfg(cfg_dict)

    encoder, _ = get_encoder(cfg.model.encoder)
    model = torch.nn.Module()
    model.add_module("encoder", encoder)
    model.add_module("decoder", get_decoder(cfg.model.decoder))
    load_evaluation_checkpoint(model, args.checkpoint)
    model.eval().requires_grad_(False).to("cuda")

    images = torch.stack([load_image(path, args.input_size) for path in args.images])
    context = {"image": ((images - 0.5) / 0.5)[None].to("cuda")}
    with torch.inference_mode():
        gaussians, cameras = model.encoder.forward_with_cameras(
            context,
            gaussian_budget_per_batch=args.budget_per_image * len(args.images),
        )
        if cameras is None:
            raise RuntimeError("The checkpoint did not return predicted cameras")
        extrinsics, intrinsics = interpolate_camera_path(cameras, args.frames)
        frames = []
        for start in range(0, args.frames, args.render_batch_size):
            stop = min(start + args.render_batch_size, args.frames)
            count = stop - start
            near = torch.full((1, count), 0.1, device="cuda")
            far = torch.full((1, count), 100.0, device="cuda")
            rendered = model.decoder(
                gaussians,
                extrinsics[:, start:stop],
                intrinsics[:, start:stop],
                near,
                far,
                (args.render_size, args.render_size),
            ).color[0]
            frames.extend(
                (rendered.clamp(0, 1).permute(0, 2, 3, 1).cpu().numpy() * 255)
                .astype(np.uint8)
            )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    import imageio.v3 as iio

    iio.imwrite(args.output, np.stack(frames), fps=args.fps)
    print(f"Saved {len(frames)}-frame video to {args.output}")


if __name__ == "__main__":
    main()
