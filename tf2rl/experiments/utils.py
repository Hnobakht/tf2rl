"""Compatibility exports for trajectory helpers and optional GIF rendering."""
from tf2rl.experiments.demonstrations import (
    save_path, get_filenames, restore_latest_n_traj,
    load_trajectories, load_trajectories_with_logps,
)


def frames_to_gif(frames, prefix, save_dir, interval=50, fps=30):
    """Save a sequence of image frames as an animated GIF."""
    from pathlib import Path
    from PIL import Image
    if not frames:
        raise ValueError("No frames to export")
    images = [Image.fromarray(frame) for frame in frames]
    images[0].save(Path(save_dir) / f"{prefix}.gif", save_all=True,
                   append_images=images[1:], duration=1000 / fps, loop=0)
