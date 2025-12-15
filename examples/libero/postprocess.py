"""
Post-process LIBERO rollouts data using rollouts_log.json.
Creates a final fine-tuning dataset by taking only the successful episodes.
"""

import dataclasses
import io
import json
import logging
import os
import pathlib
import shutil

from PIL import Image
import pyarrow.parquet as pq
import tyro

from lerobot.common.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata


@dataclasses.dataclass
class Args:
    """Arguments for postprocessing LIBERO rollouts and filtering successful episodes."""

    rollouts_log: str = "examples/libero/data/rollouts_log.json" # path to the rollouts log file
    input_dataset: str = "examples/libero/data/datasets/libero_rollouts" # path to the input dataset
    output_dataset: str = "examples/libero/data/datasets/libero_rollouts_success" # path to the output dataset


def collect_successful_episodes(rollouts_log: list) -> tuple[list, int, int, dict]:
    """
    Collect all successful episode indices from the rollouts log.

    Returns:
        successful_episodes: List of dicts with episode info
        total_episodes: Total number of episodes
        successful_count: Number of successful episodes
        task_stats: Per-task statistics
    """
    successful_episodes = []
    total_episodes = 0
    successful_count = 0
    task_stats = {}

    for task in rollouts_log:
        task_id = task["task_id"]
        task_desc = task["task_description"]
        task_successes = 0
        task_total = 0

        for run in task["runs"]:
            total_episodes += 1
            task_total += 1
            if run["success"] == 1:
                successful_episodes.append({
                    "episode_index": run["dataset_episode_index"],
                    "task_id": task_id,
                    "task_description": task_desc,
                })
                successful_count += 1
                task_successes += 1

        task_stats[task_id] = {
            "description": task_desc,
            "successes": task_successes,
            "total": task_total,
        }

    successful_episodes.sort(key=lambda x: x["episode_index"])
    return successful_episodes, total_episodes, successful_count, task_stats


def load_image_from_bytes(image_data: dict) -> Image.Image:
    """Convert image data from parquet (dict with 'bytes' key) to PIL Image."""
    if isinstance(image_data, dict) and "bytes" in image_data:
        return Image.open(io.BytesIO(image_data["bytes"]))
    return image_data


def move_successful_episodes(
    input_dataset_dir: pathlib.Path,
    output_dataset_dir: pathlib.Path,
    successful_episodes: list,
) -> tuple[dict, int]:
    """
    Move successful episodes from input dataset to output dataset.

    Returns:
        episode_mapping: Mapping from old episode indices to new indices
        total_frames: Total number of frames copied
    """
    metadata = LeRobotDatasetMetadata(input_dataset_dir.name, root=input_dataset_dir)

    if output_dataset_dir.exists():
        shutil.rmtree(output_dataset_dir)

    new_dataset = LeRobotDataset.create(
        repo_id=output_dataset_dir.name,
        root=output_dataset_dir,
        robot_type=metadata.robot_type,
        fps=metadata.fps,
        features=metadata.features,
        image_writer_threads=10,
        image_writer_processes=5,
    )

    # Build set of successful episode indices
    successful_indices = {ep["episode_index"] for ep in successful_episodes}
    episode_to_task = {ep["episode_index"]: ep["task_description"] for ep in successful_episodes}

    # Read all parquet files and filter to successful episodes only
    data_dir = input_dataset_dir / "data"
    all_frames = []

    for parquet_file in sorted(data_dir.rglob("*.parquet")):
        table = pq.read_table(parquet_file)
        df = table.to_pandas()
        # Filter to only successful episodes
        mask = df["episode_index"].isin(successful_indices)
        if mask.any():
            all_frames.append(df[mask])

    # Concatenate and sort by episode_index, then frame_index
    import pandas as pd
    all_data = pd.concat(all_frames, ignore_index=True)
    all_data = all_data.sort_values(["episode_index", "frame_index"]).reset_index(drop=True)

    # Process episodes
    episode_mapping = {}
    total_frames = 0

    for new_idx, episode_info in enumerate(successful_episodes):
        old_episode_idx = episode_info["episode_index"]
        episode_data = all_data[all_data["episode_index"] == old_episode_idx]

        for _, row in episode_data.iterrows():
            new_dataset.add_frame({
                "image": load_image_from_bytes(row["image"]),
                "wrist_image": load_image_from_bytes(row["wrist_image"]),
                "state": row["state"],
                "actions": row["actions"],
                "task": episode_to_task[old_episode_idx],
            })
            total_frames += 1

        new_dataset.save_episode()
        episode_mapping[old_episode_idx] = new_idx

        if (new_idx + 1) % 5 == 0:
            logging.info(f"Processed {new_idx + 1}/{len(successful_episodes)} episodes")

    return episode_mapping, total_frames


def save_results(
    output_dataset_dir: pathlib.Path,
    input_dataset_dir: pathlib.Path,
    total_episodes: int,
    successful_count: int,
    episode_mapping: dict,
    task_stats: dict,
) -> tuple[pathlib.Path, pathlib.Path]:
    """Save postprocessing results to JSON files."""
    results = {
        "original_dataset": str(input_dataset_dir),
        "filtered_dataset": str(output_dataset_dir),
        "total_original_episodes": total_episodes,
        "successful_episodes": successful_count,
        "success_rate": successful_count / total_episodes,
        "episodes_removed": total_episodes - successful_count,
        "episode_mapping": {str(k): v for k, v in episode_mapping.items()},
    }

    results_path = output_dataset_dir.parent / "postprocess_results.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)

    per_task_results = [
        {
            "task_id": task_id,
            "task_description": stats["description"],
            "total_episodes": stats["total"],
            "successful_episodes": stats["successes"],
            "success_rate": stats["successes"] / stats["total"] if stats["total"] > 0 else 0.0,
        }
        for task_id, stats in task_stats.items()
    ]

    per_task_path = output_dataset_dir.parent / "postprocess_results_per_task.json"
    with open(per_task_path, "w") as f:
        json.dump(per_task_results, f, indent=2)

    return results_path, per_task_path


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    args = tyro.cli(Args)

    logging.info("Starting postprocessing...")

    rollouts_log_path = pathlib.Path(args.rollouts_log).resolve()
    input_dataset_dir = pathlib.Path(args.input_dataset).resolve()
    output_dataset_dir = pathlib.Path(args.output_dataset).resolve()

    os.environ["HF_LEROBOT_HOME"] = str(input_dataset_dir.parent)
    logging.info(f"Set HF_LEROBOT_HOME to: {input_dataset_dir.parent}")

    with open(rollouts_log_path) as f:
        rollouts_log = json.load(f)

    successful_episodes, total_episodes, successful_count, task_stats = collect_successful_episodes(
        rollouts_log
    )
    logging.info(
        f"Total episodes: {total_episodes}, Successful: {successful_count} "
        f"({successful_count / total_episodes * 100:.1f}%)"
    )

    episode_mapping, total_frames = move_successful_episodes(
        input_dataset_dir,
        output_dataset_dir,
        successful_episodes,
    )

    results_path, per_task_path = save_results(
        output_dataset_dir,
        input_dataset_dir,
        total_episodes,
        successful_count,
        episode_mapping,
        task_stats,
    )

    logging.info(
        f"Done! Created {output_dataset_dir.name} with {successful_count} episodes ({total_frames} frames)"
    )
    logging.info(f"Results saved to: {results_path.name}")
    logging.info(f"Per-task results saved to: {per_task_path.name}")


if __name__ == "__main__":
    main()
