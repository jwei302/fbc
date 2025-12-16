"""
Post-process LIBERO rollouts data using rollouts_log.json.
Generates summary statistics without copying data.
"""

import dataclasses
import json
import logging
import pathlib

import tyro


@dataclasses.dataclass
class Args:
    """Arguments for postprocessing LIBERO rollouts and generating statistics."""

    rollouts_log: str = "runs/data_full/rollouts_log.json"  # path to the rollouts log file
    output_dir: str = "runs/data_full/datasets"  # directory where results will be saved


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


def save_results(
    output_dir: pathlib.Path,
    rollouts_log_path: pathlib.Path,
    total_episodes: int,
    successful_count: int,
    successful_episodes: list,
    task_stats: dict,
) -> tuple[pathlib.Path, pathlib.Path]:
    """Save postprocessing results to JSON files."""
    # Create episode mapping from old indices to new indices
    episode_mapping = {}
    for new_idx, episode_info in enumerate(successful_episodes):
        old_episode_idx = episode_info["episode_index"]
        episode_mapping[old_episode_idx] = new_idx

    results = {
        "rollouts_log": str(rollouts_log_path),
        "total_original_episodes": total_episodes,
        "successful_episodes": successful_count,
        "success_rate": successful_count / total_episodes if total_episodes > 0 else 0.0,
        "episodes_removed": total_episodes - successful_count,
        "episode_mapping": {str(k): v for k, v in episode_mapping.items()},
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    
    results_path = output_dir / "postprocess_results.json"
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

    per_task_path = output_dir / "postprocess_results_per_task.json"
    with open(per_task_path, "w") as f:
        json.dump(per_task_results, f, indent=2)

    return results_path, per_task_path


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    args = tyro.cli(Args)

    logging.info("Starting postprocessing...")

    rollouts_log_path = pathlib.Path(args.rollouts_log).resolve()
    output_dir = pathlib.Path(args.output_dir).resolve()

    if not rollouts_log_path.exists():
        raise FileNotFoundError(f"Rollouts log not found: {rollouts_log_path}")

    with open(rollouts_log_path) as f:
        rollouts_log = json.load(f)

    successful_episodes, total_episodes, successful_count, task_stats = collect_successful_episodes(
        rollouts_log
    )
    logging.info(
        f"Total episodes: {total_episodes}, Successful: {successful_count} "
        f"({successful_count / total_episodes * 100:.1f}%)"
    )

    results_path, per_task_path = save_results(
        output_dir,
        rollouts_log_path,
        total_episodes,
        successful_count,
        successful_episodes,
        task_stats,
    )

    logging.info(f"Done! Analyzed {total_episodes} episodes.")
    logging.info(f"Results saved to: {results_path}")
    logging.info(f"Per-task results saved to: {per_task_path}")


if __name__ == "__main__":
    main()

