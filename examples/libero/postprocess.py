#!/usr/bin/env python3
"""
Post-process LIBERO rollouts data using rollouts_log.json.
Creates a final fine-tuning dataset by taking only the successful episodes.
"""

import json
import shutil
import pathlib
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset


def main():
    print("Starting postprocessing...")
    
    # Paths
    base_dir = pathlib.Path(__file__).parent / "data"
    rollouts_log_path = base_dir / "rollouts_log.json"
    dataset_dir = base_dir / "datasets" / "libero_rollouts"
    output_dir = base_dir / "datasets" / "libero_rollouts_success"
    
    # Load the rollouts log
    with open(rollouts_log_path, 'r') as f:
        rollouts_log = json.load(f)
    
    # Collect all successful episode indices
    successful_episodes = []
    total_episodes = 0
    successful_count = 0
    task_stats = {}
    
    for task in rollouts_log:
        task_id = task['task_id']
        task_desc = task['task_description']
        task_successes = 0
        task_total = 0
        
        for run in task['runs']:
            total_episodes += 1
            task_total += 1
            if run['success'] == 1:
                successful_episodes.append({
                    'episode_index': run['dataset_episode_index'],
                    'task_id': task_id,
                    'task_description': task_desc,
                })
                successful_count += 1
                task_successes += 1
        
        task_stats[task_id] = {
            'description': task_desc,
            'successes': task_successes,
            'total': task_total,
        }
    
    print(f"Total episodes: {total_episodes}, Successful: {successful_count} ({successful_count/total_episodes*100:.1f}%)")
    
    # Sort by episode index to maintain order
    successful_episodes.sort(key=lambda x: x['episode_index'])
    
    # Load the original dataset
    original_dataset = LeRobotDataset(str(dataset_dir))
    
    # Remove output directory if it exists
    if output_dir.exists():
        shutil.rmtree(output_dir)
    
    # Create new dataset with only successful episodes
    # LeRobot expects repo_id to be used with a default root or local path
    new_dataset = LeRobotDataset.create(
        repo_id=str(output_dir),
        robot_type=original_dataset.meta.robot_type,
        fps=original_dataset.fps,
        features=original_dataset.features,
        image_writer_threads=10,
        image_writer_processes=5,
    )
    
    # Copy successful episodes
    episode_mapping = {}
    
    for new_idx, episode_info in enumerate(successful_episodes):
        old_episode_idx = episode_info['episode_index']
        
        # Get all frames from this episode
        episode_data = []
        for idx in range(len(original_dataset)):
            frame = original_dataset[idx]
            if frame['episode_index'].item() == old_episode_idx:
                episode_data.append(frame)
        
        # Add frames to new dataset
        for frame in episode_data:
            new_dataset.add_frame({
                'image': frame['observation.images.image'],
                'wrist_image': frame['observation.images.wrist_image'],
                'state': frame['observation.state'],
                'actions': frame['action'],
                'task': episode_info['task_description'],
            })
        
        # Save episode
        new_dataset.save_episode()
        episode_mapping[old_episode_idx] = new_idx
    
    # Save metadata
    metadata = {
        'original_dataset': str(dataset_dir),
        'filtered_dataset': str(output_dir),
        'total_original_episodes': total_episodes,
        'successful_episodes': successful_count,
        'success_rate': successful_count / total_episodes,
        'episodes_removed': total_episodes - successful_count,
        'episode_mapping': episode_mapping,
        'task_statistics': task_stats,
    }
    
    metadata_path = output_dir.parent / "filtering_metadata.json"
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"Done! Created {output_dir.name} with {successful_count} episodes ({len(new_dataset)} frames)")
    print(f"Metadata saved to: {metadata_path.name}")


if __name__ == "__main__":
    main()
