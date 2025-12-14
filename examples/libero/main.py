import collections
import dataclasses
import json
import logging
import math
import os
import pathlib
import shutil

import imageio
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
from libero.libero import benchmark
from libero.libero import get_libero_path
from libero.libero.envs import OffScreenRenderEnv
import numpy as np
from openpi_client import image_tools
from openpi_client import websocket_client_policy as _websocket_client_policy
import tqdm
import tyro

LIBERO_DUMMY_ACTION = [0.0] * 6 + [-1.0]
LIBERO_ENV_RESOLUTION = 256  # resolution used to render training data


@dataclasses.dataclass
class Args:
    #################################################################################################################
    # Model server parameters
    #################################################################################################################
    host: str = "0.0.0.0"
    port: int = 8000
    resize_size: int = 224
    replan_steps: int = 5

    #################################################################################################################
    # LIBERO environment-specific parameters
    #################################################################################################################
    task_suite_name: str = (
        "libero_spatial"  # Task suite. Options: libero_spatial, libero_object, libero_goal, libero_10, libero_90
    )
    num_steps_wait: int = 10  # Number of steps to wait for objects to stabilize i n sim
    num_trials_per_task: int = 50  # Number of rollouts per task

    #################################################################################################################
    # LeRobot dataset parameters
    #################################################################################################################
    dataset_repo_id: str = "libero_rollouts"  # Name of the LeRobot dataset
    dataset_local_dir: str = "examples/libero/data"  # Local directory to save the dataset

    seed: int = 7  # Random Seed (for reproducibility)


def eval_libero(args: Args) -> None:
    # Set random seed
    np.random.seed(args.seed)

    # Initialize LIBERO task suite
    benchmark_dict = benchmark.get_benchmark_dict()
    task_suite = benchmark_dict[args.task_suite_name]()
    num_tasks_in_suite = task_suite.n_tasks
    logging.info(f"Task suite: {args.task_suite_name}")

    # Setup directory structure
    data_dir = pathlib.Path(args.dataset_local_dir).resolve()
    
    # Remove data directory if it exists to start fresh
    if data_dir.exists():
        shutil.rmtree(data_dir)
        logging.info(f"Removed existing data directory: {data_dir}")
    datasets_dir = data_dir / "datasets"
    videos_dir = data_dir / "videos"
    
    # Create videos directory
    videos_dir.mkdir(parents=True, exist_ok=True)
    
    # Override the HF_LEROBOT_HOME environment variable to save datasets in datasets/ subdirectory
    os.environ["HF_LEROBOT_HOME"] = str(datasets_dir)
    
    # Check if dataset exists
    dataset_repo_path = datasets_dir / args.dataset_repo_id
    if dataset_repo_path.exists():
        # Load existing dataset to continue adding episodes
        logging.info(f"Loading existing dataset from: {dataset_repo_path}")
        dataset = LeRobotDataset(args.dataset_repo_id, root=str(dataset_repo_path))
    else:
        # Create new dataset
        logging.info(f"Creating new dataset: {args.dataset_repo_id}")
        dataset = LeRobotDataset.create(
            repo_id=args.dataset_repo_id,
            root=str(dataset_repo_path),
            robot_type="panda",
            fps=10,
            features={
                "image": {
                    "dtype": "image",
                    "shape": (LIBERO_ENV_RESOLUTION, LIBERO_ENV_RESOLUTION, 3),
                    "names": ["height", "width", "channel"],
                },
                "wrist_image": {
                    "dtype": "image",
                    "shape": (LIBERO_ENV_RESOLUTION, LIBERO_ENV_RESOLUTION, 3),
                    "names": ["height", "width", "channel"],
                },
                "state": {
                    "dtype": "float32",
                    "shape": (8,),
                    "names": ["state"],
                },
                "actions": {
                    "dtype": "float32",
                    "shape": (7,),
                    "names": ["actions"],
                },
            },
            image_writer_threads=10,
            image_writer_processes=5,
        )
    
    logging.info(f"Dataset will be saved to: {dataset_repo_path}")

    if args.task_suite_name == "libero_spatial":
        max_steps = 220  # longest training demo has 193 steps
    elif args.task_suite_name == "libero_object":
        max_steps = 280  # longest training demo has 254 steps
    elif args.task_suite_name == "libero_goal":
        max_steps = 300  # longest training demo has 270 steps
    elif args.task_suite_name == "libero_10":
        max_steps = 520  # longest training demo has 505 steps
    elif args.task_suite_name == "libero_90":
        max_steps = 400  # longest training demo has 373 steps
    else:
        raise ValueError(f"Unknown task suite: {args.task_suite_name}")

    client = _websocket_client_policy.WebsocketClientPolicy(args.host, args.port)

    # Initialize or load the single log file
    log_file_path = data_dir / "rollouts_log.json"
    if log_file_path.exists():
        with open(log_file_path, "r") as f:
            all_logs = json.load(f)
    else:
        all_logs = []

    # Start evaluation
    total_episodes, total_successes = 0, 0
    for task_id in tqdm.tqdm(range(num_tasks_in_suite)):
        # Get task
        task = task_suite.get_task(task_id)

        # Get default LIBERO initial states
        initial_states = task_suite.get_task_init_states(task_id)

        # Initialize LIBERO environment and task description
        env, task_description = _get_libero_env(task, LIBERO_ENV_RESOLUTION, args.seed)

        # Initialize task logging structure
        task_log = {
            "task_id": task_id,
            "task_description": task_description,
            "runs": []
        }

        # Start episodes
        task_episodes, task_successes = 0, 0
        for episode_idx in tqdm.tqdm(range(args.num_trials_per_task)):
            logging.info(f"\nTask: {task_description}")

            # Reset environment
            env.reset()
            action_plan = collections.deque()

            # Set initial states
            obs = env.set_init_state(initial_states[episode_idx])

            # Setup
            t = 0
            replay_images = []

            logging.info(f"Starting episode {task_episodes+1}...")
            while t < max_steps + args.num_steps_wait:
                try:
                    # IMPORTANT: Do nothing for the first few timesteps because the simulator drops objects
                    # and we need to wait for them to fall
                    if t < args.num_steps_wait:
                        obs, reward, done, info = env.step(LIBERO_DUMMY_ACTION)
                        t += 1
                        continue

                    # Get preprocessed image
                    # IMPORTANT: rotate 180 degrees to match train preprocessing
                    img_raw = np.ascontiguousarray(obs["agentview_image"][::-1, ::-1])
                    wrist_img_raw = np.ascontiguousarray(obs["robot0_eye_in_hand_image"][::-1, ::-1])

                    # Convert to uint8 for dataset storage (LeRobot will handle compression)
                    img_raw_uint8 = image_tools.convert_to_uint8(img_raw)
                    wrist_img_raw_uint8 = image_tools.convert_to_uint8(wrist_img_raw)

                    # Resize for policy inference
                    img = image_tools.convert_to_uint8(
                        image_tools.resize_with_pad(img_raw, args.resize_size, args.resize_size)
                    )
                    wrist_img = image_tools.convert_to_uint8(
                        image_tools.resize_with_pad(wrist_img_raw, args.resize_size, args.resize_size)
                    )

                    # Save preprocessed image for replay video
                    replay_images.append(img)

                    if not action_plan:
                        # Finished executing previous action chunk -- compute new chunk
                        # Prepare observations dict
                        element = {
                            "observation/image": img,
                            "observation/wrist_image": wrist_img,
                            "observation/state": np.concatenate(
                                (
                                    obs["robot0_eef_pos"],
                                    _quat2axisangle(obs["robot0_eef_quat"]),
                                    obs["robot0_gripper_qpos"],
                                )
                            ),
                            "prompt": str(task_description),
                        }

                        # Query model to get action
                        action_chunk = client.infer(element)["actions"]
                        assert (
                            len(action_chunk) >= args.replan_steps
                        ), f"We want to replan every {args.replan_steps} steps, but policy only predicts {len(action_chunk)} steps."
                        action_plan.extend(action_chunk[: args.replan_steps])

                    action = action_plan.popleft()

                    # Capture state for dataset
                    current_state = np.concatenate(
                        (
                            obs["robot0_eef_pos"],
                            _quat2axisangle(obs["robot0_eef_quat"]),
                            obs["robot0_gripper_qpos"],
                        )
                    ).astype(np.float32)

                    # Execute action in environment
                    obs, reward, done, info = env.step(action.tolist())

                    # Record frame to dataset
                    dataset.add_frame(
                        {
                            "image": img_raw_uint8,
                            "wrist_image": wrist_img_raw_uint8,
                            "state": current_state,
                            "actions": action.astype(np.float32),
                            "task": task_description,
                        }
                    )

                    if done:
                        task_successes += 1
                        total_successes += 1
                        break
                    t += 1

                except Exception as e:
                    logging.error(f"Caught exception: {e}")
                    break

            # Get dataset episode index before saving
            dataset_episode_idx = dataset.meta.total_episodes
            
            # Save episode to dataset
            dataset.save_episode()

            task_episodes += 1
            total_episodes += 1

            # Save a replay video of the episode
            video_filename = f"task{task_id}_iter{episode_idx}.mp4"
            video_path = videos_dir / video_filename
            imageio.mimwrite(
                video_path,
                [np.asarray(x) for x in replay_images],
                fps=10,
            )
            
            # Get the path to the saved episode data
            episode_data_path = dataset.meta.get_data_file_path(dataset_episode_idx)
            
            # Add run info to task log
            task_log["runs"].append({
                "iteration": episode_idx,
                "success": 1 if done else 0,
                "path_video": f"videos/{video_filename}",
                "dataset_episode_index": dataset_episode_idx,
                "episode_data_path": str(episode_data_path)
            })

            # Log current results
            logging.info(f"Success: {done}")
            logging.info(f"# episodes completed so far: {total_episodes}")
            logging.info(f"# successes: {total_successes} ({total_successes / total_episodes * 100:.1f}%)")

        # Append task log to the single log file
        all_logs.append(task_log)
        with open(log_file_path, "w") as f:
            json.dump(all_logs, f, indent=2)
        logging.info(f"Appended task {task_id} to {log_file_path}")

        # Log final results
        logging.info(f"Current task success rate: {float(task_successes) / float(task_episodes)}")
        logging.info(f"Current total success rate: {float(total_successes) / float(total_episodes)}")

    logging.info(f"Total success rate: {float(total_successes) / float(total_episodes)}")
    logging.info(f"Total episodes: {total_episodes}")


def _get_libero_env(task, resolution, seed):
    """Initializes and returns the LIBERO environment, along with the task description."""
    task_description = task.language
    task_bddl_file = pathlib.Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file
    env_args = {"bddl_file_name": task_bddl_file, "camera_heights": resolution, "camera_widths": resolution}
    env = OffScreenRenderEnv(**env_args)
    env.seed(seed)  # IMPORTANT: seed seems to affect object positions even when using fixed initial state
    return env, task_description


def _quat2axisangle(quat):
    """
    Copied from robosuite: https://github.com/ARISE-Initiative/robosuite/blob/eafb81f54ffc104f905ee48a16bb15f059176ad3/robosuite/utils/transform_utils.py#L490C1-L512C55
    """
    # clip quaternion
    if quat[3] > 1.0:
        quat[3] = 1.0
    elif quat[3] < -1.0:
        quat[3] = -1.0

    den = np.sqrt(1.0 - quat[3] * quat[3])
    if math.isclose(den, 0.0):
        # This is (close to) a zero degree rotation, immediately return
        return np.zeros(3)

    return (quat[:3] * 2.0 * math.acos(quat[3])) / den


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    tyro.cli(eval_libero)
