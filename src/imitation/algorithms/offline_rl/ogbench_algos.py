import json
import os
import copy
import random
import time
from typing import Any
from collections import defaultdict
from functools import partial
from dataclasses import dataclass

import tqdm
import flax
import jax
import jax.numpy as jnp
import numpy as np
# import ml_collections
# import optax

from .utils.datasets import Dataset, GCDataset, HGCDataset
from .utils.env_utils import make_env_and_datasets
from .utils.evaluation import evaluate
from .utils.flax_utils import restore_agent, save_agent
from .utils.log_utils import CsvLogger, get_exp_name, get_flag_dict, get_wandb_video, setup_wandb

from .gciql import GCIQLAgent


@dataclass
class OGBenchBase:
    """Base algorithm for offline RL algorithms implemented in OGBench."""

    agent_class: Any

    def train_fn(self, *, run_config, dataset, progress_fn, **_):

        agent_config = self.agent_class.get_config()

        dataset_class = {
            'GCDataset': GCDataset,
            'HGCDataset': HGCDataset,
        }[agent_config['dataset_class']]

        breakpoint()

        # TODO: Abstract
        train_dataset = dataset.train_dataset_impl
        val_dataset = dataset.val_dataset_impl

        train_dataset = dataset_class(Dataset.create(**train_dataset), agent_config)
        if val_dataset is not None:
            val_dataset = dataset_class(Dataset.create(**val_dataset), agent_config)

        # Initialize agent.
        random.seed(run_config.seed)
        np.random.seed(run_config.seed)

        example_batch = train_dataset.sample(1)
        if agent_config['discrete']:
            # Fill with the maximum action to let the agent know the action space size.
            example_batch['actions'] = np.full_like(example_batch['actions'], env.action_space.n - 1)

        agent = self.agent_class.create(
            run_config.seed,
            example_batch['observations'],
            example_batch['actions'],
            agent_config,
        )

        # # Restore agent.
        # if FLAGS.restore_path is not None:
        #     agent = restore_agent(agent, FLAGS.restore_path, FLAGS.restore_epoch)

        # Train agent.
        # train_logger = CsvLogger(os.path.join(FLAGS.save_dir, 'train.csv'))
        # eval_logger = CsvLogger(os.path.join(FLAGS.save_dir, 'eval.csv'))
        first_time = time.time()
        last_time = time.time()

        train_steps = run_config.num_timesteps // agent_config['batch_size']
        eval_interval = train_steps // run_config.num_evals

        assert run_config.num_evals == train_steps // eval_interval, "num_evals not guaranteed"

        env = dataset.get_eval_env()
        
        for i in tqdm.tqdm(range(train_steps), smoothing=0.1, dynamic_ncols=True):
            # Update agent.
            batch = train_dataset.sample(agent_config['batch_size'])
            agent, update_info = agent.update(batch)

            # Evaluate agent.
            if i % eval_interval == 0:
                train_metrics = {f'training/{k}': v for k, v in update_info.items()}
                if val_dataset is not None:
                    val_batch = val_dataset.sample(agent_config['batch_size'])
                    _, val_info = agent.total_loss(val_batch, grad_params=None)
                    train_metrics.update({f'validation/{k}': v for k, v in val_info.items()})
                # train_metrics['time/epoch_time'] = (time.time() - last_time) / FLAGS.log_interval
                train_metrics['time/total_time'] = time.time() - first_time
                last_time = time.time()

                if run_config.eval_on_cpu:
                    eval_agent = jax.device_put(agent, device=jax.devices('cpu')[0])
                else:
                    eval_agent = agent
                renders = []
                eval_metrics = {}
                overall_metrics = defaultdict(list)
                task_infos = env.unwrapped.task_infos if hasattr(env.unwrapped, 'task_infos') else env.task_infos
                num_tasks = 1 # len(task_infos)

                for task_id in range(1, num_tasks+1):
                    task_name = task_infos[task_id - 1]['task_name']
                    eval_info, trajs, cur_renders = evaluate(
                        agent=eval_agent,
                        env=env,
                        task_id=task_id,
                        config=agent_config,
                        num_eval_episodes=run_config.eval_episodes,
                        # num_video_episodes=run_config.video_episodes,
                        # video_frame_skip=run_config.video_frame_skip,
                        # eval_temperature=run_config.eval_temperature,
                        # eval_gaussian=run_config.eval_gaussian,
                    )
                    renders.extend(cur_renders)
                    metric_names = ['success']
                    eval_metrics.update(
                        {f'evaluation/{task_name}_{k}': v for k, v in eval_info.items() if k in metric_names}
                    )
                    for k, v in eval_info.items():
                        if k in metric_names:
                            overall_metrics[k].append(v)

                for k, v in overall_metrics.items():
                    eval_metrics[f'evaluation/overall_{k}'] = np.mean(v)

                # if FLAGS.video_episodes > 0:
                #     video = get_wandb_video(renders=renders, n_cols=num_tasks)
                #     eval_metrics['video'] = video

                num_steps = i * agent_config['batch_size']
                metrics = train_metrics | eval_metrics
                progress_fn(num_steps, metrics)

            # # Save agent.
            # if i % FLAGS.save_interval == 0:
            #     save_agent(agent, FLAGS.save_dir, i)

        return None, None, None


GCIQL = partial(OGBenchBase, agent_class=GCIQLAgent)
