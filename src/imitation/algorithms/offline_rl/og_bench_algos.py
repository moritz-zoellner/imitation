import copy
import random
from typing import Any
from functools import partial
from dataclasses import dataclass

import flax
import jax
import jax.numpy as jnp
import numpy as np
# import ml_collections
# import optax

from .gciql import GCIQLAgent


@dataclass
class OGBenchBase:
    """Base algorithm for offline RL algorithms implemented in OGBench."""

    agent_class: Any

    def train_fn(self, *, num_timesteps, seed, env, progress_fn):

        # # Set up environment and dataset.
        # config = FLAGS.agent
        # env, train_dataset, val_dataset = make_env_and_datasets(FLAGS.env_name, frame_stack=config['frame_stack'])

        # dataset_class = {
        #     'GCDataset': GCDataset,
        #     'HGCDataset': HGCDataset,
        # }[config['dataset_class']]
        # train_dataset = dataset_class(Dataset.create(**train_dataset), config)
        # if val_dataset is not None:
        #     val_dataset = dataset_class(Dataset.create(**val_dataset), config)

        # Initialize agent.
        random.seed(seed)
        np.random.seed(seed)

        # example_batch = train_dataset.sample(1)
        # if config['discrete']:
        #     # Fill with the maximum action to let the agent know the action space size.
        #     example_batch['actions'] = np.full_like(example_batch['actions'], env.action_space.n - 1)

        # agent = self.agent_class.create(
        #     seed,
        #     example_batch['observations'],
        #     example_batch['actions'],
        #     config,
        # )

        # # Restore agent.
        # if FLAGS.restore_path is not None:
        #     agent = restore_agent(agent, FLAGS.restore_path, FLAGS.restore_epoch)

        # # Train agent.
        # train_logger = CsvLogger(os.path.join(FLAGS.save_dir, 'train.csv'))
        # eval_logger = CsvLogger(os.path.join(FLAGS.save_dir, 'eval.csv'))
        # first_time = time.time()
        # last_time = time.time()
        # for i in tqdm.tqdm(range(1, FLAGS.train_steps + 1), smoothing=0.1, dynamic_ncols=True):
        #     # Update agent.
        #     batch = train_dataset.sample(config['batch_size'])
        #     agent, update_info = agent.update(batch)

        #     # Log metrics.
        #     if i % FLAGS.log_interval == 0:
        #         train_metrics = {f'training/{k}': v for k, v in update_info.items()}
        #         if val_dataset is not None:
        #             val_batch = val_dataset.sample(config['batch_size'])
        #             _, val_info = agent.total_loss(val_batch, grad_params=None)
        #             train_metrics.update({f'validation/{k}': v for k, v in val_info.items()})
        #         train_metrics['time/epoch_time'] = (time.time() - last_time) / FLAGS.log_interval
        #         train_metrics['time/total_time'] = time.time() - first_time
        #         last_time = time.time()
        #         wandb.log(train_metrics, step=i)
        #         train_logger.log(train_metrics, step=i)

        #     # Evaluate agent.
        #     if i == 1 or i % FLAGS.eval_interval == 0:
        #         if FLAGS.eval_on_cpu:
        #             eval_agent = jax.device_put(agent, device=jax.devices('cpu')[0])
        #         else:
        #             eval_agent = agent
        #         renders = []
        #         eval_metrics = {}
        #         overall_metrics = defaultdict(list)
        #         task_infos = env.unwrapped.task_infos if hasattr(env.unwrapped, 'task_infos') else env.task_infos
        #         num_tasks = FLAGS.eval_tasks if FLAGS.eval_tasks is not None else len(task_infos)
        #         for task_id in tqdm.trange(1, num_tasks + 1):
        #             task_name = task_infos[task_id - 1]['task_name']
        #             eval_info, trajs, cur_renders = evaluate(
        #                 agent=eval_agent,
        #                 env=env,
        #                 task_id=task_id,
        #                 config=config,
        #                 num_eval_episodes=FLAGS.eval_episodes,
        #                 num_video_episodes=FLAGS.video_episodes,
        #                 video_frame_skip=FLAGS.video_frame_skip,
        #                 eval_temperature=FLAGS.eval_temperature,
        #                 eval_gaussian=FLAGS.eval_gaussian,
        #             )
        #             renders.extend(cur_renders)
        #             metric_names = ['success']
        #             eval_metrics.update(
        #                 {f'evaluation/{task_name}_{k}': v for k, v in eval_info.items() if k in metric_names}
        #             )
        #             for k, v in eval_info.items():
        #                 if k in metric_names:
        #                     overall_metrics[k].append(v)
        #         for k, v in overall_metrics.items():
        #             eval_metrics[f'evaluation/overall_{k}'] = np.mean(v)

        #         if FLAGS.video_episodes > 0:
        #             video = get_wandb_video(renders=renders, n_cols=num_tasks)
        #             eval_metrics['video'] = video

        #         wandb.log(eval_metrics, step=i)
        #         eval_logger.log(eval_metrics, step=i)

        #     # Save agent.
        #     if i % FLAGS.save_interval == 0:
        #         save_agent(agent, FLAGS.save_dir, i)

        # train_logger.close()
        # eval_logger.close()


        return None, None, None


GCIQL = partial(OGBenchBase, agent_class=GCIQLAgent)
