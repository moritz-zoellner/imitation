# CS592 Imitation Learning

## Train

    python -m cs592_proj.scripts.mlflow_train --algo DQN --env space_invaders --num_timesteps 50000000 --seed 4

## Generate Dataset

    python -m cs592_proj.scripts.mlflow_generate_dataset --id b355a8cc377c46318f3a3e0639e040ba --policies best --output "./custom-datasets/dqn-space-invaders.npz"

## Visualize Policy

    python -m cs592_proj.scripts.mlflow_visualize --id b355a8cc377c46318f3a3e0639e040ba
