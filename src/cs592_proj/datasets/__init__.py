from .minari_dataset import MinariDataset, MinigridWrapper, AtariWrapper
from .ogbench_dataset import OGBenchDataset
from .custom_dataset import CustomDataset

# MINARI

PenHumanV2 = MinariDataset.from_name("D4RL/pen/human-v2")
BabyAIPickupOptimalFull = MinariDataset.from_name("minigrid/BabyAI-Pickup/optimal-fullobs-v0",
                                                  wrappers=[MinigridWrapper])
PongExpertV0 = MinariDataset.from_name("atari/pong/expert-v0", wrappers=[AtariWrapper])

# OGBENCH

AntMazeMediumNavigateV0 = OGBenchDataset.from_name('antmaze-medium-navigate-v0')

# CUSTOM
#SpaceInvadersV0 = CustomDataset.from_resource_path('custom_datasets/dqn-space-invaders.npz', env_name="space_invaders")
#SpaceInvadersV1 = CustomDataset.from_resource_path('custom_datasets/dqn-space-invaders-v1.npz', env_name="space_invaders")




#suboptimal_dataset = CustomDataset.from_resource_path('custom_datasets/policy_params.npz', env_name="space_invaders")