from .minari_dataset import MinariDataset
from .ogbench_dataset import OGBenchDataset

# MINARI

PenHumanV2 = MinariDataset.from_name("D4RL/pen/human-v2")
BabyAIPickupOptimalFull = MinariDataset.from_name("minigrid/BabyAI-Pickup/optimal-fullobs-v0")

# OGBENCH

AntMazeMediumNavigateV0 = OGBenchDataset.from_name('antmaze-medium-navigate-v0')
