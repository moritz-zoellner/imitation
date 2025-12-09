from typing import Optional
from dataclasses import dataclass

from .gymnax_env import GymnaxEnv
from .ogbench_env import OGBenchEnv


# GYMNAX

four_rooms = GymnaxEnv.from_name("FourRooms-misc")
space_invaders = GymnaxEnv.from_name("SpaceInvaders-MinAtar")
reacher = GymnaxEnv.from_name("Reacher-misc")
breakout = GymnaxEnv.from_name("Breakout-MinAtar")

# BRAX


# OGBench

PointmazeLarge = OGBenchEnv.from_name("pointmaze-large")
