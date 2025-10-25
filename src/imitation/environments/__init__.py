from typing import Optional
from dataclasses import dataclass

from .gymnax_env import GymnaxEnv
    

four_rooms = GymnaxEnv.from_name("FourRooms-misc")
space_invaders = GymnaxEnv.from_name("SpaceInvaders-MinAtar")
