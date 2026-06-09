"""Constants for K1 humanoid."""

from etils import epath

from mujoco_playground._src import mjx_env

ROOT_PATH = mjx_env.ROOT_PATH / "locomotion" / "k1"
KICK_FLAT_TERRAIN_XML = ROOT_PATH / "xmls" / "scene_mjx_kick_flat_terrain.xml"


def task_to_xml(task_name: str) -> epath.Path:
  return {
      "kick_flat_terrain": KICK_FLAT_TERRAIN_XML,
  }[task_name]


FEET_SITES = ["l_foot", "r_foot"]
LEFT_FEET_GEOMS = ["l_foot_collision"]
RIGHT_FEET_GEOMS = ["r_foot_collision"]
FEET_GEOMS = LEFT_FEET_GEOMS + RIGHT_FEET_GEOMS
FEET_POS_SENSOR = [f"{site}_pos" for site in FEET_SITES]

ROOT_BODY = "Trunk"

GRAVITY_SENSOR = "upvector"
GLOBAL_LINVEL_SENSOR = "global_linvel"
GLOBAL_ANGVEL_SENSOR = "global_angvel"
LOCAL_LINVEL_SENSOR = "local_linvel"
ACCELEROMETER_SENSOR = "accelerometer"
GYRO_SENSOR = "gyro"
