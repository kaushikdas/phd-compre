import os
import sys
import numpy as np
from PIL import Image
import math
import pybullet as p

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, project_root)

from simulation.sim_env import SimulationEnvironment
from inference.omnivla_bridge import OmniVLAEdgeBridge

# Create env and load model
env = SimulationEnvironment(gui=False)
bridge = OmniVLAEdgeBridge(model_weights_dir=os.path.join(project_root, "omnivla-edge"))

# 1. TEST CASE A: No Obstacle in front
print("\n" + "="*50)
print(" TEST CASE A: No Obstacle, Target straight ahead (2.0m)")
print("="*50)

# Position red ball 2.0m straight ahead
p.resetBasePositionAndOrientation(env.objects["red_ball"]["id"], [2.0, 0.0, 0.15], p.getQuaternionFromEuler([0, 0, 0]))
env.objects["red_ball"]["position"] = [2.0, 0.0, 0.15]

# Reset context and predict
bridge.reset_context()
cam_img_a = env.get_camera_image()
robot_pos = env.get_robot_position()
robot_yaw = env.get_robot_orientation()

result_a = bridge.predict(
    camera_image=cam_img_a,
    language_command="go to red ball",
    robot_position=robot_pos[:2],
    robot_yaw=robot_yaw,
    target_position=np.array([2.0, 0.0])
)

print("No Obstacle Waypoints:")
for i, wp in enumerate(result_a["waypoints"]):
    dx = wp[0] * 0.1
    dy = wp[1] * 0.1
    print(f"  wp[{i}]: dx={dx:+.3f}m (fwd), dy={dy:+.3f}m (lat)")

print(f"Calculated Velocity: linear={result_a['linear_vel']:.3f} m/s, angular={result_a['angular_vel']:.3f} rad/s")

# 2. TEST CASE B: Large Obstacle placed at (1.0, 0.0)
print("\n" + "="*50)
print(" TEST CASE B: Large Obstacle directly on path to target")
print("="*50)

# Create a large visual and collision box representing a stone obstacle at (1.0, 0.0)
obs_visual = p.createVisualShape(p.GEOM_BOX, halfExtents=[0.2, 0.2, 0.3], rgbaColor=[0.5, 0.5, 0.5, 1.0])
obs_collision = p.createCollisionShape(p.GEOM_BOX, halfExtents=[0.2, 0.2, 0.3])
obs_id = p.createMultiBody(baseMass=0.0, baseCollisionShapeIndex=obs_collision, baseVisualShapeIndex=obs_visual, basePosition=[1.0, 0.0, 0.3])

# Reset context and predict again with the obstacle present
bridge.reset_context()
cam_img_b = env.get_camera_image()
# Save debug images to verify the obstacle is visible
os.makedirs(os.path.join(project_root, "output"), exist_ok=True)
cam_img_a.save(os.path.join(project_root, "output", "test_obs_a_clear.png"))
cam_img_b.save(os.path.join(project_root, "output", "test_obs_b_blocked.png"))

result_b = bridge.predict(
    camera_image=cam_img_b,
    language_command="go to red ball",
    robot_position=robot_pos[:2],
    robot_yaw=robot_yaw,
    target_position=np.array([2.0, 0.0])
)

print("Obstacle Waypoints:")
for i, wp in enumerate(result_b["waypoints"]):
    dx = wp[0] * 0.1
    dy = wp[1] * 0.1
    print(f"  wp[{i}]: dx={dx:+.3f}m (fwd), dy={dy:+.3f}m (lat)")

print(f"Calculated Velocity: linear={result_b['linear_vel']:.3f} m/s, angular={result_b['angular_vel']:.3f} rad/s")

# Cleanup
env.close()
