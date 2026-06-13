"""Quick motor direction test"""
import os, math, sys
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from simulation.sim_env import SimulationEnvironment

print("TEST 1: Motor Direction Verification")

# Test A: Forward
env = SimulationEnvironment(gui=False)
env.apply_velocity(linear_vel=0.3, angular_vel=0.0)
for _ in range(240): env.step_simulation()
pos = env.get_robot_position()
yaw = env.get_robot_orientation()
print(f"  Forward: pos=({pos[0]:+.3f}, {pos[1]:+.3f}), yaw={math.degrees(yaw):+.1f} -> {'PASS' if pos[0] > 0.05 else 'FAIL'}")
env.close()

# Test B: Turn left
env = SimulationEnvironment(gui=False)
env.apply_velocity(linear_vel=0.0, angular_vel=0.5)
for _ in range(480): env.step_simulation()
yaw = env.get_robot_orientation()
print(f"  Left turn: yaw={math.degrees(yaw):+.1f} -> {'PASS' if yaw > 0.05 else 'FAIL - INVERTED!'}")
env.close()

# Test C: Turn right
env = SimulationEnvironment(gui=False)
env.apply_velocity(linear_vel=0.0, angular_vel=-0.5)
for _ in range(480): env.step_simulation()
yaw = env.get_robot_orientation()
print(f"  Right turn: yaw={math.degrees(yaw):+.1f} -> {'PASS' if yaw < -0.05 else 'FAIL - INVERTED!'}")
env.close()

# Test D: Forward + left
env = SimulationEnvironment(gui=False)
env.apply_velocity(linear_vel=0.3, angular_vel=0.3)
for _ in range(480): env.step_simulation()
pos = env.get_robot_position()
yaw = env.get_robot_orientation()
print(f"  Fwd+Left: pos=({pos[0]:+.3f}, {pos[1]:+.3f}), yaw={math.degrees(yaw):+.1f} -> {'PASS' if pos[1] > 0.01 else 'FAIL'}")
env.close()
