"""
OmniVLA Demo — Navigate to Both Targets
=========================================
Runs two navigation tasks back-to-back in the GUI:
  1. "go to the red ball"
  2. "go to the green cube"

Usage:
  conda activate env_aiml
  python demo_both_targets.py
"""

import os
import sys
import math
import time
import numpy as np

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)

from simulation.sim_env import SimulationEnvironment
from inference.omnivla_bridge import OmniVLAEdgeBridge


def pre_rotate(env, target_pos, gui=True, threshold_deg=30):
    """Rotate robot in-place to face the target."""
    threshold = math.radians(threshold_deg)
    robot_pos = env.get_robot_position()
    robot_yaw = env.get_robot_orientation()
    delta = np.array(target_pos[:2]) - robot_pos[:2]
    desired_yaw = math.atan2(delta[1], delta[0])
    err = desired_yaw - robot_yaw
    while err > math.pi:  err -= 2 * math.pi
    while err < -math.pi: err += 2 * math.pi

    if abs(err) <= threshold:
        return  # already facing close enough

    print(f"    Rotating {math.degrees(err):+.0f}deg to face target...")
    for _ in range(500):
        if abs(err) <= threshold:
            break
        w = np.clip(1.5 * err, -0.6, 0.6)
        env.apply_velocity(0.0, w)
        if gui:
            for _ in range(24):
                env.step_simulation()
                time.sleep(env.time_step)
        else:
            env.step_simulation(num_steps=24)
        robot_yaw = env.get_robot_orientation()
        err = desired_yaw - robot_yaw
        while err > math.pi:  err -= 2 * math.pi
        while err < -math.pi: err += 2 * math.pi
    env.stop_robot()
    print(f"    Done. Residual error: {math.degrees(err):+.1f}deg")


def navigate(env, bridge, target_name, command, max_steps=60, gui=True):
    """Run OmniVLA navigation toward a target. Returns True if reached."""
    target_pos = env.objects[target_name]["position"]
    stop_dist = 0.7

    # Pre-rotate
    pre_rotate(env, target_pos, gui=gui)

    print(f"    Navigating with OmniVLA (max {max_steps} steps)...")
    print(f"    {'Step':>6} | {'Position':^20} | {'Dist':>6} | {'Vel (lin, ang)':^18}")
    print(f"    {'-'*6}-+-{'-'*20}-+-{'-'*6}-+-{'-'*18}")

    for step in range(max_steps):
        cam_img = env.get_camera_image()
        robot_pos = env.get_robot_position()
        robot_yaw = env.get_robot_orientation()
        dist = env.get_distance_to_object(target_name)

        if dist < stop_dist:
            print(f"\n    >> REACHED {target_name}! Distance: {dist:.2f}m")
            env.stop_robot()
            return True

        target_xy = np.array(target_pos[:2])
        result = bridge.predict(
            camera_image=cam_img,
            language_command=command,
            robot_position=robot_pos[:2],
            robot_yaw=robot_yaw,
            target_position=target_xy,
        )

        env.apply_velocity(result["linear_vel"], result["angular_vel"])

        if gui:
            physics_steps = int((1.0 / 3.0) / env.time_step)
            for _ in range(physics_steps):
                env.step_simulation()
                time.sleep(env.time_step)
        else:
            env.step_simulation(num_steps=80)

        if step % 5 == 0:  # print every 5th step to avoid clutter
            print(
                f"    {step:5d} | ({robot_pos[0]:+6.2f}, {robot_pos[1]:+6.2f}) "
                f"     | {dist:5.2f}m | ({result['linear_vel']:+.2f}, {result['angular_vel']:+.2f})"
            )

    dist = env.get_distance_to_object(target_name)
    print(f"\n    >> Max steps reached. Final distance: {dist:.2f}m")
    env.stop_robot()
    return dist < stop_dist


def main():
    print("=" * 60)
    print("  OmniVLA Demo - Navigate to Both Targets")
    print("=" * 60)

    # --- Create environment (GUI visible) ---
    print("\n[1/3] Creating simulation environment (GUI)...")
    env = SimulationEnvironment(gui=True)

    # --- Load model (once, reuse for both targets) ---
    print("\n[2/3] Loading OmniVLA-edge model...")
    bridge = OmniVLAEdgeBridge(
        model_weights_dir=os.path.join(project_root, "omnivla-edge")
    )

    # Let user see the initial scene
    print("\n  Scene loaded! You should see:")
    print("    - Blue robot at origin")
    print("    - Red ball at (3, 2)")
    print("    - Green cube at (-2, 3)")
    print("\n  Starting navigation in 3 seconds...")
    for i in range(3, 0, -1):
        print(f"    {i}...")
        for _ in range(240):
            env.step_simulation()
            time.sleep(env.time_step)

    # ========================================
    # Task 1: Go to the red ball
    # ========================================
    print("\n" + "=" * 60)
    print("  TASK 1: \"go to the red ball\"")
    print("=" * 60)
    bridge.reset_context()
    result1 = navigate(env, bridge, "red_ball", "go to the red ball", max_steps=60, gui=True)
    status1 = "SUCCESS" if result1 else "FAILED"
    print(f"  Result: {status1}")

    # Pause between tasks
    print("\n  Pausing 3 seconds before next task...")
    for _ in range(720):
        env.step_simulation()
        time.sleep(env.time_step)

    # ========================================
    # Task 2: Go to the green cube (from current position!)
    # ========================================
    print("\n" + "=" * 60)
    print("  TASK 2: \"go to the green cube\"")
    print("  (starting from wherever the robot ended up!)")
    print("=" * 60)
    bridge.reset_context()
    result2 = navigate(env, bridge, "green_cube", "go to the green cube", max_steps=80, gui=True)
    status2 = "SUCCESS" if result2 else "FAILED"
    print(f"  Result: {status2}")

    # ========================================
    # Summary
    # ========================================
    print("\n" + "=" * 60)
    print("  RESULTS SUMMARY")
    print("=" * 60)
    print(f"  Task 1 (red ball):   {status1}")
    print(f"  Task 2 (green cube): {status2}")
    print("=" * 60)

    # Keep window open
    print("\n  Demo complete! Close the PyBullet window or press Ctrl+C to exit.")
    try:
        while True:
            env.step_simulation()
            time.sleep(1.0 / 240.0)
    except KeyboardInterrupt:
        pass

    env.close()


if __name__ == "__main__":
    main()
