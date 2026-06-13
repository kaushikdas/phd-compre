"""
OmniVLA Navigation Debugger
============================
Systematically tests each part of the pipeline to find why
the robot doesn't navigate toward the red ball correctly.

Tests:
  1. Motor direction test (does "turn left" actually turn left?)
  2. OmniVLA inference on its own sample images vs our sim images
  3. Raw waypoint analysis (what directions does the model predict?)
  4. Language prompt comparison
  5. Ball directly in front test
"""

import os
import sys
import math
import time
import numpy as np
from PIL import Image

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)


def test_1_motor_directions():
    """
    TEST 1: Verify that positive angular_vel actually turns the robot LEFT.
    We command the robot with known velocities and check which direction it turns.
    """
    from simulation.sim_env import SimulationEnvironment

    print("\n" + "=" * 70)
    print("  TEST 1: Motor Direction Verification")
    print("=" * 70)

    env = SimulationEnvironment(gui=False)

    # --- Test A: Pure forward ---
    env.apply_velocity(linear_vel=0.3, angular_vel=0.0)
    for _ in range(240):  # 1 second
        env.step_simulation()
    pos_fwd = env.get_robot_position()
    yaw_fwd = env.get_robot_orientation()
    print(f"\n  Forward only (v=0.3, w=0.0):")
    print(f"    Position: ({pos_fwd[0]:+.3f}, {pos_fwd[1]:+.3f})")
    print(f"    Yaw: {math.degrees(yaw_fwd):+.1f}deg")
    print(f"    Expected: X > 0, Y ~ 0, Yaw ~ 0")
    print(f"    Result: {'PASS' if pos_fwd[0] > 0.05 and abs(pos_fwd[1]) < 0.05 else 'FAIL'}")

    env.close()

    # --- Test B: Turn left ---
    env2 = SimulationEnvironment(gui=False)
    env2.apply_velocity(linear_vel=0.0, angular_vel=0.5)  # pure left turn
    for _ in range(480):  # 2 seconds
        env2.step_simulation()
    yaw_left = env2.get_robot_orientation()
    print(f"\n  Turn left (v=0.0, w=+0.5):")
    print(f"    Yaw: {math.degrees(yaw_left):+.1f}deg")
    print(f"    Expected: Yaw > 0 (counterclockwise = left)")
    print(f"    Result: {'PASS' if yaw_left > 0.05 else 'FAIL - SIGN IS INVERTED!'}")

    env2.close()

    # --- Test C: Turn right ---
    env3 = SimulationEnvironment(gui=False)
    env3.apply_velocity(linear_vel=0.0, angular_vel=-0.5)  # pure right turn
    for _ in range(480):  # 2 seconds
        env3.step_simulation()
    yaw_right = env3.get_robot_orientation()
    print(f"\n  Turn right (v=0.0, w=-0.5):")
    print(f"    Yaw: {math.degrees(yaw_right):+.1f}deg")
    print(f"    Expected: Yaw < 0 (clockwise = right)")
    print(f"    Result: {'PASS' if yaw_right < -0.05 else 'FAIL - SIGN IS INVERTED!'}")

    env3.close()

    # --- Test D: Forward + left turn ---
    env4 = SimulationEnvironment(gui=False)
    env4.apply_velocity(linear_vel=0.3, angular_vel=0.3)
    for _ in range(480):
        env4.step_simulation()
    pos_fl = env4.get_robot_position()
    yaw_fl = env4.get_robot_orientation()
    print(f"\n  Forward + left (v=0.3, w=+0.3):")
    print(f"    Position: ({pos_fl[0]:+.3f}, {pos_fl[1]:+.3f})")
    print(f"    Yaw: {math.degrees(yaw_fl):+.1f}deg")
    print(f"    Expected: X > 0, Y > 0 (curved left), Yaw > 0")
    print(f"    Result: {'PASS' if pos_fl[1] > 0.01 and yaw_fl > 0.05 else 'FAIL'}")

    env4.close()


def test_2_model_on_sample_vs_sim():
    """
    TEST 2: Compare OmniVLA-edge output on:
      a) Its own sample image (from the OmniVLA repo)
      b) Our simulation camera image
    This tells us if the model works correctly in general.
    """
    print("\n" + "=" * 70)
    print("  TEST 2: Model output on real vs synthetic images")
    print("=" * 70)

    from inference.omnivla_bridge import OmniVLAEdgeBridge
    from simulation.sim_env import SimulationEnvironment

    bridge = OmniVLAEdgeBridge(
        model_weights_dir=os.path.join(project_root, "omnivla-edge")
    )

    # --- Test A: OmniVLA's own sample image ---
    sample_img_path = os.path.join(project_root, "OmniVLA", "inference", "current_img.jpg")
    sample_img = Image.open(sample_img_path).convert("RGB")
    print(f"\n  A) OmniVLA sample image ({sample_img.size}):")

    for prompt in ["red ball", "blue trash bin", "move toward the object"]:
        bridge.reset_context()
        result = bridge.predict(camera_image=sample_img, language_command=prompt)
        print(f"    Prompt: '{prompt}'")
        print(f"      Vel: linear={result['linear_vel']:+.3f}, angular={result['angular_vel']:+.3f}")
        print(f"      Waypoint 4: {result['selected_waypoint']}")
        print(f"      All waypoints (dx column): {result['waypoints'][:, 0]}")

    # --- Test B: Our simulation image ---
    env = SimulationEnvironment(gui=False)
    sim_img = env.get_camera_image()
    sim_img.save(os.path.join(project_root, "output", "debug_sim_image.png"))
    print(f"\n  B) Simulation camera image ({sim_img.size}):")

    for prompt in ["red ball", "go to the red ball", "move toward red ball"]:
        bridge.reset_context()
        result = bridge.predict(camera_image=sim_img, language_command=prompt)
        print(f"    Prompt: '{prompt}'")
        print(f"      Vel: linear={result['linear_vel']:+.3f}, angular={result['angular_vel']:+.3f}")
        print(f"      Waypoint 4: {result['selected_waypoint']}")
        print(f"      All 8 waypoints:")
        for i, wp in enumerate(result['waypoints']):
            print(f"        wp[{i}]: dx={wp[0]:+.3f}, dy={wp[1]:+.3f}, hx={wp[2]:+.3f}, hy={wp[3]:+.3f}")

    env.close()


def test_3_waypoint_analysis():
    """
    TEST 3: Run multiple steps and analyze the waypoint trajectory predictions.
    Plot the predicted waypoints vs actual movement.
    """
    print("\n" + "=" * 70)
    print("  TEST 3: Detailed waypoint analysis over multiple steps")
    print("=" * 70)

    from inference.omnivla_bridge import OmniVLAEdgeBridge
    from simulation.sim_env import SimulationEnvironment
    import matplotlib.pyplot as plt

    env = SimulationEnvironment(gui=False)
    bridge = OmniVLAEdgeBridge(
        model_weights_dir=os.path.join(project_root, "omnivla-edge")
    )

    # Run 10 steps and collect data
    all_waypoints = []
    all_positions = []
    all_velocities = []

    for step in range(10):
        cam_img = env.get_camera_image()
        robot_pos = env.get_robot_position()
        robot_yaw = env.get_robot_orientation()

        result = bridge.predict(camera_image=cam_img, language_command="red ball")

        # The angle from robot to ball
        ball_pos = np.array([3.0, 2.0])
        delta = ball_pos - robot_pos[:2]
        angle_to_ball = math.atan2(delta[1], delta[0]) - robot_yaw
        # Normalize to [-pi, pi]
        while angle_to_ball > math.pi: angle_to_ball -= 2 * math.pi
        while angle_to_ball < -math.pi: angle_to_ball += 2 * math.pi

        dist = np.linalg.norm(delta)

        print(f"\n  Step {step}:")
        print(f"    Robot pos: ({robot_pos[0]:+.2f}, {robot_pos[1]:+.2f}), yaw: {math.degrees(robot_yaw):+.1f}deg")
        print(f"    Angle to ball (in robot frame): {math.degrees(angle_to_ball):+.1f}deg")
        print(f"    Distance to ball: {dist:.2f}m")
        print(f"    Model output: linear={result['linear_vel']:+.3f}, angular={result['angular_vel']:+.3f}")
        print(f"    Waypoint[4]: dx={result['selected_waypoint'][0]:+.4f}, dy={result['selected_waypoint'][1]:+.4f}")
        print(f"    -> Model wants to turn: {'LEFT' if result['angular_vel'] > 0 else 'RIGHT'}")
        print(f"    -> Ball is to the: {'LEFT' if angle_to_ball > 0 else 'RIGHT'}")
        print(f"    -> Match: {'YES' if (angle_to_ball > 0) == (result['angular_vel'] > 0) else 'NO - MISMATCH!'}")

        all_waypoints.append(result['waypoints'])
        all_positions.append(robot_pos[:2].copy())
        all_velocities.append((result['linear_vel'], result['angular_vel']))

        # Apply velocity and step
        env.apply_velocity(result['linear_vel'], result['angular_vel'])
        for _ in range(80):
            env.step_simulation()

    env.close()

    # Save analysis plot
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    # Plot 1: Robot trajectory
    positions = np.array(all_positions)
    axes[0].plot(positions[:, 0], positions[:, 1], 'b-o', label='Robot path')
    axes[0].plot(3.0, 2.0, 'ro', markersize=15, label='Red ball')
    axes[0].plot(-2.0, 3.0, 'gs', markersize=15, label='Green cube')
    axes[0].plot(0, 0, 'b^', markersize=15, label='Start')
    axes[0].set_xlabel('X (m)')
    axes[0].set_ylabel('Y (m)')
    axes[0].set_title('Robot Trajectory')
    axes[0].legend()
    axes[0].set_aspect('equal')
    axes[0].grid(True)

    # Plot 2: Predicted waypoints at step 0
    wp0 = all_waypoints[0] * 0.1  # scale by metric_waypoint_spacing
    axes[1].plot(wp0[:, 1], wp0[:, 0], 'g-o', label='Predicted path')
    axes[1].plot(0, 0, 'b^', markersize=15, label='Robot')
    axes[1].set_xlabel('Lateral (m)')
    axes[1].set_ylabel('Forward (m)')
    axes[1].set_title('Predicted Waypoints at Step 0\n(robot frame: X=right, Y=forward)')
    axes[1].legend()
    axes[1].grid(True)

    # Plot 3: Angular velocities
    angvels = [v[1] for v in all_velocities]
    axes[2].bar(range(len(angvels)), angvels, color=['green' if v > 0 else 'red' for v in angvels])
    axes[2].axhline(y=0, color='k', linestyle='-')
    axes[2].set_xlabel('Step')
    axes[2].set_ylabel('Angular velocity (rad/s)')
    axes[2].set_title('Angular velocities\n(green=left, red=right)')
    axes[2].grid(True)

    plt.tight_layout()
    plt.savefig(os.path.join(project_root, "output", "debug_analysis.png"), dpi=100)
    print(f"\n  Analysis plot saved to output/debug_analysis.png")


def test_4_ball_in_front():
    """
    TEST 4: Place the ball directly in front of the robot.
    If the robot goes straight, the model works for forward motion at least.
    """
    print("\n" + "=" * 70)
    print("  TEST 4: Ball directly in front (no turning needed)")
    print("=" * 70)

    import pybullet as p
    from simulation.sim_env import SimulationEnvironment
    from inference.omnivla_bridge import OmniVLAEdgeBridge

    env = SimulationEnvironment(gui=False)

    # Move the red ball to directly in front of the robot
    p.resetBasePositionAndOrientation(
        env.objects["red_ball"]["id"],
        [2.0, 0.0, 0.15],  # directly in front
        p.getQuaternionFromEuler([0, 0, 0])
    )
    env.objects["red_ball"]["position"] = [2.0, 0.0, 0.15]
    print("  Moved red ball to (2.0, 0.0) — directly in front of robot")

    bridge = OmniVLAEdgeBridge(
        model_weights_dir=os.path.join(project_root, "omnivla-edge")
    )

    # Run 20 steps
    for step in range(20):
        cam_img = env.get_camera_image()
        if step == 0:
            cam_img.save(os.path.join(project_root, "output", "debug_ball_in_front.png"))

        result = bridge.predict(camera_image=cam_img, language_command="red ball")
        dist = env.get_distance_to_object("red_ball")
        pos = env.get_robot_position()

        print(f"  Step {step:2d} | Pos: ({pos[0]:+.2f}, {pos[1]:+.2f}) | Dist: {dist:.2f}m | "
              f"Vel: ({result['linear_vel']:+.3f}, {result['angular_vel']:+.3f})")

        if dist < 0.4:
            print(f"\n  [SUCCESS] Reached the ball!")
            break

        env.apply_velocity(result['linear_vel'], result['angular_vel'])
        for _ in range(80):
            env.step_simulation()

    env.close()


if __name__ == "__main__":
    os.makedirs(os.path.join(project_root, "output"), exist_ok=True)

    print("=" * 70)
    print("  OmniVLA Navigation Debugger")
    print("=" * 70)

    # Run all tests
    test_1_motor_directions()
    test_2_model_on_sample_vs_sim()
    test_3_waypoint_analysis()
    test_4_ball_in_front()

    print("\n" + "=" * 70)
    print("  All debug tests complete!")
    print("  Check output/ folder for saved images and analysis plots.")
    print("=" * 70)
