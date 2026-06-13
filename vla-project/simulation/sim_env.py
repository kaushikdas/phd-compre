"""
PyBullet Simulation Environment for OmniVLA Navigation
=======================================================
Creates a flat world with:
  - A 4-wheeled differential-drive robot
  - A red ball (target A)
  - A green cube (target B)
  - A virtual egocentric camera attached to the robot
"""

import os
import math
import numpy as np
import pybullet as p
import pybullet_data
from PIL import Image


class SimulationEnvironment:
    """
    Manages the full PyBullet simulation:
    robot loading, object placement, camera rendering, and motor control.
    """

    def __init__(self, gui=True, time_step=1.0 / 240.0):
        """
        Args:
            gui: If True, opens a 3D visualization window. False = headless.
            time_step: Physics timestep (default 1/240 s for stable simulation).
        """
        self.time_step = time_step
        self.gui = gui

        # ---------- Start PyBullet ----------
        if gui:
            self.physics_client = p.connect(p.GUI)
            # Camera angle for the visualization window (bird's eye view)
            p.resetDebugVisualizerCamera(
                cameraDistance=5.0,
                cameraYaw=0,
                cameraPitch=-45,
                cameraTargetPosition=[0, 0, 0]
            )
        else:
            self.physics_client = p.connect(p.DIRECT)

        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.setGravity(0, 0, -9.81)
        p.setTimeStep(time_step)

        # ---------- Load Ground Plane ----------
        self.plane_id = p.loadURDF("plane.urdf")
        # Give the plane a textured appearance
        p.changeVisualShape(self.plane_id, -1,
                            rgbaColor=[0.85, 0.82, 0.75, 1.0])

        # ---------- Load Robot ----------
        robot_urdf_path = os.path.join(
            os.path.dirname(os.path.dirname(__file__)),
            "robot", "four_wheeled_robot.urdf"
        )
        self.robot_id = p.loadURDF(
            robot_urdf_path,
            basePosition=[0, 0, 0.05],
            baseOrientation=p.getQuaternionFromEuler([0, 0, 0]),
            useFixedBase=False
        )

        # Map joint names to indices for motor control
        self.joint_map = {}
        for i in range(p.getNumJoints(self.robot_id)):
            joint_info = p.getJointInfo(self.robot_id, i)
            joint_name = joint_info[1].decode("utf-8")
            self.joint_map[joint_name] = i
            # Disable default damping so we can control wheels freely
            p.setJointMotorControl2(
                self.robot_id, i,
                controlMode=p.VELOCITY_CONTROL, force=0
            )

        # Wheel joint indices (for differential drive)
        self.left_wheels = [
            self.joint_map["front_left_wheel_joint"],
            self.joint_map["rear_left_wheel_joint"],
        ]
        self.right_wheels = [
            self.joint_map["front_right_wheel_joint"],
            self.joint_map["rear_right_wheel_joint"],
        ]
        self.camera_link_index = self.joint_map["camera_joint"]

        # Robot physical parameters
        self.wheel_radius = 0.05       # meters
        self.wheel_separation = 0.34   # distance between left and right wheels

        # ---------- Place Objects ----------
        self.objects = {}
        self._create_objects()

        # ---------- Camera Settings ----------
        self.cam_width = 320
        self.cam_height = 240
        self.cam_fov = 60
        self.cam_near = 0.05
        self.cam_far = 20.0
        self.cam_projection = p.computeProjectionMatrixFOV(
            fov=self.cam_fov,
            aspect=self.cam_width / self.cam_height,
            nearVal=self.cam_near,
            farVal=self.cam_far
        )

        # Simulation step counter
        self.step_count = 0

        print("[SimEnv] Environment initialized successfully!")
        print(f"  Robot ID: {self.robot_id}")
        print(f"  Joint map: {self.joint_map}")
        print(f"  Objects: {list(self.objects.keys())}")

    def _create_objects(self):
        """
        Place a red ball and a green cube in the scene.
        These are the navigation targets for OmniVLA commands.
        """
        # --- Red Ball ---
        ball_radius = 0.15
        ball_position = [3.0, 2.0, ball_radius]

        ball_visual = p.createVisualShape(
            shapeType=p.GEOM_SPHERE,
            radius=ball_radius,
            rgbaColor=[0.9, 0.1, 0.1, 1.0]  # Red
        )
        ball_collision = p.createCollisionShape(
            shapeType=p.GEOM_SPHERE,
            radius=ball_radius
        )
        ball_id = p.createMultiBody(
            baseMass=0.5,
            baseCollisionShapeIndex=ball_collision,
            baseVisualShapeIndex=ball_visual,
            basePosition=ball_position
        )
        self.objects["red_ball"] = {
            "id": ball_id,
            "position": ball_position,
            "description": "red ball"
        }

        # --- Green Cube ---
        cube_half_size = 0.12
        cube_position = [-2.0, 3.0, cube_half_size]

        cube_visual = p.createVisualShape(
            shapeType=p.GEOM_BOX,
            halfExtents=[cube_half_size] * 3,
            rgbaColor=[0.1, 0.8, 0.1, 1.0]  # Green
        )
        cube_collision = p.createCollisionShape(
            shapeType=p.GEOM_BOX,
            halfExtents=[cube_half_size] * 3
        )
        cube_id = p.createMultiBody(
            baseMass=0.5,
            baseCollisionShapeIndex=cube_collision,
            baseVisualShapeIndex=cube_visual,
            basePosition=cube_position
        )
        self.objects["green_cube"] = {
            "id": cube_id,
            "position": cube_position,
            "description": "green cube"
        }

        # Add text labels above objects (visible in GUI)
        if self.gui:
            p.addUserDebugText("Red Ball", [3.0, 2.0, 0.5],
                               textColorRGB=[1, 0, 0], textSize=1.5)
            p.addUserDebugText("Green Cube", [-2.0, 3.0, 0.5],
                               textColorRGB=[0, 0.8, 0], textSize=1.5)

        print(f"  Red ball at {ball_position}")
        print(f"  Green cube at {cube_position}")

    # =========================================================
    # CAMERA
    # =========================================================
    def get_camera_image(self):
        """
        Render an egocentric RGB image from the robot's front camera.

        Returns:
            PIL.Image: RGB image from the robot's perspective (320x240).
        """
        # Get the camera link's world position and orientation
        cam_state = p.getLinkState(self.robot_id, self.camera_link_index,
                                  computeForwardKinematics=True)
        cam_pos = list(cam_state[0])   # world position
        cam_orn = cam_state[1]         # world orientation (quaternion)

        # Convert quaternion to rotation matrix
        rot_matrix = np.array(p.getMatrixFromQuaternion(cam_orn)).reshape(3, 3)

        # Camera looks along the local X-axis (forward direction of robot)
        forward_vec = rot_matrix @ np.array([1, 0, 0])
        up_vec = rot_matrix @ np.array([0, 0, 1])

        # Target point the camera looks at (1m ahead)
        target_pos = np.array(cam_pos) + forward_vec

        # Compute view matrix
        view_matrix = p.computeViewMatrix(
            cameraEyePosition=cam_pos,
            cameraTargetPosition=target_pos.tolist(),
            cameraUpVector=up_vec.tolist()
        )

        # Render
        _, _, rgb_pixels, _, _ = p.getCameraImage(
            width=self.cam_width,
            height=self.cam_height,
            viewMatrix=view_matrix,
            projectionMatrix=self.cam_projection,
            renderer=p.ER_BULLET_HARDWARE_OPENGL if self.gui else p.ER_TINY_RENDERER
        )

        # Convert to PIL Image
        rgb_array = np.array(rgb_pixels, dtype=np.uint8).reshape(
            self.cam_height, self.cam_width, 4
        )
        rgb_image = Image.fromarray(rgb_array[:, :, :3], "RGB")

        return rgb_image

    # =========================================================
    # MOTOR CONTROL
    # =========================================================
    def apply_velocity(self, linear_vel, angular_vel):
        """
        Apply differential-drive velocity commands to the robot.

        This converts (linear_vel, angular_vel) into left/right wheel speeds
        using the standard differential drive model:
            v_left  = (linear - angular * wheel_sep / 2) / wheel_radius
            v_right = (linear + angular * wheel_sep / 2) / wheel_radius

        Args:
            linear_vel:  Forward velocity in m/s (positive = forward)
            angular_vel: Turning rate in rad/s (positive = turn left)
        """
        # Differential drive kinematics
        # Note: wheel joints are rotated 90deg in URDF, so negate to match forward = +X
        v_left = -(linear_vel - angular_vel * self.wheel_separation / 2.0) / self.wheel_radius
        v_right = -(linear_vel + angular_vel * self.wheel_separation / 2.0) / self.wheel_radius

        max_force = 5.0  # torque applied to each wheel

        for joint_idx in self.left_wheels:
            p.setJointMotorControl2(
                self.robot_id, joint_idx,
                controlMode=p.VELOCITY_CONTROL,
                targetVelocity=v_left,
                force=max_force
            )
        for joint_idx in self.right_wheels:
            p.setJointMotorControl2(
                self.robot_id, joint_idx,
                controlMode=p.VELOCITY_CONTROL,
                targetVelocity=v_right,
                force=max_force
            )

    def stop_robot(self):
        """Immediately stop all wheels."""
        self.apply_velocity(0.0, 0.0)

    # =========================================================
    # STATE QUERIES
    # =========================================================
    def get_robot_position(self):
        """Get robot's (x, y, z) world position."""
        pos, _ = p.getBasePositionAndOrientation(self.robot_id)
        return np.array(pos)

    def get_robot_orientation(self):
        """Get robot's yaw angle in radians."""
        _, orn = p.getBasePositionAndOrientation(self.robot_id)
        euler = p.getEulerFromQuaternion(orn)
        return euler[2]  # yaw

    def get_distance_to_object(self, object_name):
        """
        Compute distance from robot to a named object.

        Args:
            object_name: Key from self.objects (e.g., "red_ball", "green_cube")

        Returns:
            float: Euclidean distance in meters (2D, ignoring height)
        """
        robot_pos = self.get_robot_position()
        obj_pos = np.array(self.objects[object_name]["position"])
        # 2D distance (x, y only)
        return np.linalg.norm(robot_pos[:2] - obj_pos[:2])

    # =========================================================
    # SIMULATION STEP
    # =========================================================
    def step_simulation(self, num_steps=1):
        """
        Advance the physics simulation by num_steps timesteps.
        At 240 Hz, 80 steps ≈ 0.33s (matching OmniVLA's 3 Hz control rate).
        """
        for _ in range(num_steps):
            p.stepSimulation()
        self.step_count += num_steps

    def close(self):
        """Disconnect from PyBullet."""
        p.disconnect()
        print("[SimEnv] Environment closed.")


# =========================================================
# Quick test: run this file directly to verify the setup
# =========================================================
if __name__ == "__main__":
    import time

    print("=" * 60)
    print("Testing PyBullet Simulation Environment")
    print("=" * 60)

    env = SimulationEnvironment(gui=True)

    # Let the robot sit for 2 seconds so you can see the scene
    print("\n[Test] Scene loaded. Waiting 2 seconds...")
    for _ in range(480):  # 2 seconds at 240 Hz
        env.step_simulation()
        time.sleep(1.0 / 240.0)

    # Test camera
    print("[Test] Capturing camera image...")
    img = env.get_camera_image()
    img.save(os.path.join(os.path.dirname(__file__), "..", "test_camera.png"))
    print(f"  Camera image saved: test_camera.png ({img.size})")

    # Test motor control: drive forward for 2 seconds
    print("[Test] Driving forward for 2 seconds...")
    env.apply_velocity(linear_vel=0.3, angular_vel=0.0)
    for _ in range(480):
        env.step_simulation()
        time.sleep(1.0 / 240.0)

    pos = env.get_robot_position()
    print(f"  Robot position after driving: ({pos[0]:.2f}, {pos[1]:.2f})")
    print(f"  Distance to red ball: {env.get_distance_to_object('red_ball'):.2f}m")
    print(f"  Distance to green cube: {env.get_distance_to_object('green_cube'):.2f}m")

    # Turn left for 2 seconds
    print("[Test] Turning left for 2 seconds...")
    env.apply_velocity(linear_vel=0.1, angular_vel=0.5)
    for _ in range(480):
        env.step_simulation()
        time.sleep(1.0 / 240.0)

    # Capture another image after moving
    img2 = env.get_camera_image()
    img2.save(os.path.join(os.path.dirname(__file__), "..", "test_camera_moved.png"))
    print(f"  Second camera image saved: test_camera_moved.png")

    # Keep window open for inspection
    print("\n[Test] All tests passed! Press Ctrl+C to exit.")
    try:
        while True:
            env.step_simulation()
            time.sleep(1.0 / 240.0)
    except KeyboardInterrupt:
        pass

    env.close()
