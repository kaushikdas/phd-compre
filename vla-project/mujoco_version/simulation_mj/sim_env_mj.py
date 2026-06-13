import os
import numpy as np
import mujoco
import mujoco.viewer
from PIL import Image
import time

class SimulationEnvironmentMJ:
    """
    MuJoCo implementation of the OmniVLA navigation environment.
    Maintains API compatibility with the PyBullet version.
    """
    def __init__(self, gui=True):
        self.gui = gui
        
        # Load model from the MJCF XML
        model_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "robot", "robot_scene.xml")
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"MuJoCo scene file not found at {model_path}")
            
        self.model = mujoco.MjModel.from_xml_path(model_path)
        self.data = mujoco.MjData(self.model)
        
        # Renderer for CLIP/OmniVLA (Match PyBullet 320x240)
        self.renderer = mujoco.Renderer(self.model, height=240, width=320)
        
        # Physical constants
        self.wheel_radius = 0.05
        self.wheel_separation = 0.34
        self.time_step = self.model.opt.timestep
        
        # Viewer for GUI mode
        self.viewer = None
        if self.gui:
            # Launch with both UI panels hidden by default for a clean view
            self.viewer = mujoco.viewer.launch_passive(
                self.model, self.data, 
                show_left_ui=False, 
                show_right_ui=False
            )
            
    def step_simulation(self, num_steps=None):
        """
        Step physics.
        Control loop runs at 3Hz, so one 'logical step' is ~0.33s.
        """
        steps = num_steps if num_steps else int(0.33 / self.time_step)
        
        # We want to sync the viewer at roughly 60Hz for smoothness
        sync_every = max(1, int(0.016 / self.time_step))
        
        for i in range(steps):
            mujoco.mj_step(self.model, self.data)
            if self.gui and self.viewer and self.viewer.is_running():
                if i % sync_every == 0:
                    self.viewer.sync()
            
        # Throttling to keep it near real-time (optional)
        if self.gui:
            time.sleep(0.01)

    def apply_velocity(self, linear_vel, angular_vel):
        """
        Convert (v, w) commands to MuJoCo velocity actuators.
        """
        # Linear/Angular to Wheel Velocity (rad/s)
        v_left = (linear_vel - angular_vel * self.wheel_separation / 2.0) / self.wheel_radius
        v_right = (linear_vel + angular_vel * self.wheel_separation / 2.0) / self.wheel_radius
        
        # Apply to 4-wheel velocity actuators (indices 0-3 in XML)
        self.data.ctrl[0] = v_left
        self.data.ctrl[1] = v_right
        self.data.ctrl[2] = v_left
        self.data.ctrl[3] = v_right

    def stop_robot(self):
        """Set all actuators to zero and dampen velocity."""
        self.data.ctrl[:] = 0.0
        # Instant mathematical stop for stabilization
        self.data.qvel[0:6] = 0.0

    def get_camera_image(self):
        """Render RGB from the 'robot_eye' camera."""
        self.renderer.update_scene(self.data, camera="robot_eye")
        rgb = self.renderer.render()
        return Image.fromarray(rgb)

    def get_robot_position(self):
        """Returns [x, y, z] world coordinates."""
        return self.data.body("robot").xpos.copy()

    def get_robot_orientation(self):
        """Returns Yaw (around Z) in radians."""
        quat = self.data.body("robot").xquat
        # MuJoCo quat is [w, x, y, z]
        w, x, y, z = quat
        siny_cosp = 2 * (w * z + x * y)
        cosy_cosp = 1 - 2 * (y * y + z * z)
        return np.arctan2(siny_cosp, cosy_cosp)

    def get_distance_to_object(self, name):
        """Compute 2D distance to a target body."""
        robot_pos = self.get_robot_position()[:2]
        target_pos = self.data.body(name).xpos[:2]
        return np.linalg.norm(robot_pos - target_pos)

    def close(self):
        """Shutdown the viewer and renderer."""
        if self.viewer:
            self.viewer.close()
