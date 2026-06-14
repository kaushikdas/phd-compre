import os
import numpy as np
import mujoco
import mujoco.viewer
from PIL import Image
import time

class ReconstructedSimulationEnvironmentMJ:
    """
    MuJoCo implementation of the reconstructed environment.
    Loads the custom robot_scene_reconstructed.xml and translates
    semantic targets (e.g. "sofa") to physical MuJoCo bodies.
    """
    def __init__(self, gui=True):
        self.gui = gui
        
        # Load model from the generated XML
        model_path = os.path.join(os.path.dirname(__file__), "robot_scene_reconstructed.xml")
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Reconstructed MuJoCo scene file not found at {model_path}! Run vlm_scene_annotator.py first.")
            
        self.model = mujoco.MjModel.from_xml_path(model_path)
        self.data = mujoco.MjData(self.model)
        
        # Renderer for VLA/CLIP camera feed (matching PyBullet 320x240)
        self.renderer = mujoco.Renderer(self.model, height=240, width=320)
        
        # Physical constants for differential drive
        self.wheel_radius = 0.05
        self.wheel_separation = 0.34
        self.time_step = self.model.opt.timestep
        
        # Viewer for GUI mode
        self.viewer = None
        if self.gui:
            self.viewer = mujoco.viewer.launch_passive(
                self.model, self.data, 
                show_left_ui=False, 
                show_right_ui=False
            )
            
    def step_simulation(self, num_steps=None):
        """
        Step physics.
        Control loop runs at 3Hz, so one logical step is ~0.33s.
        """
        steps = num_steps if num_steps else int(0.33 / self.time_step)
        sync_every = max(1, int(0.016 / self.time_step))
        
        for i in range(steps):
            mujoco.mj_step(self.model, self.data)
            if self.gui and self.viewer and self.viewer.is_running():
                if i % sync_every == 0:
                    self.viewer.sync()
            
        if self.gui:
            time.sleep(0.01)

    def apply_velocity(self, linear_vel, angular_vel):
        """
        Convert (v, w) commands to MuJoCo wheel velocity actuators.
        """
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
        self.data.qvel[0:6] = 0.0  # Reset robot base velocity to stop sliding

    def get_camera_image(self):
        """Render RGB from the robot's front-facing camera."""
        self.renderer.update_scene(self.data, camera="robot_eye")
        rgb = self.renderer.render()
        return Image.fromarray(rgb)

    def get_robot_position(self):
        """Returns [x, y, z] world coordinates of the robot."""
        return self.data.body("robot").xpos.copy()

    def get_robot_orientation(self):
        """Returns Yaw (around Z) in radians."""
        quat = self.data.body("robot").xquat
        w, x, y, z = quat
        siny_cosp = 2 * (w * z + x * y)
        cosy_cosp = 1 - 2 * (y * y + z * z)
        return np.arctan2(siny_cosp, cosy_cosp)

    def find_body_name_by_prefix(self, prefix):
        """Find a body index/name by prefix match (e.g. 'obstacle_sofa')."""
        for i in range(self.model.nbody):
            b_name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_BODY, i)
            if b_name and b_name.lower().startswith(prefix.lower()):
                return b_name
        return None

    def get_distance_to_object(self, name):
        """Compute 2D distance to a target body, automatically mapping semantic tags."""
        robot_pos = self.get_robot_position()[:2]
        
        target_body = name
        try:
            self.model.body(target_body)
        except KeyError:
            # If target body is not found directly, check VLM obstacle prefixes
            possible_name = self.find_body_name_by_prefix(f"obstacle_{name}")
            if possible_name:
                target_body = possible_name
            else:
                possible_name = self.find_body_name_by_prefix(name)
                if possible_name:
                    target_body = possible_name
                else:
                    print(f"[Warn] Target body '{name}' not found in model. Falling back to origin check.")
                    return 999.0
                    
        target_pos = self.data.body(target_body).xpos[:2]
        return np.linalg.norm(robot_pos - target_pos)

    def close(self):
        """Shutdown the viewer and renderer."""
        if self.viewer:
            self.viewer.close()
