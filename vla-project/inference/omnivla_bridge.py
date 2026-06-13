"""
OmniVLA-edge Bridge
====================
Adapter between PyBullet simulation and OmniVLA-edge model.

Handles:
  1. Loading OmniVLA-edge model + CLIP text encoder
  2. Converting PyBullet camera images → OmniVLA input format
  3. Running inference → velocity commands
  4. PD controller for waypoint tracking
"""

import os
import sys
import math
import numpy as np
from PIL import Image

import torch
import torch.nn as nn
import clip


class OmniVLAEdgeBridge:
    """
    Loads OmniVLA-edge and provides a simple interface:
        velocity = bridge.predict(camera_image, language_command)
    """

    def __init__(self, model_weights_dir, device=None):
        """
        Args:
            model_weights_dir: Path to the omnivla-edge directory containing
                              'omnivla-edge.pth'
            device: torch device (defaults to cuda:0 if available)
        """
        self.device = device or torch.device(
            "cuda:0" if torch.cuda.is_available() else "cpu"
        )
        print(f"[OmniVLA-edge] Using device: {self.device}")

        # Image sizes expected by OmniVLA-edge
        self.img_size = (96, 96)
        self.clip_img_size = (224, 224)

        # Context queue: stores recent frames (OmniVLA-edge uses 5 + 1 current)
        self.context_size = 5
        self.context_queue = []

        # No fisheye mask — we use a simple rectangular camera
        self.mask_96 = np.ones((96, 96, 3), dtype=np.float32)
        self.mask_224 = np.ones((224, 224, 3), dtype=np.float32)

        # Metric waypoint spacing (from OmniVLA-edge default)
        self.metric_waypoint_spacing = 0.1

        # Model parameters matching OmniVLA-edge architecture
        self.model_params = {
            "model_type": "omnivla-edge",
            "len_traj_pred": 8,
            "learn_angle": True,
            "context_size": 5,
            "obs_encoder": "efficientnet-b0",
            "encoding_size": 256,
            "obs_encoding_size": 1024,
            "goal_encoding_size": 1024,
            "late_fusion": False,
            "mha_num_attention_heads": 4,
            "mha_num_attention_layers": 4,
            "mha_ff_dim_factor": 4,
            "clip_type": "ViT-B/32",
        }

        # Load model
        self._load_model(model_weights_dir)

        # Velocity limits (from OmniVLA-edge defaults)
        self.max_linear = 0.3   # m/s
        self.max_angular = 0.3  # rad/s
        self.dt = 1.0 / 3.0     # control period (3 Hz)

        print("[OmniVLA-edge] Bridge initialized successfully!")

    def _load_model(self, model_weights_dir):
        """Load OmniVLA-edge model and CLIP text encoder."""
        # We need to import OmniVLA's utility functions
        # The OmniVLA repo provides: load_model, transform_images_PIL, etc.
        omnivla_inference_dir = os.path.join(
            os.path.dirname(os.path.dirname(__file__)),
            "OmniVLA", "inference"
        )

        if not os.path.exists(omnivla_inference_dir):
            raise FileNotFoundError(
                f"OmniVLA repo not found at expected path.\n"
                f"Expected: {omnivla_inference_dir}\n"
                f"Please clone: git clone https://github.com/NHirose/OmniVLA.git"
            )

        # Add OmniVLA inference directory to path
        sys.path.insert(0, omnivla_inference_dir)
        from utils_policy import load_model, transform_images_PIL, transform_images_PIL_mask

        self.transform_images_PIL = transform_images_PIL
        self.transform_images_PIL_mask = transform_images_PIL_mask

        # Also need the map image transform
        from utils_policy import transform_images_map
        self.transform_images_map = transform_images_map

        # Load checkpoint
        ckpt_path = os.path.join(model_weights_dir, "omnivla-edge.pth")
        if not os.path.exists(ckpt_path):
            raise FileNotFoundError(
                f"Model weights not found at {ckpt_path}\n"
                f"Please download: git clone https://huggingface.co/NHirose/omnivla-edge"
            )

        print(f"[OmniVLA-edge] Loading model from {ckpt_path}...")
        # load_model returns (model, text_encoder, clip_preprocess)
        self.model, self.text_encoder, self.preprocess = load_model(
            ckpt_path, self.model_params, self.device
        )
        self.text_encoder = self.text_encoder.to(self.device).eval()
        self.model = self.model.to(self.device).eval()
        print("[OmniVLA-edge] Model loaded successfully!")

    def semantic_search(self, camera_image, text_query):
        """
        Multi-Scale Semantic Radar.
        Checks the image at Global (wide) and Pinpoint (narrow) scales.
        Returns the best bearing and confidence across all scales.
        """
        w, h = camera_image.size
        
        # 1. Prepare Text Features (Cached for internal loop)
        with torch.no_grad():
            text_tokens = clip.tokenize([text_query]).to(self.device)
            text_features = self.text_encoder.encode_text(text_tokens)
            text_features /= text_features.norm(dim=-1, keepdim=True)

        best_score = -1.0
        best_bearing = 0.0

        # ---- Scale A: GLOBAL VIEW (Close range / High Context) ----
        # Best for when the ball is right in front of the robot
        global_input = self.preprocess(camera_image).unsqueeze(0).to(self.device)
        with torch.no_grad():
            global_feat = self.text_encoder.encode_image(global_input)
            global_feat /= global_feat.norm(dim=-1, keepdim=True)
            global_score = (100.0 * global_feat @ text_features.T).item()
            
            best_score = global_score
            best_bearing = 0.0 # Straight ahead

        # ---- Scale B: PINPOINT RADAR (Long range / High Precision) ----
        # 9 Overlapping strips across the 60deg FOV
        strips = 9
        strip_w = w // 3 
        # CORRECTION: Pixel 0 (Left) must be Positive Angle, Pixel W (Right) must be Negative
        bearings = np.linspace(math.radians(26), -math.radians(26), strips)
        centers = np.linspace(strip_w//2, w - strip_w//2, strips)

        for i, cx in enumerate(centers):
            left = int(cx - strip_w // 2)
            right = int(cx + strip_w // 2)
            crop = camera_image.crop((left, 0, right, h))
            
            img_input = self.preprocess(crop).unsqueeze(0).to(self.device)
            
            with torch.no_grad():
                img_features = self.text_encoder.encode_image(img_input)
                img_features /= img_features.norm(dim=-1, keepdim=True)
                score = (100.0 * img_features @ text_features.T).item()
                
                # We reward the pinpoint score if it's significantly better than global
                if score > best_score:
                    best_score = score
                    best_bearing = bearings[i]
        
        return best_bearing, best_score

    def predict(self, camera_image, language_command, goal_image=None,
                robot_position=None, robot_yaw=None, target_position=None):
        """
        Run OmniVLA-edge inference on a camera image + language command.

        Args:
            camera_image: PIL.Image — current egocentric RGB image from robot
            language_command: str — natural language instruction
                            (e.g., "red ball", "go to the green cube")
            goal_image: PIL.Image or None — egocentric goal image (optional)
            robot_position: np.ndarray [x, y] — robot's world position (optional)
            robot_yaw: float — robot's heading in radians (optional)
            target_position: np.ndarray [x, y] — target's world position (optional)

        When robot_position, robot_yaw, and target_position are ALL provided,
        uses pose+language mode (modality 8) for much better navigation.
        Otherwise falls back to language-only mode (modality 7).

        Returns:
            dict with:
                'linear_vel': float — forward velocity (m/s)
                'angular_vel': float — turning rate (rad/s)
                'waypoints': np.ndarray — predicted 8 waypoints (8x4)
                'selected_waypoint': np.ndarray — waypoint used for control (4,)
        """
        # ---- Determine modality ----
        use_pose = (robot_position is not None and
                    robot_yaw is not None and
                    target_position is not None)

        # ---- Prepare images ----
        # Resize to expected sizes
        img_96 = camera_image.resize(self.img_size)
        img_224 = camera_image.resize(self.clip_img_size)

        # Update context queue (sliding window of recent frames)
        self.context_queue.append(img_96)
        if len(self.context_queue) > self.context_size + 1:
            self.context_queue.pop(0)

        # Pad context if not enough frames yet (repeat oldest frame)
        while len(self.context_queue) < self.context_size + 1:
            self.context_queue.insert(0, self.context_queue[0])

        # Transform observation images
        obs_images = self.transform_images_PIL_mask(
            self.context_queue, self.mask_96
        )
        obs_images = torch.split(obs_images.to(self.device), 3, dim=1)
        obs_image_cur = obs_images[-1].to(self.device)
        obs_images = torch.cat(obs_images, dim=1).to(self.device)

        # Current large image for CLIP
        cur_large_img = self.transform_images_PIL_mask(
            img_224, self.mask_224
        ).to(self.device)

        # ---- Dummy satellite images ----
        # OmniVLA-edge supports satellite input but we don't use it
        satellite_cur = Image.new("RGB", (352, 352), color=(0, 0, 0))
        satellite_goal = Image.new("RGB", (352, 352), color=(0, 0, 0))
        current_map_image = self.transform_images_map(satellite_cur)
        goal_map_image = self.transform_images_map(satellite_goal)
        map_images = torch.cat(
            (current_map_image.to(self.device),
             goal_map_image.to(self.device),
             obs_image_cur),
            axis=1
        )

        # ---- Goal pose ----
        if use_pose:
            # Compute relative goal in robot-local frame
            # PyBullet: when yaw=0, robot faces +X, left is +Y
            # OmniVLA robot frame: X=forward, Y=left
            # goal_pose format: [Y_local/spacing, -X_local/spacing, cos, sin]
            delta_x = target_position[0] - robot_position[0]
            delta_y = target_position[1] - robot_position[1]

            # Standard 2D rotation to robot-local frame
            # robot_yaw is in standard math convention (CCW positive)
            cos_h = math.cos(robot_yaw)
            sin_h = math.sin(robot_yaw)
            # forward = how far ahead the goal is (along robot's X)
            # left = how far left the goal is (along robot's Y)
            forward = delta_x * cos_h + delta_y * sin_h
            left = -delta_x * sin_h + delta_y * cos_h

            # Clamp to threshold distance (30m)
            radius = math.sqrt(forward**2 + left**2)
            thres_dist = 30.0
            if radius > thres_dist:
                forward *= thres_dist / radius
                left *= thres_dist / radius

            # Goal heading: angle from robot to target
            angle_to_target = math.atan2(left, forward)

            # OmniVLA goal_pose format: [left/s, -forward/s, cos, sin]
            # which corresponds to [Y_local/spacing, -X_local/spacing, cos, sin]
            goal_pose_torch = torch.from_numpy(np.array([
                left / self.metric_waypoint_spacing,
                -forward / self.metric_waypoint_spacing,
                np.cos(angle_to_target),
                np.sin(angle_to_target)
            ])).unsqueeze(0).float().to(self.device)
        else:
            # Dummy pose for language-only mode
            goal_pose_torch = torch.from_numpy(np.array([
                0.0, 0.0, 1.0, 0.0
            ])).unsqueeze(0).float().to(self.device)

        # ---- Goal image ----
        if goal_image is None:
            goal_image_pil = Image.new("RGB", self.img_size, color=(0, 0, 0))
        else:
            goal_image_pil = goal_image.resize(self.img_size)
        goal_image_tensor = self.transform_images_PIL_mask(
            goal_image_pil, self.mask_96
        ).to(self.device)

        # ---- Language instruction ----
        obj_inst_lan = clip.tokenize(language_command, truncate=True).to(self.device)

        # ---- Build batch ----
        batch = {
            "obs_images": obs_images,
            "goal_pose_torch": goal_pose_torch,
            "map_images": map_images,
            "goal_image": goal_image_tensor,
            "obj_inst_lan": obj_inst_lan,
            "cur_large_img": cur_large_img,
        }

        # ---- Modality selection ----
        # modality 4 = pose only (strongest directional signal)
        # modality 8 = pose + language (directional + language understanding)
        # modality 7 = language only (fallback)
        if use_pose:
            modality_id = 4  # pose only — best for simulation
        else:
            modality_id = 7  # language only
        modality_id_select = torch.tensor([modality_id]).to(self.device)

        # ---- Forward pass ----
        bimg = batch["goal_image"].size(0) if len(batch["goal_image"].shape) == 4 else 1
        with torch.no_grad():
            feat_text_lan = self.text_encoder.encode_text(batch["obj_inst_lan"])
            predicted_actions, distances, mask_number = self.model(
                batch["obs_images"].repeat(bimg, 1, 1, 1),
                batch["goal_pose_torch"].repeat(bimg, 1),
                batch["map_images"].repeat(bimg, 1, 1, 1),
                batch["goal_image"],
                modality_id_select.repeat(bimg),
                feat_text_lan.repeat(bimg, 1),
                batch["cur_large_img"].repeat(bimg, 1, 1, 1),
            )

        waypoints = predicted_actions.float().cpu().numpy()

        # ---- Select waypoint and compute velocity ----
        # Use waypoint #4 (5th waypoint) as the control target
        # This is the default in OmniVLA's inference code
        waypoint_select = 4
        chosen_wp = waypoints[0][waypoint_select].copy()
        chosen_wp[:2] *= self.metric_waypoint_spacing
        dx, dy, hx, hy = chosen_wp

        # Calculate correct time horizon for the selected waypoint
        # Waypoint select is 0-indexed, so the 5th waypoint corresponds to (waypoint_select + 1) steps
        horizon_time = (waypoint_select + 1) * self.dt
        linear_vel, angular_vel = self._pd_controller(dx, dy, hx, hy, dt=horizon_time)

        return {
            "linear_vel": linear_vel,
            "angular_vel": angular_vel,
            "waypoints": waypoints[0],
            "selected_waypoint": chosen_wp,
        }

    def _pd_controller(self, dx, dy, hx, hy, dt=None):
        """
        Convert a waypoint (dx, dy, hx, hy) to velocity commands
        using a PD controller (from OmniVLA's original code).

        Args:
            dx: forward displacement
            dy: lateral displacement
            hx: heading cosine
            hy: heading sine
            dt: time horizon for the waypoint (defaults to control period self.dt)

        Returns:
            (linear_vel, angular_vel) tuple
        """
        if dt is None:
            dt = self.dt

        EPS = 1e-8

        if abs(dx) < EPS and abs(dy) < EPS:
            linear_vel = 0.0
            angular_vel = 1.0 * self._clip_angle(math.atan2(hy, hx)) / dt
        elif abs(dx) < EPS:
            linear_vel = 0.0
            angular_vel = 1.0 * np.sign(dy) * math.pi / (2 * dt)
        else:
            linear_vel = dx / dt
            angular_vel = math.atan(dy / dx) / dt

        # Clip to base limits
        linear_vel = np.clip(linear_vel, 0, 0.5)
        angular_vel = np.clip(angular_vel, -1.0, 1.0)

        # Apply velocity limits (from OmniVLA's code)
        linear_vel, angular_vel = self._limit_velocity(
            linear_vel, angular_vel
        )

        return float(linear_vel), float(angular_vel)

    def _limit_velocity(self, v, w):
        """Apply velocity limits preserving the v/w ratio."""
        maxv = self.max_linear
        maxw = self.max_angular

        if abs(v) <= maxv:
            if abs(w) <= maxw:
                return v, w
            else:
                rd = v / w if abs(w) > 1e-8 else 0
                return maxw * np.sign(v) * abs(rd), maxw * np.sign(w)
        else:
            if abs(w) <= 0.001:
                return maxv * np.sign(v), 0.0
            else:
                rd = v / w
                if abs(rd) >= maxv / maxw:
                    return maxv * np.sign(v), maxv * np.sign(w) / abs(rd)
                else:
                    return maxw * np.sign(v) * abs(rd), maxw * np.sign(w)

    @staticmethod
    def _clip_angle(angle):
        """Clip angle to [-pi, pi]."""
        while angle > math.pi:
            angle -= 2 * math.pi
        while angle < -math.pi:
            angle += 2 * math.pi
        return angle

    def reset_context(self):
        """Clear the context queue (call when starting a new episode)."""
        self.context_queue.clear()
        print("[OmniVLA-edge] Context queue reset.")
