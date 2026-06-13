# Autonomous Multi-Target Navigation (MuJoCo Edition)

## Language-Conditioned Visual Search & High-Fidelity Physics Simulation
*MuJoCo implementation of the OmniVLA + CLIP Navigation Stack*

---

## 1. Introduction: Why MuJoCo?
While the project initially successfully launched in PyBullet, this specialized **MuJoCo Version** was developed to leverage high-fidelity contact dynamics and superior visual rendering. 

This version maintains the same "Intelligence" (CLIP + OmniVLA) as the original but runs it through an isolated, high-performance physics backend. It solves the "Zero-Shot Search" problem: finding and reaching arbitrary objects (like a "red ball") using only natural language commands.

---

## 2. The MuJoCo Robot (MJCF Architecture)
The robot is defined using the **MJCF (MuJoCo Model File)** format in `mujoco_version/robot/robot_scene.xml`. 

### 2.1 Physics & Stability
- **Precision Timestep:** Set to `0.002s` for rock-solid stability during high-speed turns.
- **Damping:** Each wheel joint has `0.8` damping to prevent oscillations.
- **Traction:** Floor and wheel friction are set to `2.0` to ensure precise stopping at targets.

### 2.2 The Camera Assembly ("Robot Eye")
The perception system is anchored by a front-mounted camera with specific robotics alignment:
- **Visual Marker:** A black "lens" sphere and grey mount are visible on the robot's front to distinguish the "front" from the "back" in 3D views.
- **Orientation:** Uses `xyaxes="0 -1 0 0 0 1"`. This explicitly maps:
  - **-Z (Forward)** to World +X.
  - **Y (Up)** to World +Z.
  - **X (Right)** to World -Y.
- **FOV:** A wide `75°` Field of View ensures the robot doesn't easily "lose" targets during high-acceleration maneuvers.

---

## 3. Communication & Logic Flow
The MuJoCo version is isolated in the `mujoco_version/` directory to prevent dependency contamination with the stable PyBullet codebase.

### 3.1 Directory Structure
```text
End Sem project/
  ├── mujoco_version/
  │   ├── simulation_mj/    <-- MuJoCo Environment Logic
  │   ├── robot/            <-- MJCF XML Model
  │   └── main_mj.py        <-- MuJoCo Interactive Loop
  ├── inference/            <-- SHARED AI Logic (CLIP/OmniVLA)
```

### 3.2 The Perpetual Loop
`main_mj.py` implements a persistent session handler. It initializes the model once and waits for natural language commands. It imports the **Shared Brain** (`omnivla_bridge.py`) from the parent directory, ensuring that both PyBullet and MuJoCo robots behave identically.

---

## 4. Semantic Search via CLIP
The robot interprets the world as a **Semantic Radar**. It treats language as a directional vector.

### 4.1 How it works:
1. **Command:** "go to the red ball."
2. **Slicing:** The 320x240 RGB camera feed is sliced into 9 vertical strips.
3. **Similarity:** CLIP compares the text "red ball" to each strip.
4. **Bearing:** The strip with the highest score determines the angular bearing relative to the robot's heading.
   - **Peak Confidence Score:** Used to trigger the shift from "Scanning" to "Navigating."
   - **Early Exit:** During a 360° scan, if a score exceeds **28.5**, the robot stops immediately and locks on.

---

## 5. Navigation via OmniVLA-edge
Once the target is spotted, the **OmniVLA-edge** foundation model takes over:
- **Trajectory:** It predicts high-level `v` (linear velocity) and `w` (angular velocity) based on the current camera view and the target bearing.
- **Scaling:** commands are translated into wheel velocities in the `SimulationEnvironmentMJ` class using a differential drive transform.
- **Safety Buffer:** Upon reaching a target, the robot performs a mandatory 5-frame reverse maneuver to clear its view before searching for the next target.

---

## 6. Real-Time Monitoring
Unlike the standard physics-only view, the MuJoCo version includes a **Robot Eye Monitor** popup (via OpenCV).
- **Function:** Displays the raw egocentric view processed by CLIP.
- **Debugging:** Essential for verifying that the target is centered and not rotated (following the "90-degree CW fix").

---

## 7. Setup & Usage

### 7.1 Dependencies
Required in your `env_aiml` environment:
```bash
pip install mujoco glfw pyopengl opencv-python
```

### 7.2 Running the Simulation
Navigate to the project root and run:
```bash
python ./mujoco_version/main_mj.py
```

### 7.3 Interacting with the View
Inside the MuJoCo window:
- **Mouse Scroll:** Zoom in/out (click window for focus first).
- **Tab:** Toggle Left Settings Panel.
- **Shift + Tab:** Toggle Right Info Panel.
- **Q / ESC:** Close viewer.

---

## 8. Summary of Accomplishments
- **Dual-Engine Support:** Established a parallel workspace that shares a single AI backend.
- **Hardware Parity:** Ported the differential drive dynamics and camera field-of-view from PyBullet to MuJoCo MJCF.
- **Visual Awareness:** Integrated physical markers and live monitor feeds for enhanced user debugging.
- **Robust State Machine:** Verified the "Search-Pivot-Drive-Achieve" pipeline across two different physics kernels.
