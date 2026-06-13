import os
import sys
import time
import math
import argparse
import numpy as np
from PIL import Image

# Add parent directory to sys.path so we can import 'inference'
curr_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(curr_dir)
if parent_dir not in sys.path:
    sys.path.append(parent_dir)

# Now it is safe to import shared modules
try:
    from simulation_mj.sim_env_mj import SimulationEnvironmentMJ
    from inference.omnivla_bridge import OmniVLAEdgeBridge
except ImportError as e:
    print(f"[Error] Could not find required modules: {e}")
    sys.exit(1)

# Windows-specific fix
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

try:
    import cv2
    HAS_CV2 = True
except ImportError:
    HAS_CV2 = False

def run_simulation(gui=True, initial_command=None, max_steps=1000):
    print("\n" + "="*70)
    print("  OMNIVLA - MUJOCO PERSISTENT SESSION")
    print("="*70)

    print("\n[Phase 1] Creating MuJoCo environment...")
    env = SimulationEnvironmentMJ(gui=gui)
    
    print("\n[Phase 2] Initializing OmniVLA-edge & CLIP Bridge...")
    bridge = OmniVLAEdgeBridge(model_weights_dir=os.path.join(parent_dir, "omnivla-edge"))
    
    # ---- State Definitions ----
    STATE_NAVIGATING = "NAVIGATING"
    STATE_ACHIEVED = "ACHIEVED"
    STATE_SCANNING = "SCANNING"
    STATE_P_ROTATING = "P_ROTATING"
    STATE_RELOCATING = "RELOCATING"

    last_command = initial_command
    
    while True:
        if last_command is None:
            print("\n" + "-"*70)
            print("  [IDLE] Waiting for MuJoCo mission...")
            try:
                user_input = input(">> Enter command: ").strip()
            except EOFError: break
                
            if user_input.lower() in ['exit', 'quit', 'q']:
                break
            if not user_input:
                env.step_simulation(num_steps=10)
                continue
            current_command = user_input
        else:
            current_command = last_command
            last_command = None 

        target_sequence = []
        cmd_lower = current_command.lower()
        if "red" in cmd_lower or "ball" in cmd_lower: target_sequence.append("red_ball")
        if "green" in cmd_lower or "cube" in cmd_lower: target_sequence.append("green_cube")
            
        if "first" in cmd_lower and "green" in cmd_lower and "red" in cmd_lower:
            target_sequence = ["green_cube", "red_ball"] if cmd_lower.find("green") < cmd_lower.find("red") else ["red_ball", "green_cube"]

        if not target_sequence:
            print(f"  [WARN] No targets in: '{current_command}'")
            continue

        print(f"\n[NEW MISSION] Plan: {' -> '.join(target_sequence)}")
        
        target_idx = 0
        search_attempts = 0
        current_state = STATE_NAVIGATING
        bridge.reset_context()
        conf_history = []
        scan_steps = 0
        best_conf = 0
        static_verify_count = 0
        
        CONFIDENCE_FIND = 27.2
        CONFIDENCE_SURE = 28.5 
        CONFIDENCE_KEEP = 23.0
        stop_distance = 0.75 # Slightly larger for MuJoCo stability

        mission_active = True
        mission_step = 0
        
        while mission_active:
            target_object = target_sequence[target_idx]
            camera_image = env.get_camera_image()
            
            # Show Camera Feed in Window
            if HAS_CV2:
                cv_img = cv2.cvtColor(np.array(camera_image), cv2.COLOR_RGB2BGR)
                cv2.imshow("Robot Eye Monitor", cv_img)
                if cv2.waitKey(1) & 0xFF == ord('q'): mission_active = False

            robot_pos = env.get_robot_position()
            robot_yaw = env.get_robot_orientation()
            dist_to_target = env.get_distance_to_object(target_object)

            bearing, raw_confidence = bridge.semantic_search(camera_image, f"go to the {target_object.replace('_', ' ')}")
            
            conf_history.append(raw_confidence)
            if len(conf_history) > 3: conf_history.pop(0)
            confidence = sum(conf_history) / len(conf_history)
            
            if mission_step == 0:
                print(f"  [INIT] Confidence: {confidence:.1f}")
                if confidence < CONFIDENCE_FIND:
                    current_state = STATE_SCANNING
                    scan_steps = 0
                    best_conf = 0

            msg = ""

            if current_state == STATE_NAVIGATING:
                if dist_to_target < stop_distance:
                    print(f"\n  [STRIKE] Reached {target_object}!")
                    current_state = STATE_ACHIEVED
                    clear_steps = 0
                    continue
                
                if confidence > CONFIDENCE_KEEP:
                    visual_yaw = robot_yaw + bearing
                    est_target = np.array([robot_pos[0] + 2.0 * math.cos(visual_yaw), robot_pos[1] + 2.0 * math.sin(visual_yaw)])
                    result = bridge.predict(camera_image, f"go to {target_object}", 
                                         robot_position=robot_pos[:2], robot_yaw=robot_yaw, 
                                         target_position=est_target)
                    env.apply_velocity(result["linear_vel"], result["angular_vel"])
                    msg = f"Tracking (Conf: {confidence:.1f}, Dist: {dist_to_target:.1f}m)"
                else:
                    print(f"\n  [LOST] {target_object} dropped. Scanning...")
                    current_state = STATE_SCANNING
                    scan_steps = 0
                    continue

            elif current_state == STATE_ACHIEVED:
                env.apply_velocity(linear_vel=-0.15, angular_vel=0.1)
                clear_steps += 1
                msg = f"Clearing... ({clear_steps}/5)"
                if clear_steps >= 5:
                    env.stop_robot()
                    target_idx += 1
                    if target_idx >= len(target_sequence):
                        print(f"  [SUCCESS] Mission complete.")
                        mission_active = False
                        continue
                    current_state = STATE_SCANNING
                    scan_steps = 0
                    mission_step = -1

            elif current_state == STATE_SCANNING:
                env.apply_velocity(linear_vel=0.0, angular_vel=0.5)
                scan_steps += 1
                if confidence > best_conf:
                    best_conf, best_bearing = confidence, robot_yaw + bearing 
                
                msg = f"Scanning ({scan_steps}/42, Peak: {best_conf:.1f})"
                
                if confidence > CONFIDENCE_SURE: # Early Exit
                    print(f"\n  [SPOT] Found at {confidence:.1f}!")
                    best_bearing = robot_yaw + bearing
                    current_state = STATE_P_ROTATING
                    static_verify_count = 0
                    continue

                if scan_steps >= 42:
                    if best_conf > CONFIDENCE_FIND:
                        current_state = STATE_P_ROTATING
                        static_verify_count = 0
                    else:
                        search_attempts += 1
                        if search_attempts >= 5:
                            mission_active = False
                            continue
                        current_state = STATE_RELOCATING
                        relocate_steps = 0

            elif current_state == STATE_P_ROTATING:
                diff = best_bearing - robot_yaw
                while diff > math.pi: diff -= 2*math.pi
                while diff < -math.pi: diff += 2*math.pi
                
                if abs(diff) < 0.1:
                    env.stop_robot()
                    static_verify_count += 1
                    msg = f"Confirming... {static_verify_count}/5"
                    if static_verify_count >= 5:
                        if confidence > CONFIDENCE_KEEP:
                            current_state = STATE_NAVIGATING
                        else:
                            current_state = STATE_SCANNING
                            scan_steps = 0
                else:
                    w = np.clip(1.5 * diff, -1.0, 1.0) # More aggressive turn for MuJoCo
                    env.apply_velocity(linear_vel=0.0, angular_vel=w)
                    msg = f"Pivoting... (Err: {math.degrees(diff):.1f}deg)"

            elif current_state == STATE_RELOCATING:
                v_map = [(0.3, 0), (-0.3, 0), (0.2, 0.5), (0.2, -1.0), (0.2, 0.4)]
                v, w = v_map[search_attempts-1]
                env.apply_velocity(linear_vel=v, angular_vel=w) 
                relocate_steps += 1
                msg = f"Relocating... ({relocate_steps}/15)"
                if relocate_steps >= 15:
                    env.stop_robot()
                    current_state = STATE_SCANNING
                    scan_steps = 0

            if msg: print(f"    Step {mission_step:3d} | {msg}", end="\r")
            env.step_simulation()
            mission_step += 1
            if mission_step > max_steps:
                mission_active = False

    print("\n[FINISH] Cleaning up MuJoCo resources...")
    env.close()
    if HAS_CV2:
        cv2.destroyAllWindows()
        for i in range(10): cv2.waitKey(1) # Final flush for OpenCV
    
    print("Done. Goodbye!")
    sys.exit(0)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--command", type=str, default=None)
    parser.add_argument("--no-gui", action="store_true")
    args = parser.parse_args()

    run_simulation(
        gui=not args.no_gui,
        initial_command=args.command
    )
