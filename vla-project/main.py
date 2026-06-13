import os
import time
import math
import random
import argparse
import numpy as np
from PIL import Image

# Windows-specific fix for OpenMP duplication
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# Absolute paths
project_root = os.path.dirname(os.path.abspath(__file__))

from simulation.sim_env import SimulationEnvironment

def run_simulation(gui=True, initial_command=None, max_steps=1000):
    print("\n" + "="*70)
    print("  OMNIVLA PERSISTENT INTERACTIVE SESSION")
    print("="*70)

    print("\n[Phase 1] Creating simulation environment...")
    env = SimulationEnvironment(gui=gui)
    
    print("\n[Phase 2] Initializing OmniVLA-edge & CLIP Bridge...")
    # Delay import to show Phase 1 GUI immediately
    from inference.omnivla_bridge import OmniVLAEdgeBridge
    bridge = OmniVLAEdgeBridge(model_weights_dir="omnivla-edge")
    
    # ---- State Definitions ----
    STATE_NAVIGATING = "NAVIGATING"
    STATE_ACHIEVED = "ACHIEVED"
    STATE_SCANNING = "SCANNING"
    STATE_P_ROTATING = "P_ROTATING"
    STATE_RELOCATING = "RELOCATING"

    last_command = initial_command
    
    while True:
        # ---- Step 1: Command Acquisition ----
        if last_command is None:
            print("\n" + "-"*70)
            print("  [IDLE] Waiting for instruction...")
            print("  Options: 'go to the red ball', 'first green cube then red ball', 'exit'")
            try:
                user_input = input(">> Enter command: ").strip()
            except EOFError:
                break
                
            if user_input.lower() in ['exit', 'quit', 'q']:
                print("\n[EXIT] Shutting down simulation...")
                break
            if not user_input:
                # Small step to keep GUI alive
                env.step_simulation(num_steps=10)
                continue
            current_command = user_input
        else:
            current_command = last_command
            last_command = None 

        # ---- Step 2: Mission Setup ----
        target_sequence = []
        cmd_lower = current_command.lower()
        
        # Simple parser for multiple targets
        if "red" in cmd_lower or "ball" in cmd_lower: 
            target_sequence.append("red_ball")
        if "green" in cmd_lower or "cube" in cmd_lower:
            target_sequence.append("green_cube")
            
        # Handle logic for "first/then" sequencing
        if "first" in cmd_lower and "green" in cmd_lower and "red" in cmd_lower:
            if cmd_lower.find("green") < cmd_lower.find("red"):
                target_sequence = ["green_cube", "red_ball"]
            else:
                target_sequence = ["red_ball", "green_cube"]

        if not target_sequence:
            print(f"  [WARN] No recognizable targets in: '{current_command}'")
            continue

        print(f"\n[NEW MISSION] Mission: '{current_command}'")
        print(f"  Plan: {' -> '.join(target_sequence)}")
        
        # Reset Mission Parameters
        target_idx = 0
        search_attempts = 0
        current_state = STATE_NAVIGATING
        bridge.reset_context()
        conf_history = []
        scan_steps = 0
        best_conf = 0
        static_verify_count = 0
        
        # Thresholds
        CONFIDENCE_FIND = 27.2
        CONFIDENCE_SURE = 28.5 
        CONFIDENCE_KEEP = 23.0
        MAX_VERIFY = 5 
        stop_distance = 0.7

        # ---- Step 3: Mission Execution Loop ----
        mission_active = True
        mission_step = 0
        
        while mission_active:
            target_object = target_sequence[target_idx]
            camera_image = env.get_camera_image()
            robot_pos = env.get_robot_position()
            robot_yaw = env.get_robot_orientation()
            dist_to_target = env.get_distance_to_object(target_object)

            # 1. Semantic Awareness
            bearing, raw_confidence = bridge.semantic_search(camera_image, f"go to the {target_object.replace('_', ' ')}")
            
            # Smoothing (3-frame MA)
            conf_history.append(raw_confidence)
            if len(conf_history) > 3: conf_history.pop(0)
            confidence = sum(conf_history) / len(conf_history)
            
            if mission_step == 0:
                print(f"  [INIT] Perception of {target_object}: Confidence {confidence:.1f}")
                # Use FIND threshold at startup to be skeptical
                if confidence < CONFIDENCE_FIND:
                    print(f"  [INIT] Low confidence ({confidence:.1f}). Starting search...")
                    current_state = STATE_SCANNING
                    scan_steps = 0
                    best_conf = 0
                    conf_history = []

            msg = ""

            # 2. State Logic
            if current_state == STATE_NAVIGATING:
                if dist_to_target < stop_distance:
                    print(f"\n  [STRIKE] Reached target: {target_object}!")
                    current_state = STATE_ACHIEVED
                    clear_steps = 0
                    conf_history = []
                    continue
                
                if confidence > CONFIDENCE_KEEP:
                    visual_yaw = robot_yaw + bearing
                    est_target = np.array([
                        robot_pos[0] + 2.0 * math.cos(visual_yaw),
                        robot_pos[1] + 2.0 * math.sin(visual_yaw)
                    ])
                    result = bridge.predict(camera_image, f"go to {target_object}", 
                                         robot_position=robot_pos[:2], robot_yaw=robot_yaw, 
                                         target_position=est_target)
                    env.apply_velocity(result["linear_vel"], result["angular_vel"])
                    msg = f"Tracking {target_object} (Conf: {confidence:.1f}, Dist: {dist_to_target:.1f}m)"
                else:
                    print(f"\n  [LOST] {target_object} not seen. Starting scan...")
                    current_state = STATE_SCANNING
                    scan_steps = 0
                    best_conf = 0
                    conf_history = []
                    continue

            elif current_state == STATE_ACHIEVED:
                env.apply_velocity(linear_vel=-0.15, angular_vel=0.1)
                clear_steps += 1
                msg = f"Clearing viewpoint... ({clear_steps}/5)"
                if clear_steps >= 5:
                    env.stop_robot() # Stop after every clear
                    target_idx += 1
                    if target_idx >= len(target_sequence):
                        print(f"  [MISSION COMPLETE] All targets reached.")
                        mission_active = False
                        continue
                    
                    print(f"\n[NEXT] Searching for target {target_idx+1}: {target_sequence[target_idx]}")
                    current_state = STATE_SCANNING
                    scan_steps = 0
                    best_conf = 0
                    conf_history = []
                    mission_step = -1

            elif current_state == STATE_SCANNING:
                env.apply_velocity(linear_vel=0.0, angular_vel=0.5)
                scan_steps += 1
                if confidence > best_conf:
                    best_conf = confidence
                    best_bearing = robot_yaw + bearing 
                
                msg = f"Searching for {target_object} ({scan_steps}/42, Peak: {best_conf:.1f})"
                
                if confidence > CONFIDENCE_SURE:
                    print(f"\n  [SPOT] High confidence detection ({confidence:.1f})!")
                    best_bearing = robot_yaw + bearing
                    current_state = STATE_P_ROTATING
                    static_verify_count = 0
                    continue

                if scan_steps >= 42:
                    if best_conf > CONFIDENCE_FIND:
                        print(f"\n  [FOUND] Target identified (Peak: {best_conf:.1f}).")
                        current_state = STATE_P_ROTATING
                        static_verify_count = 0
                    else:
                        search_attempts += 1
                        if search_attempts >= 5:
                            print(f"\n  [ERROR] Mission failed. {target_object} not found.")
                            env.stop_robot()
                            mission_active = False
                            continue
                        
                        dirs = ["FRONT", "BACK", "LEFT", "RIGHT", "RANDOM"]
                        print(f"\n  [RETRY] Relocating to {dirs[search_attempts-1]}...")
                        current_state = STATE_RELOCATING
                        relocate_steps = 0

            elif current_state == STATE_P_ROTATING:
                diff = best_bearing - robot_yaw
                while diff > math.pi: diff -= 2*math.pi
                while diff < -math.pi: diff += 2*math.pi
                
                if abs(diff) < 0.1:
                    env.stop_robot()
                    static_verify_count += 1
                    msg = f"Verifying... {static_verify_count}/5"
                    if static_verify_count >= 5:
                        if confidence > CONFIDENCE_KEEP:
                            print(f"\n  [LOCKED] Engaging {target_object}.")
                            current_state = STATE_NAVIGATING
                        else:
                            print(f"\n  [GLITCH] False detection. Resuming scan...")
                            current_state = STATE_SCANNING
                            scan_steps = 0
                else:
                    w = np.clip(1.5 * diff, -0.6, 0.6)
                    env.apply_velocity(linear_vel=0.0, angular_vel=w)
                    msg = f"Aligning... (Err: {math.degrees(diff):.1f}deg)"

            elif current_state == STATE_RELOCATING:
                v_map = [(0.3, 0), (-0.3, 0), (0.2, 0.5), (0.2, -1.0), (0.2, 0.4)]
                v, w = v_map[search_attempts-1]
                env.apply_velocity(linear_vel=v, angular_vel=w) 
                relocate_steps += 1
                msg = f"Moving... {relocate_steps}/15"
                if relocate_steps >= 15:
                    env.stop_robot()
                    current_state = STATE_SCANNING
                    scan_steps = 0

            # Physics Management
            if msg: print(f"    Mission Step {mission_step:3d} | {msg}", end="\r")
            
            # Step the simulation
            if gui:
                # 80 physics steps per 1/3 sec control
                for _ in range(int((1/3)/0.00416)): # 240Hz
                    env.step_simulation()
                time.sleep(0.01) # Small GUI sleep
            else:
                env.step_simulation(num_steps=80)
                
            mission_step += 1
            if mission_step > max_steps:
                print("\n  [ABORT] Mission timed out.")
                env.stop_robot()
                mission_active = False

    env.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--command", type=str, default=None)
    parser.add_argument("--no-gui", action="store_true")
    args = parser.parse_args()

    run_simulation(
        gui=not args.no_gui,
        initial_command=args.command
    )
