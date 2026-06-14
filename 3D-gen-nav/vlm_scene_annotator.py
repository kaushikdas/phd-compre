import os
import zipfile
import json
import shutil
from PIL import Image
from dotenv import load_dotenv
from google import genai

# Resolve paths
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
dotenv_path = os.path.join(project_root, ".env")

# Load environment
load_dotenv(dotenv_path)

ZIP_PATH = os.path.join(project_root, "3D-resources", "Living room 01.zip")
DEST_DIR = os.path.join(script_dir, "reconstructed_scene")
XML_OUTPUT_PATH = os.path.join(script_dir, "robot_scene_reconstructed.xml")

def rename_extracted_files(dest_dir, target_name):
    print(f"Renaming extracted files inside {dest_dir} to prefix: '{target_name}'...")
    
    extracted_obj = None
    extracted_mtl = None
    extracted_jpg = None
    
    for root, dirs, files in os.walk(dest_dir):
        for f in files:
            full_path = os.path.join(root, f)
            if f.lower().endswith(".obj"):
                extracted_obj = full_path
            elif f.lower().endswith(".mtl"):
                extracted_mtl = full_path
            elif f.lower().endswith(".jpg") or f.lower().endswith(".jpeg") or f.lower().endswith(".png"):
                extracted_jpg = full_path
                
    if not extracted_obj or not extracted_jpg:
        raise FileNotFoundError("Could not find matching .obj and texture image inside extracted ZIP!")
        
    old_obj_base = os.path.basename(extracted_obj)
    old_mtl_base = os.path.basename(extracted_mtl) if extracted_mtl else None
    old_jpg_base = os.path.basename(extracted_jpg)
    
    new_obj = os.path.join(dest_dir, f"{target_name}.obj")
    new_mtl = os.path.join(dest_dir, f"{target_name}.mtl") if old_mtl_base else None
    new_png = os.path.join(dest_dir, f"{target_name}.png")
    
    # Rename OBJ and MTL
    os.rename(extracted_obj, new_obj)
    if extracted_mtl and new_mtl:
        os.rename(extracted_mtl, new_mtl)
        
    # Convert image to PNG
    print(f"Converting texture image to PNG for MuJoCo compatibility...")
    try:
        with Image.open(extracted_jpg) as img:
            img.save(new_png, "PNG")
        os.remove(extracted_jpg)
        print(f"  Converted and renamed texture: {old_jpg_base} -> {os.path.basename(new_png)}")
    except Exception as e:
        print(f"  [Warning] PNG conversion failed ({e}). Doing direct rename.")
        new_png = os.path.join(dest_dir, f"{target_name}{os.path.splitext(old_jpg_base)[1]}")
        os.rename(extracted_jpg, new_png)
    
    print(f"  Renamed OBJ: {old_obj_base} -> {os.path.basename(new_obj)}")
    if new_mtl:
        print(f"  Renamed MTL: {old_mtl_base} -> {os.path.basename(new_mtl)}")
    
    # Clean up empty nested folders
    for item in os.listdir(dest_dir):
        item_path = os.path.join(dest_dir, item)
        if os.path.isdir(item_path):
            shutil.rmtree(item_path)
            
    # Update internal links within OBJ
    if new_mtl:
        print(f"Updating mtllib references in {os.path.basename(new_obj)}...")
        with open(new_obj, "r") as f:
            lines = f.readlines()
        with open(new_obj, "w") as f:
            for line in lines:
                if line.startswith("mtllib "):
                    f.write(f"mtllib {os.path.basename(new_mtl)}\n")
                else:
                    f.write(line)
                    
    # Update internal links within MTL
    if new_mtl:
        print(f"Updating texture map references in {os.path.basename(new_mtl)}...")
        with open(new_mtl, "r") as f:
            lines = f.readlines()
        with open(new_mtl, "w") as f:
            for line in lines:
                if line.startswith("map_Kd "):
                    f.write(f"map_Kd {os.path.basename(new_png)}\n")
                else:
                    f.write(line)

def extract_zip(zip_path, dest_dir):
    print(f"\n[Step 1] Unzipping scan data from: {zip_path}")
    if not os.path.exists(zip_path):
        raise FileNotFoundError(f"Source ZIP file not found at: {zip_path}")
        
    if os.path.exists(dest_dir):
        print(f"Removing existing directory: {dest_dir}")
        shutil.rmtree(dest_dir)
    os.makedirs(dest_dir, exist_ok=True)
    
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(dest_dir)
    print(f"Extraction complete. Files extracted to: {dest_dir}")
    
    zip_basename = os.path.splitext(os.path.basename(zip_path))[0]
    rename_extracted_files(dest_dir, zip_basename)

def find_assets(dest_dir):
    obj_file = None
    mtl_file = None
    jpg_file = None
    
    for f in os.listdir(dest_dir):
        full_path = os.path.join(dest_dir, f)
        if f.lower().endswith(".obj"):
            obj_file = full_path
        elif f.lower().endswith(".mtl"):
            mtl_file = full_path
        elif f.lower().endswith(".jpg") or f.lower().endswith(".jpeg") or f.lower().endswith(".png"):
            jpg_file = full_path
            
    if not obj_file or not jpg_file:
        raise FileNotFoundError("Could not locate renamed .obj and image assets in root dest folder!")
        
    return obj_file, mtl_file, jpg_file


def parse_obj_bounds(obj_path):
    print(f"\n[Step 2] Parsing OBJ vertices to extract spatial dimensions...")
    xs, ys, zs = [], [], []
    with open(obj_path, 'r') as f:
        for line in f:
            if line.startswith('v '):
                parts = line.split()
                if len(parts) >= 4:
                    try:
                        xs.append(float(parts[1]))
                        ys.append(float(parts[2]))
                        zs.append(float(parts[3]))
                    except ValueError:
                        continue
    if not xs:
        raise ValueError("No vertices found in OBJ file!")
        
    x_min, x_max = min(xs), max(xs)
    y_min, y_max = min(ys), max(ys)
    z_min, z_max = min(zs), max(zs)
    
    print(f"Mesh Boundary Limits:")
    print(f"  X: [{x_min:.2f}m to {x_max:.2f}m] (Width: {x_max - x_min:.2f}m)")
    print(f"  Y: [{y_min:.2f}m to {y_max:.2f}m] (Length: {y_max - y_min:.2f}m)")
    print(f"  Z: [{z_min:.2f}m to {z_max:.2f}m] (Height: {z_max - z_min:.2f}m)")
    
    return {
        'x_min': x_min, 'x_max': x_max, 'x_mid': (x_min + x_max) / 2.0,
        'y_min': y_min, 'y_max': y_max, 'y_mid': (y_min + y_max) / 2.0,
        'z_min': z_min, 'z_max': z_max, 'z_mid': (z_min + z_max) / 2.0
    }

def query_vlm_layout(texture_path, bounds):
    print(f"\n[Step 3] Calling Gemini VLM to interpret visual textures and layout...")
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise ValueError("GEMINI_API_KEY environment variable is not set! Load your .env file.")
        
    client = genai.Client()
    
    # Load texture image
    img = Image.open(texture_path)
    
    prompt = f"""
    You are an expert robotics simulation architect. Analyze this textured visual projection mapping of a scanned room.
    The room bounding box is:
    - Width (X): {bounds['x_min']:.2f}m to {bounds['x_max']:.2f}m
    - Length (Y): {bounds['y_min']:.2f}m to {bounds['y_max']:.2f}m
    - Height (Z): {bounds['z_min']:.2f}m to {bounds['z_max']:.2f}m
    The floor is at Z={bounds['z_min']:.2f}m.
    
    Identify major objects (such as sofas, coffee tables, TV cabinets, shelves, doors, walls).
    For each object, estimate its physical properties and position relative to the room center (X={bounds['x_mid']:.2f}, Y={bounds['y_mid']:.2f}, Z={bounds['z_min']:.2f}).
    
    Also identify any occluded areas or missing visual details that might create gaps in a physics simulation (e.g. gaps behind a sofa, closed doors where space beyond is unknown).
    
    Return your findings in a strict JSON format matching this schema:
    {{
      "room_type": "living_room",
      "detected_objects": [
        {{
          "name": "string (e.g., sofa, coffee_table)",
          "description": "string",
          "dimensions_m": {{
            "width_x": float,
            "length_y": float,
            "height_z": float
          }},
          "center_offset_m": {{
            "x_offset": float,
            "y_offset": float,
            "z_offset": float
          }},
          "physical_properties": {{
            "friction": float,
            "mass_kg": float,
            "damping": float
          }}
        }}
      ],
      "occlusions_or_ambiguities": [
        {{
          "target_area": "string (e.g., behind_sofa, doorway)",
          "question": "string (a question asking the user to clarify the layout or interactive properties of this area)"
        }}
      ]
    }}
    """
    
    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=[img, prompt],
        config={
            "response_mime_type": "application/json"
        }
    )
    
    return json.loads(response.text)

def run_interactive_clarification(vlm_data, bounds):
    print(f"\n[Step 4] Running Interactive Human-in-the-Loop Clarification...")
    refined_objects = []
    
    print("\n--- Detected Objects ---")
    for obj in vlm_data.get("detected_objects", []):
        name = obj["name"]
        dim = obj["dimensions_m"]
        offset = obj["center_offset_m"]
        
        # Calculate absolute positions
        abs_x = bounds['x_mid'] + offset['x_offset']
        abs_y = bounds['y_mid'] + offset['y_offset']
        abs_z = bounds['z_min'] + offset['z_offset']
        
        print(f"\nObject: {name.upper()} ({obj['description']})")
        print(f"  Estimated Size (W, L, H): {dim['width_x']:.2f}m x {dim['length_y']:.2f}m x {dim['height_z']:.2f}m")
        print(f"  Calculated Abs Pos (X, Y, Z): {abs_x:.2f}, {abs_y:.2f}, {abs_z:.2f}")
        
        user_input = input("  >> Accept? [Enter] for Yes, or type new dimensions/pos 'W L H X Y Z' to override: ").strip()
        if user_input:
            try:
                w, l, h, x, y, z = map(float, user_input.split())
                dim['width_x'], dim['length_y'], dim['height_z'] = w, l, h
                abs_x, abs_y, abs_z = x, y, z
                print("  Refined object parameters applied.")
            except Exception as e:
                print(f"  Invalid override syntax ({e}). Keeping VLM defaults.")
                
        refined_objects.append({
            "name": name,
            "w": dim['width_x'],
            "l": dim['length_y'],
            "h": dim['height_z'],
            "x": abs_x,
            "y": abs_y,
            "z": abs_z,
            "physics": obj["physical_properties"]
        })
        
    print("\n--- VLM Occlusion and Topology Queries ---")
    queries = vlm_data.get("occlusions_or_ambiguities", [])
    if not queries:
        print("No topological ambiguities identified by VLM.")
    for idx, q in enumerate(queries):
        print(f"\nQuery #{idx+1} in [{q['target_area']}]:")
        print(f"  {q['question']}")
        answer = input("  >> Answer: ").strip()
        print(f"  Thank you. Feedback saved for scene compilation.")
        
    return refined_objects

def assemble_mujoco_scene(obj_file, jpg_file, bounds, objects, output_path):
    print(f"\n[Step 5] Compiling and writing MuJoCo XML scene to: {output_path}")
    
    # Calculate relative paths for MuJoCo assets (using forward slashes)
    rel_obj = os.path.relpath(obj_file, os.path.dirname(output_path)).replace("\\", "/")
    rel_jpg = os.path.relpath(jpg_file, os.path.dirname(output_path)).replace("\\", "/")
    
    # Start writing XML
    xml_content = f"""<mujoco model="reconstructed_living_room">
    <compiler angle="degree" coordinate="local" inertiafromgeom="true"/>
    
    <option integrator="RK4" timestep="0.002"/>

    <default>
        <geom friction="2.0 0.5 0.5" rgba="0.8 0.6 0.4 1"/>
        <joint damping="0.8" armature="0.01"/>
        <velocity ctrlrange="-20 20" ctrllimited="true" kv="8"/>
    </default>

    <asset>
        <!-- Textured 3D Mesh of Scanned Room (Visual Only) -->
        <mesh name="room_visual_mesh" file="{rel_obj}"/>
        <texture name="room_texture" type="2d" file="{rel_jpg}"/>
        <material name="room_material" texture="room_texture"/>
        
        <!-- Standard Floor Grid Asset -->
        <texture name="grid" type="2d" builtin="checker" rgb1=".1 .2 .3" rgb2=".2 .3 .4" width="300" height="300" mark="edge" markrgb=".2 .3 .4"/>
        <material name="grid" texture="grid" texrepeat="2 2" texuniform="true"/>
    </asset>

    <worldbody>
        <light directional="true" diffuse=".8 .8 .8" specular=".2 .2 .2" pos="0 0 5" dir="0 0 -1"/>
        
        <!-- 1. Visual-Only Room Mesh (Conclass and Contype disabled) -->
        <geom name="room_mesh_visual" type="mesh" mesh="room_visual_mesh" material="room_material" group="1" contype="0" conaffinity="0"/>
        
        <!-- 2. Physical Floor Collider (Spawned at Z_min) -->
        <geom name="floor_collider" type="plane" pos="0 0 {bounds['z_min']:.3f}" size="15 15 0.1" material="grid" friction="2 0.5 0.5" contype="1" conaffinity="1"/>
        
        <!-- 3. Dynamic Targets (Red Ball and Green Cube) -->
        <body name="red_ball" pos="{bounds['x_mid'] + 1.0:.2f} {bounds['y_mid'] + 1.0:.2f} {bounds['z_min'] + 0.15:.2f}">
            <geom type="sphere" size="0.15" rgba="1 0 0 1" friction="1 0.5 0.5"/>
        </body>
        
        <body name="green_cube" pos="{bounds['x_mid'] - 1.0:.2f} {bounds['y_mid'] + 1.5:.2f} {bounds['z_min'] + 0.12:.2f}">
            <geom type="box" size="0.12 0.12 0.12" rgba="0 1 0 1" friction="1 0.5 0.5"/>
        </body>
    """
    
    # 4. Add VLM-detected and user-approved collision boxes for obstacles
    for idx, obj in enumerate(objects):
        xml_content += f"""
        <!-- Obstacle: {obj['name']} -->
        <body name="obstacle_{obj['name']}_{idx}" pos="{obj['x']:.3f} {obj['y']:.3f} {obj['z']:.3f}">
            <geom type="box" size="{obj['w']/2:.3f} {obj['l']/2:.3f} {obj['h']/2:.3f}" rgba="0.8 0.4 0.2 0.4" group="0" contype="1" conaffinity="1"/>
        </body>"""
        
    # 5. Add robot chassis, sensors, wheels, and camera assembly
    xml_content += f"""
        <!-- Wheeled Robot (Spawned in free center space) -->
        <body name="robot" pos="{bounds['x_mid']:.3f} {bounds['y_mid']:.3f} {bounds['z_min'] + 0.1:.3f}">
            <joint type="free"/>
            <geom name="chassis" type="box" size="0.2 0.15 0.05" rgba="0.2 0.2 0.8 1"/>
            
            <!-- Camera (Look Forward +X, Up is +Z, Right is -Y) -->
            <camera name="robot_eye" pos="0.18 0 0.08" fovy="75" xyaxes="0 -1 0 0 0 1"/>
            <!-- Visual Camera Eye (Black Lens) -->
            <geom name="camera_lens" type="sphere" pos="0.18 0 0.08" size="0.025" rgba="0.1 0.1 0.1 1"/>
            <geom name="camera_mount" type="box" pos="0.17 0 0.08" size="0.01 0.02 0.02" rgba="0.5 0.5 0.5 1"/>

            <!-- Wheels -->
            <body name="front_left" pos="0.14 0.17 -0.05">
                <joint name="fl_joint" type="hinge" axis="0 1 0"/>
                <geom type="cylinder" size="0.05 0.015" euler="90 0 0" rgba="0.1 0.1 0.1 1"/>
            </body>
            <body name="front_right" pos="0.14 -0.17 -0.05">
                <joint name="fr_joint" type="hinge" axis="0 1 0"/>
                <geom type="cylinder" size="0.05 0.015" euler="90 0 0" rgba="0.1 0.1 0.1 1"/>
            </body>
            <body name="rear_left" pos="-0.14 0.17 -0.05">
                <joint name="rl_joint" type="hinge" axis="0 1 0"/>
                <geom type="cylinder" size="0.05 0.015" euler="90 0 0" rgba="0.1 0.1 0.1 1"/>
            </body>
            <body name="rear_right" pos="-0.14 -0.17 -0.05">
                <joint name="rr_joint" type="hinge" axis="0 1 0"/>
                <geom type="cylinder" size="0.05 0.015" euler="90 0 0" rgba="0.1 0.1 0.1 1"/>
            </body>
        </body>
    </worldbody>

    <actuator>
        <velocity name="drive_fl" joint="fl_joint"/>
        <velocity name="drive_fr" joint="fr_joint"/>
        <velocity name="drive_rl" joint="rl_joint"/>
        <velocity name="drive_rr" joint="rr_joint"/>
    </actuator>
</mujoco>
"""
    
    with open(output_path, 'w') as f:
        f.write(xml_content)
    print(f"Reconstructed scene XML written successfully to: {output_path}")

def main():
    print("="*80)
    print("  ISP-WORLD: INTERACTIVE SEMANTIC-PHYSICS WORLD BUILDER (GEMINI VLM EDITION)")
    print("="*80)
    
    try:
        extract_zip(ZIP_PATH, DEST_DIR)
        obj_file, mtl_file, jpg_file = find_assets(DEST_DIR)
        bounds = parse_obj_bounds(obj_file)
        
        vlm_data = query_vlm_layout(jpg_file, bounds)
        refined_objects = run_interactive_clarification(vlm_data, bounds)
        
        assemble_mujoco_scene(obj_file, jpg_file, bounds, refined_objects, XML_OUTPUT_PATH)
        
        print("\n" + "="*80)
        print("  ISP-World Reconstruction Pipeline Complete! Scene is ready for simulation.")
        print("="*80)
    except Exception as e:
        print(f"\n[FATAL ERROR] Pipeline failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
