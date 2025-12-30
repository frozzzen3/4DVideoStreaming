import os
import shutil
import subprocess
import open3d as o3d

ground_truth_mesh_path = "/media/frozzzen/DataDrive/ChromeDownloads/Dancer_dataset/C4/dancer_fr0001.obj"
compressed_path = "/media/frozzzen/DataDrive/ChromeDownloads/Dancer_dataset/compressed"
os.makedirs(compressed_path, exist_ok=True)

base = os.path.splitext(os.path.basename(ground_truth_mesh_path))[0]  # dancer_fr0001
output_path = os.path.join(compressed_path, f"{base}.drc")
decoded_path = os.path.join(compressed_path, f"{base}_decoded.obj")

# --- Read & preview GT mesh ---
ground_truth_mesh = o3d.io.read_triangle_mesh(ground_truth_mesh_path, enable_post_processing=True)
ground_truth_mesh.compute_vertex_normals()
o3d.visualization.draw_geometries([ground_truth_mesh])

# --- Draco encode ---
result = subprocess.run([
    r"/home/frozzzen/Documents/Github_SINRG/TSMC/draco/build/draco_encoder",
    "-i", ground_truth_mesh_path,
    "-o", output_path,
    "-qp", "14"
], capture_output=True, text=True)
print(result.stdout)
if result.returncode != 0:
    print(result.stderr)
    raise RuntimeError("draco_encoder failed")

# --- Draco decode ---
result = subprocess.run([
    r"/home/frozzzen/Documents/Github_SINRG/TSMC/draco/build/draco_decoder",
    "-i", output_path,
    "-o", decoded_path
], capture_output=True, text=True)
print(result.stdout)
if result.returncode != 0:
    print(result.stderr)
    raise RuntimeError("draco_decoder failed")

# --- Copy .mtl and referenced textures, then re-add mtllib to decoded OBJ ---
src_dir = os.path.dirname(ground_truth_mesh_path)
src_obj_name = os.path.basename(ground_truth_mesh_path)

# Find the mtllib referenced by the original OBJ (fallback to <base>.mtl)
mtl_name = None
with open(ground_truth_mesh_path, "r", encoding="utf-8", errors="ignore") as f:
    for line in f:
        s = line.strip()
        if s.lower().startswith("mtllib "):
            # mtllib can list multiple files; we take the first token after mtllib
            parts = s.split(maxsplit=1)
            if len(parts) == 2:
                # If there are multiple, split them; pick first
                mtl_name = parts[1].split()[0]
            break
if mtl_name is None:
    mtl_name = f"{base}.mtl"

src_mtl_path = os.path.join(src_dir, mtl_name)
dst_mtl_path = os.path.join(compressed_path, os.path.basename(mtl_name))

# Copy MTL
if os.path.isfile(src_mtl_path):
    shutil.copy2(src_mtl_path, dst_mtl_path)
else:
    print(f"[WARN] Could not find MTL next to OBJ: {src_mtl_path}")

# Parse texture filenames from MTL and copy them too
def parse_mtl_textures(mtl_path: str):
    """
    Return a set of texture paths referenced in an MTL file.
    Only copies the file basename; assumes textures live next to the original MTL.
    Handles common map_* entries and ignores options like -s, -o, -bm, etc.
    """
    tex_files = set()
    if not os.path.isfile(mtl_path):
        return tex_files

    map_keys = {
        "map_kd", "map_ka", "map_ks", "map_ke", "map_bump", "bump",
        "map_d", "map_ns", "map_pr", "map_pm", "map_ps", "map_norm", "norm"
    }

    with open(mtl_path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue

            parts = line.split()
            if not parts:
                continue

            key = parts[0].lower()
            if key not in map_keys:
                continue

            tex_path = parts[-1]
            # Remove quotes if present
            tex_path = tex_path.strip("\"'")
            tex_files.add(tex_path)

    return tex_files

'''
textures = parse_mtl_textures(src_mtl_path)
print("textures", textures)
for tex_rel in textures:
    # texture path could be relative
    src_tex_path = os.path.join(src_dir, tex_rel)
    # copy into compressed_path, preserving subdirs if present
    dst_tex_path = os.path.join(compressed_path, tex_rel)

    if os.path.isfile(src_tex_path):
        os.makedirs(os.path.dirname(dst_tex_path), exist_ok=True)
        shutil.copy2(src_tex_path, dst_tex_path)
    else:
        fallback = os.path.join(src_dir, os.path.basename(tex_rel))
        if os.path.isfile(fallback):
            shutil.copy2(fallback, os.path.join(compressed_path, os.path.basename(tex_rel)))
        else:
            print(f"[WARN] Could not find texture referenced in MTL: {tex_rel}")
'''
src_tex_path = "/media/frozzzen/DataDrive/ChromeDownloads/Dancer_dataset/compressed/texture/dancer_fr0001.png"
dst_tex_path = "/media/frozzzen/DataDrive/ChromeDownloads/Dancer_dataset/compressed/dancer_fr0001.png"

if os.path.isfile(src_tex_path):
    os.makedirs(os.path.dirname(dst_tex_path), exist_ok=True)
    shutil.copy2(src_tex_path, dst_tex_path)


# Re-add mtllib line into decoded OBJ (and remove any existing mtllib to avoid duplicates)
def ensure_mtllib(obj_path: str, mtl_file_name: str):
    with open(obj_path, "r", encoding="utf-8", errors="ignore") as f:
        lines = f.readlines()

    new_lines = []
    for ln in lines:
        if ln.strip().lower().startswith("mtllib "):
            continue
        new_lines.append(ln)

    # Insert mtllib near the top (after initial comments if any)
    insert_idx = 0
    while insert_idx < len(new_lines) and new_lines[insert_idx].lstrip().startswith("#"):
        insert_idx += 1

    new_lines.insert(insert_idx, f"mtllib {mtl_file_name}\n")

    with open(obj_path, "w", encoding="utf-8") as f:
        f.writelines(new_lines)

if os.path.isfile(dst_mtl_path):
    ensure_mtllib(decoded_path, os.path.basename(dst_mtl_path))
else:
    print("[WARN] Skipping mtllib injection because MTL was not copied/found.")

print(f"Decoded OBJ: {decoded_path}")
print(f"Copied MTL : {dst_mtl_path}")

decoded_mesh = o3d.io.read_triangle_mesh(decoded_path, enable_post_processing=True)
decoded_mesh.compute_vertex_normals()
o3d.visualization.draw_geometries([decoded_mesh])
