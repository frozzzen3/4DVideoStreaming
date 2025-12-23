import os
import shutil
import subprocess
import open3d as o3d
from util import select_viewpoints, evaluate_meshes, compute_bitrate, flip_uv_v_inplace
from metrics import compute_D1_psnr, compute_D2_psnr, compute_D1_D2_psnr

ground_truth_mesh_path = "/media/frozzzen/DataDrive/ChromeDownloads/Dancer_dataset/C4/dancer_fr0001.obj"
compressed_path = "/media/frozzzen/DataDrive/ChromeDownloads/Dancer_dataset/compressed"
os.makedirs(compressed_path, exist_ok=True)

base = os.path.splitext(os.path.basename(ground_truth_mesh_path))[0]  # dancer_fr0001
output_path = os.path.join(compressed_path, f"{base}.drc")
decoded_path = os.path.join(compressed_path, f"{base}_decoded.obj")

# --- Read & preview GT mesh ---
ground_truth_mesh = o3d.io.read_triangle_mesh(ground_truth_mesh_path, enable_post_processing=True)
ground_truth_mesh.compute_vertex_normals()
#o3d.visualization.draw_geometries([ground_truth_mesh])

decoded_mesh = o3d.io.read_triangle_mesh(decoded_path, enable_post_processing=True)
decoded_mesh.compute_vertex_normals()
#o3d.visualization.draw_geometries([decoded_mesh])

# select viewpoints manually, set the number of viewpoints (num_views) then select viewpoints in open3d viewer.
out_dir = "/media/frozzzen/DataDrive/ChromeDownloads/Dancer_dataset/render"
os.makedirs(out_dir, exist_ok=True)

view_files_exist = all(os.path.exists(f"{out_dir}/view_{i:02d}.json") for i in range(4))
num_views = 4
if not (view_files_exist):
    viewpoints = select_viewpoints(decoded_mesh, ground_truth_mesh, num_views=num_views, width=1080, height=1920)
    for i, cam in enumerate(viewpoints):
        o3d.io.write_pinhole_camera_parameters(f"{out_dir}/view_{i:02d}.json", cam)
else:
    viewpoints = [o3d.io.read_pinhole_camera_parameters(f"{out_dir}/view_{i:02d}.json") for i in range(num_views)]

flip_uv_v_inplace(ground_truth_mesh)
flip_uv_v_inplace(decoded_mesh)

avg_ssim_depth, avg_ssim_color, avg_psnr_depth, avg_psnr_normal = evaluate_meshes(ground_truth_mesh, decoded_mesh, viewpoints, output_dir=os.path.join(out_dir, "renderings"), width=1080, height=1920)

print(f"  SSIM Depth: {avg_ssim_depth:.4f}")
print(f"  SSIM Color: {avg_ssim_color:.4f}")
print(f"  PSNR Depth: {avg_psnr_depth:.3f}")
print(f"  PSNR Color: {avg_psnr_normal:.3f}\n")

d1_a, d2_a = compute_D1_D2_psnr(ground_truth_mesh, decoded_mesh)
d1_b, d2_b = compute_D1_D2_psnr(decoded_mesh, ground_truth_mesh)

d1 = max(d1_a, d1_b)
d2 = max(d2_a, d2_b)

print(f"  D1 PSNR: {d1:.4f}")
print(f"  D2 PSNR: {d2:.4f}")