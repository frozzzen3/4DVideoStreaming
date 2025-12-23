import os
import random

import constriction
import cv2
import numpy as np
import torch
import trimesh
import kaolin
from tqdm import tqdm
import point_cloud_utils as pcu
import math
import torch.nn.functional as F
import open3d as o3d
from skimage.metrics import structural_similarity as ssim
import py7zr
import zipfile
from metrics import compute_D1_psnr, compute_D2_psnr, compute_D1_D2_psnr

def flip_uv_v_inplace(mesh: o3d.geometry.TriangleMesh):
    if not mesh.has_triangle_uvs():
        return
    uvs = np.asarray(mesh.triangle_uvs)
    uvs[:, 1] = 1.0 - uvs[:, 1]
    mesh.triangle_uvs = o3d.utility.Vector2dVector(uvs)

def render_mesh(mesh, width=512, height=512, camera_params=None, enable_lighting=False):
    """
    Render textured RGB + depth using OffscreenRenderer (Open3D 0.19.x).
    Returns: (rgb_uint8 HxWx3, depth_uint8 HxW, (dmin, dmax))
    """
    mesh_copy = mesh

    if enable_lighting and not mesh_copy.has_vertex_normals():
        mesh_copy.compute_vertex_normals()

    # ---- Make a "held" texture image if present ----
    tex_img = None
    if hasattr(mesh_copy, "textures") and len(mesh_copy.textures) > 0:
        # Convert to numpy and re-wrap -> ensures a held instance owned by Python
        tex_np = np.asarray(mesh_copy.textures[0])
        if tex_np is not None and tex_np.size > 0:
            tex_img = o3d.geometry.Image(np.ascontiguousarray(tex_np))

    use_texture = (tex_img is not None) and mesh_copy.has_triangle_uvs()

    renderer = o3d.visualization.rendering.OffscreenRenderer(width, height)
    renderer.scene.set_background([1.0, 1.0, 1.0, 1.0])

    mat = o3d.visualization.rendering.MaterialRecord()
    mat.shader = "defaultLit" if enable_lighting else "defaultUnlit"
    mat.base_color = [1.0, 1.0, 1.0, 1.0]

    if use_texture:
        mat.albedo_img = tex_img
        if enable_lighting:
            mat.roughness = 1.0
            mat.metallic = 0.0

    renderer.scene.add_geometry("mesh", mesh_copy, mat)

    # Camera
    if camera_params is not None:
        intrinsic = camera_params.intrinsic.intrinsic_matrix
        extrinsic = camera_params.extrinsic
        renderer.setup_camera(intrinsic, extrinsic, width, height)
    else:
        center = mesh_copy.get_center()
        eye = center + np.array([0, 0, 2.0])
        renderer.scene.camera.look_at(center, eye, [0, 1, 0])

    rgb_img = np.asarray(renderer.render_to_image(), dtype=np.uint8)
    depth_f = np.asarray(renderer.render_to_depth_image())

    dmin, dmax = float(depth_f.min()), float(depth_f.max())
    if dmax > dmin:
        depth_img = ((depth_f - dmin) / (dmax - dmin) * 255.0).astype(np.uint8)
    else:
        depth_img = np.zeros_like(depth_f, dtype=np.uint8)

    renderer.scene.clear_geometry()
    del renderer

    return rgb_img, depth_img, (dmin, dmax)


def select_viewpoints(mesh, gt_mesh, num_views=4, width=1080, height=1920):
    """
    Interactively select viewpoints for rendering.

    Args:
        mesh (o3d.geometry.TriangleMesh): Mesh to visualize.
        gt_mesh (o3d.geometry.TriangleMesh): Ground truth mesh.
        num_views (int): Number of viewpoints to select.

    Returns:
        list: List of PinholeCameraParameters for selected viewpoints.
    """
    viewpoints = []
    print(f"Select {num_views} viewpoints. Adjust the view, press 'Q' to close the window, then choose to save via console.")

    for i in range(num_views * 2):  # Allow extra attempts
        vis = o3d.visualization.Visualizer()
        vis.create_window(width=width, height=height)
        vis.add_geometry(mesh)
        vis.add_geometry(gt_mesh)
        vis.run()

        # Get current view parameters
        param = vis.get_view_control().convert_to_pinhole_camera_parameters()
        vis.destroy_window()

        # Prompt user to save viewpoint
        response = input(f"Save viewpoint {len(viewpoints)+1}? (y/n): ").strip().lower()
        if response == 'y':
            # Check if viewpoint is unique
            is_unique = True
            for existing_param in viewpoints:
                if np.allclose(param.extrinsic, existing_param.extrinsic, atol=1e-3):
                    print("Viewpoint is too similar to a previous one. Please select a different view.")
                    is_unique = False
                    break
            if is_unique:
                viewpoints.append(param)
                print(f"Viewpoint {len(viewpoints)} saved. Extrinsic: {param.extrinsic}")
        else:
            print(f"Viewpoint skipped. Select another.")

        if len(viewpoints) >= num_views:
            break

    if len(viewpoints) < num_views:
        print(f"Warning: Only {len(viewpoints)} viewpoints saved. Proceeding with available views.")

    return viewpoints[:num_views]

def compute_ssim(img1, img2, multichannel=False):
    """
    Compute SSIM between two images.

    Args:
        img1 (np.ndarray): First image.
        img2 (np.ndarray): Second image.
        multichannel (bool): True for RGB images, False for grayscale.

    Returns:
        float: SSIM score.
    """
    if multichannel:
        score = ssim(img1, img2, channel_axis=2, data_range=255)
    else:
        score = ssim(img1, img2, data_range=255)
    return score

def compute_psnr(img1, img2):
    """
    Compute PSNR between two images.

    Args:
        img1 (np.ndarray): First image.
        img2 (np.ndarray): Second image.

    Returns:
        float: PSNR score.
    """
    mse = np.mean((img1.astype(float) - img2.astype(float)) ** 2)
    if mse == 0:
        return float('inf')
    max_pixel = 255.0
    psnr = 20 * np.log10(max_pixel / np.sqrt(mse))
    return psnr

def evaluate_meshes(gt_mesh, recon_mesh, viewpoints, output_dir="renderings", width=1080, height=1920):
    """
    Evaluate two meshes by rendering normal maps and depth images from multiple viewpoints
    and computing SSIM and PSNR scores.

    Args:
        gt_mesh (o3d.geometry.TriangleMesh): Ground truth mesh.
        recon_mesh (o3d.geometry.TriangleMesh): Reconstructed mesh.
        viewpoints (list): List of PinholeCameraParameters.
        output_dir (str): Directory to save rendered images.

    Returns:
        tuple: (avg_ssim_depth, avg_ssim_normal) - Average SSIM for depth and normal map images.
    """
    # Ensure meshes have vertex normals
    if not gt_mesh.has_vertex_normals():
        gt_mesh.compute_vertex_normals()
    if not recon_mesh.has_vertex_normals():
        recon_mesh.compute_vertex_normals()

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    ssim_scores_depth = []
    ssim_scores_normal = []
    psnr_scores_depth = []
    psnr_scores_normal = []

    # Render and compare for each viewpoint
    for i, view in enumerate(viewpoints):
        # Render both meshes (normal map and depth)
        gt_normal, gt_depth, gt_depth_range = render_mesh(gt_mesh, width=width, height=height, camera_params=view)
        recon_normal, recon_depth, recon_depth_range = render_mesh(recon_mesh, width=width, height=height, camera_params=view)

        # Save renderings
        cv2.imwrite(os.path.join(output_dir, f"gt_view_{i}_rgb.png"), cv2.cvtColor(gt_normal, cv2.COLOR_RGB2BGR))
        cv2.imwrite(os.path.join(output_dir, f"recon_view_{i}_rgb.png"), cv2.cvtColor(recon_normal, cv2.COLOR_RGB2BGR))
        #cv2.imwrite(os.path.join(output_dir, f"gt_view_{i}_depth.png"), gt_depth)
        #cv2.imwrite(os.path.join(output_dir, f"recon_view_{i}_depth.png"), recon_depth)

        # Save debug difference image (normal map)
        #normal_diff = np.abs(gt_normal.astype(float) - recon_normal.astype(float))
        #normal_diff = (normal_diff / normal_diff.max() * 255).astype(np.uint8) if normal_diff.max() > 0 else normal_diff.astype(np.uint8)
        #cv2.imwrite(os.path.join(output_dir, f"view_{i}_normal_diff.png"), cv2.cvtColor(normal_diff, cv2.COLOR_RGB2BGR))

        # Compute SSIM for depth and normal maps
        score_depth = compute_ssim(gt_depth, recon_depth, multichannel=False)
        score_normal = compute_ssim(gt_normal, recon_normal, multichannel=True)
        ssim_scores_depth.append(score_depth)
        ssim_scores_normal.append(score_normal)
        #print(f"View {i+1} - Depth SSIM: {score_depth:.4f}, Normal SSIM: {score_normal:.4f}")

        # Compute PSNR for completeness
        psnr_depth = compute_psnr(gt_depth, recon_depth)
        psnr_normal = compute_psnr(gt_normal, recon_normal)
        psnr_scores_depth.append(psnr_depth)
        psnr_scores_normal.append(psnr_normal)
        #print(f"View {i+1} - Depth PSNR: {psnr_depth:.4f}, Normal PSNR: {psnr_normal:.4f}")

    # Compute average SSIM
    avg_ssim_depth = np.mean(ssim_scores_depth) if ssim_scores_depth else 0
    avg_ssim_normal = np.mean(ssim_scores_normal) if ssim_scores_normal else 0
    #print(f"Average Depth SSIM: {avg_ssim_depth:.4f}")
    #print(f"Average Normal SSIM: {avg_ssim_normal:.4f}")

    avg_psnr_depth = np.mean(psnr_scores_depth) if psnr_scores_depth else 0
    avg_psnr_normal = np.mean(psnr_scores_normal) if psnr_scores_normal else 0
    # print(f"Average Depth PSNR: {avg_psnr_depth:.4f}")
    # print(f"Average Normal PSNR: {avg_psnr_normal:.4f}")

    return avg_ssim_depth, avg_ssim_normal, avg_psnr_depth, avg_psnr_normal

def compute_bitrate(total_bits, num_frames, fps):
    duration_sec = num_frames / fps
    return total_bits / duration_sec / 1000  # kbps

