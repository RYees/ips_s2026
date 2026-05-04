import open3d as o3d

def view_ply():
    ply_path = "dataset/pointcloud/img0000.ply"  # Change this to your .ply file path
    pcd = o3d.io.read_point_cloud(ply_path)
    if pcd.is_empty():
        print(f"Failed to load point cloud or file is empty: {ply_path}")
        return
    o3d.visualization.draw_geometries([pcd])

if __name__ == "__main__":
    view_ply()
