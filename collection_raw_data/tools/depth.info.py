import numpy as np
import cv2

def load_and_display_depth(npy_path):
    # Load the depth image (.npy)
    depth = np.load(npy_path)

    # Print information about the depth data
    print(f"Depth shape: {depth.shape}")
    print(f"Depth dtype: {depth.dtype}")
    print(f"Depth min value (excluding 0): {depth[depth > 0].min() if np.any(depth > 0) else 'No valid depths'}")
    print(f"Depth max value: {depth.max()}")
    print(f"Number of valid depth pixels (>0): {np.count_nonzero(depth)}")

    # Normalize depth for visualization (ignore zeros, scale valid depths to 0-255)
    valid_depth = depth.astype(np.float32)
    valid_mask = valid_depth > 0
    if np.any(valid_mask):
        min_val = valid_depth[valid_mask].min()
        max_val = valid_depth[valid_mask].max()
        norm_depth = np.zeros_like(valid_depth)
        norm_depth[valid_mask] = (valid_depth[valid_mask] - min_val) / (max_val - min_val) * 255.0
        norm_depth = norm_depth.astype(np.uint8)
    else:
        print("No valid depth pixels found for visualization.")
        return

    # Apply a color map for better visualization
    depth_colormap = cv2.applyColorMap(norm_depth, cv2.COLORMAP_JET)

    # Display the depth map
    cv2.imshow("Depth Map", depth_colormap)
    print("Press any key on the image window to exit.")
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    npy_file = "dataset/depth/img0008.npy"  # Change to your .npy file path
    load_and_display_depth(npy_file)
