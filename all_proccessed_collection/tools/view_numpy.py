import numpy as np
import cv2

# Load a .npy depth file
depth = np.load("dataset/depth/img0010.npy")  # Change filename accordingly

# Normalize and colorize for visualization
depth_vis = cv2.normalize(depth, None, 0, 255, cv2.NORM_MINMAX)
depth_vis = depth_vis.astype(np.uint8)
depth_colored = cv2.applyColorMap(depth_vis, cv2.COLORMAP_JET)

# Show the depth image
cv2.imshow("Depth Visualization", depth_colored)

while True:
    key = cv2.waitKey(100)
    if key == 27 or key == ord('q'):  # ESC or 'q' to quit
        break

cv2.destroyAllWindows()
