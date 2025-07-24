import cv2
import os

# Correct path with raw string
image_folder = r'pathtoimagefolder'
video_name = '3d_matplotlib_cube_rotation.mp4'

# Get sorted list of .png files
images = sorted([img for img in os.listdir(image_folder) if img.endswith(".png")])

# Read first frame to get dimensions
frame = cv2.imread(os.path.join(image_folder, images[0]))
height, width, layers = frame.shape

# Define video codec and create VideoWriter object
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
fps = 60  # Set desired frames per second
video = cv2.VideoWriter(video_name, fourcc, fps, (width, height))

# Write each frame
for image in images:
    img_path = os.path.join(image_folder, image)
    video.write(cv2.imread(img_path))

# Cleanup
video.release()
cv2.destroyAllWindows()
