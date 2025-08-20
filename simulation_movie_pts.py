import numpy as np
import matplotlib.pyplot as plt
import rebound
import pandas as pd
import cv2
import os


# # --- Load initial conditions (for binary IDs) ---
# ic = pd.read_parquet("data/init_state_all.pq")

# # --- Loop over each binary group ---
# for binary_id, group in ic.groupby(ic.index, sort=False):
#     archive_path = os.path.join('./archives', str(binary_id), f"{binary_id}.bin")

#     if not os.path.exists(archive_path):
#       continue
      
#     print(f"Creating video for {binary_id}")

#     # Create output folder for frames
#     output_dir = os.path.join('./archives', str(binary_id), "frames")
#     os.makedirs(output_dir, exist_ok=True)

#     # Load archive and create plot
#     sa = rebound.Simulationarchive(archive_path)
#     frames = len(sa)
#     fig, ax = plt.subplots(figsize=(10, 10))
#     sim = sa[0]
#     p0 = sim.particles[0]
#     p1 = sim.particles[1]
#     sep = np.array(p0.xyz) - np.array(p1.xyz)
#     sep = np.linalg.norm(sep)
#     xlim = 2 * sep
#     ylim = 2 * sep

#     for idx in range(frames):
#         sim = sa[idx]
#         ax.clear()

#         ##Graph scatter plot
#         for ii, p in enumerate(sim.particles):
#             if ii == 0:
#                 ax.plot(p.x, p.y, marker='s', color="tab:gray", label='Binary Stars')
#             elif ii == 1:
#                 ax.plot(p.x, p.y, marker='s', color="tab:gray")    
#             else:
#                 ax.plot(p.x, p.y, marker='o', color="red", label='Companion Stars')
#         ax.annotate(f"t={sim.t:.2g}", (0.01, 0.99), va='top', xycoords='axes fraction')
#         ax.legend()
#         ax.set_xlabel('x')
#         ax.set_ylabel('y')
#         ax.set_xlim(-xlim, xlim)
#         ax.set_ylim(-ylim, ylim)

#         frame_path = os.path.join(output_dir, f"frame_{idx:03d}.png")
#         fig.savefig(frame_path)

#     plt.close(fig)

#     image_folder = os.path.join('./archives', str(binary_id),'frames')
#     video_dir = os.path.join('./archives',str(binary_id),'video')
#     os.makedirs(video_dir, exist_ok=True)

#     ##Create path for video
#     output_video_name = os.path.join(video_dir, f'{str(binary_id)}.mp4')
#     ##Frames / second
#     fps = 3

#     images = [img for img in os.listdir(image_folder) if img.endswith(".png")]
#     images.sort()

#     first_image_path = os.path.join(image_folder, images[0])
#     frame = cv2.imread(first_image_path)
#     height, width, layers = frame.shape

#     fourcc = cv2.VideoWriter_fourcc(*'mp4v')
#     video = cv2.VideoWriter(output_video_name, fourcc, fps, (width, height))

#     for image in images:
#         frame_path = os.path.join(image_folder, image)
#         frame = cv2.imread(frame_path)
#         video.write(frame)
#     video.release()    

# print("Done")

import numpy as np
import matplotlib.pyplot as plt
import rebound
import cv2
import os

ARCHIVE_ROOT = r"C:\Users\Admin\starforge_mult_vis\archives"
ARCHIVE_FILENAME = "newtonian.bin"

# --- Build archive path ---
archive_path = os.path.join(ARCHIVE_ROOT, ARCHIVE_FILENAME)

if not os.path.exists(archive_path):
    raise FileNotFoundError(f"Archive not found: {archive_path}")

print(f"Creating video for newtonian")

# Create output folder for frames
output_dir = os.path.join(ARCHIVE_ROOT, "frames")
os.makedirs(output_dir, exist_ok=True)

# Load archive and create plot
sa = rebound.Simulationarchive(archive_path)
frames = len(sa)
fig, ax = plt.subplots(figsize=(10, 10))

# Use first snapshot to set limits
sim = sa[0]
p0 = sim.particles[0]
p1 = sim.particles[1]
sep = np.linalg.norm(np.array(p0.xyz) - np.array(p1.xyz))
xlim = 2 * sep
ylim = 2 * sep

# Loop through archive snapshots
for idx in range(frames):
    sim = sa[idx]
    ax.clear()

    # Graph scatter plot
    for ii, p in enumerate(sim.particles):
        if ii == 0:
            ax.plot(p.x, p.y, marker='s', color="tab:gray", label='Binary Stars')
        elif ii == 1:
            ax.plot(p.x, p.y, marker='s', color="tab:gray")
        else:
            ax.plot(p.x, p.y, marker='o', color="red", label='Companion Stars')
    ax.annotate(f"t={sim.t:.2g}", (0.01, 0.99), va='top', xycoords='axes fraction')
    ax.legend()
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_xlim(-xlim, xlim)
    ax.set_ylim(-ylim, ylim)

    frame_path = os.path.join(output_dir, f"frame_{idx:03d}.png")
    fig.savefig(frame_path)

plt.close(fig)

# Create video folder
video_dir = os.path.join(ARCHIVE_ROOT, 'video')
os.makedirs(video_dir, exist_ok=True)

# Create path for video
output_video_name = os.path.join(video_dir, 'newtonian.mp4')
fps = 3

# Collect frame images
images = [img for img in os.listdir(output_dir) if img.endswith(".png")]
images.sort()

# Get frame size from first image
first_image_path = os.path.join(output_dir, images[0])
frame = cv2.imread(first_image_path)
height, width, layers = frame.shape

fourcc = cv2.VideoWriter_fourcc(*'mp4v')
video = cv2.VideoWriter(output_video_name, fourcc, fps, (width, height))

for image in images:
    frame_path = os.path.join(output_dir, image)
    frame = cv2.imread(frame_path)
    video.write(frame)
video.release()

print("Done")
