# For Mac

import cv2
import os
import glob
from groundingdino.util.inference import load_model, load_image, predict, annotate

# Load the pre-trained GroundingDINO model
CONFIG_PATH = "groundingdino/config/GroundingDINO_SwinT_OGC.py"
WEIGHTS_PATH = "weights/groundingdino_swint_ogc.pth"

model = load_model(CONFIG_PATH, WEIGHTS_PATH)

# Function to extract frames from a video
def FrameCapture(path, output_dir):
    """
    Extract frames from a video and save them to the specified directory.

    Parameters:
        path (str): Path to the input video file.
        output_dir (str): Directory to save extracted frames.
    """
    vidObj = cv2.VideoCapture(path)  # Open the video file
    count = 0  # Frame counter
    success = True  # To track if frames are successfully read

    while success:
        success, image = vidObj.read()  # Read the next frame
        if success:  # If the frame is valid
            # Generate frame filename with zero-padded numbering
            frame_path = os.path.join(output_dir, f'frame{count:04d}.jpg')
            cv2.imwrite(frame_path, image)  # Save the frame as an image
            count += 1  # Increment the frame counter

    vidObj.release()  # Release the video file after processing

# Function to process frames and annotate
def process_video(video_path, output_folder):
    """
    Process the video, extract frames, and save annotated images.
    """
    video_name = os.path.basename(video_path).split('.')[0]  # Get video name without extension
    annotated_folder = os.path.join(output_folder, video_name)
    os.makedirs(annotated_folder, exist_ok=True)  # Create folder for annotated images

    # Extract frames from the video
    frames_folder = os.path.join(output_folder, "frames")
    os.makedirs(frames_folder, exist_ok=True)
    FrameCapture(video_path, frames_folder)

    # Process each frame
    IMAGE_FOLDER_PATH = frames_folder  # Path to folder containing frames
    jpg_files = sorted(glob.glob(IMAGE_FOLDER_PATH + "/*.jpg"))  # Get all frame files sorted by name

    TEXT_PROMPT = "human . cigarette . hand . licenseplate ."
    BOX_THRESHOLD = 0.35  # Confidence threshold for bounding box detection
    TEXT_THRESHOLD = 0.25  # Confidence threshold for text matching

    # Iterate through all frames (except the last one)
    for i in range(len(jpg_files) - 1):
        image_path = jpg_files[i]  # Path to the current frame

        # Load the image and its source for annotation
        image_source, image = load_image(image_path)

        # Predict bounding boxes, confidence scores, and phrases for the current frame
        boxes, logits, phrases = predict(
            model=model,
            image=image,
            caption=TEXT_PROMPT,
            box_threshold=BOX_THRESHOLD,
            text_threshold=TEXT_THRESHOLD
        )

        # Annotate the frame if specific objects are detected
        if "human" in phrases or "cigarette" in phrases or "licenseplate" in phrases or "hand" in phrases:
            annotated_frame = annotate(image_source=image_source, boxes=boxes, logits=logits, phrases=phrases)
            
            # Save the annotated frame to the folder
            annotated_frame_path = os.path.join(annotated_folder, f'annotated_frame{i:04d}.jpg')
            cv2.imwrite(annotated_frame_path, annotated_frame)

    print(f"Annotated frames saved in: {annotated_folder}")

# Path to the input video
video_path = "dataset/cigarette_video.mp4"

# Output folder to store the annotated frames
output_folder = "dataset/annotated_videos"

# Process the video and save annotated frames
process_video(video_path, output_folder)
