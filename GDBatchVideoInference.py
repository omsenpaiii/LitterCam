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
    frames_folder = os.path.join(output_folder, "frames", video_name)
    os.makedirs(frames_folder, exist_ok=True)
    FrameCapture(video_path, frames_folder)

    # Process each frame
    jpg_files = sorted(glob.glob(frames_folder + "/*.jpg"))  # Get all frame files sorted by name

    TEXT_PROMPT = "cigarette_butt ."
    BOX_THRESHOLD = 0.35  # Confidence threshold for bounding box detection
    TEXT_THRESHOLD = 0.25  # Confidence threshold for text matching

    for i, image_path in enumerate(jpg_files):
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

    print(f"Annotated frames for video '{video_name}' saved in: {annotated_folder}")

# Main function to process all videos in the dataset/videos directory
def process_all_videos(input_folder, output_folder):
    """
    Process all videos in the input folder and save annotated frames in the output folder.

    Parameters:
        input_folder (str): Path to the folder containing video files.
        output_folder (str): Path to the folder to save annotated videos.
    """
    video_files = glob.glob(os.path.join(input_folder, "*.mp4"))  # Adjust extension if needed
    print(f"Found {len(video_files)} videos to process.")

    for video_path in video_files:
        print(f"Processing video: {video_path}")
        process_video(video_path, output_folder)

# Define paths
input_folder = "dataset/videos"  # Folder containing videos
output_folder = "dataset/annotated_videos"  # Folder to save annotated frames

# Process all videos
os.makedirs(output_folder, exist_ok=True)
process_all_videos(input_folder, output_folder)
