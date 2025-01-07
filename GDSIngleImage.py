// For Mac

from groundingdino.util.inference import load_model, load_image, predict, annotate
import cv2
import os

# Paths to configuration and weights
CONFIG_PATH = "groundingdino/config/GroundingDINO_SwinT_OGC.py"
WEIGHTS_PATH = "weights/groundingdino_swint_ogc.pth"

# Load the GroundingDINO model
My_GD_Model = load_model(CONFIG_PATH, WEIGHTS_PATH)

# Parameters for the cigarette image
IMAGE_PATH = "dataset/cigarettebutt_in_grass.png"
TEXT_PROMPT = "cigarette_butt ."
BOX_THRESHOLD = 0.35
TEXT_THRESHOLD = 0.25

# Load the cigarette image
image_source, image = load_image(IMAGE_PATH)

# Run prediction
boxes, logits, phrases = predict(
    model=My_GD_Model,
    device="mps",
    image=image,
    caption=TEXT_PROMPT,
    box_threshold=BOX_THRESHOLD,
    text_threshold=TEXT_THRESHOLD,
)

# Print results
print(f"Results for Image: {IMAGE_PATH}")
print("Boxes:", boxes)
print("Logits:", logits)
print("Phrases:", phrases)

# Annotate the image
annotated_frame = annotate(
    image_source=image_source,
    boxes=boxes,
    logits=logits,
    phrases=phrases,
)

# Ensure the 'annotated_images' folder exists
output_folder = "dataset/annotated_images"
os.makedirs(output_folder, exist_ok=True)

# Define the output path for the annotated image
output_path = os.path.join(output_folder, "annotated_cigarette_image.jpg")

# Save the annotated image
cv2.imwrite(output_path, annotated_frame)
print(f"Annotated image saved as: {output_path}")
