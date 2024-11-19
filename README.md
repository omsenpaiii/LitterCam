# GroundingDINO Human Littering Detection

This code utilizes the GroundingDINO model to detect instances of human littering by detecting hands and trash objects in consecutive frames of a video. It calculates the distance between the detected hand and trash objects and compares it to a threshold distance. If the distance exceeds the threshold, it displays the annotated frame showing the detections.

## Prerequisites

- Python 3.x
- OpenCV (cv2) library
- groundingdino library (make sure it is installed)

## Installation

1. Clone the repository or download the code files.

2. Install the required libraries by running the following command:

   ```bash
   pip install opencv-python
   ```

3. Install the groundingdino library by following the instructions in the [groundingdino repository](https://github.com/facebookresearch/groundingdino).

4. Prepare the model and configuration files for the GroundingDINO model and place them in the appropriate directories as specified in the code:
   - `/content/drive/MyDrive/GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py`
   - `/content/drive/MyDrive/GroundingDINO/groundingdino_swint_ogc.pth`

## Usage

1. Place the video file you want to analyze in a folder.

2. Update the `IMAGE_FOLDER_PATH` variable in the code to the path of the folder containing the video frames.

3. Optionally, modify the values of the following variables according to your requirements:
   - `TEXT_PROMPT`: Text prompt used for captioning the image.
   - `BOX_THRESHOLD`: Threshold value for box confidence.
   - `TEXT_THRESHOLD`: Threshold value for text confidence.
   - `distance_threshold`: Threshold distance for determining human littering.

4. Run the code by executing the following command:

   ```bash
   python <filename>.py
   ```

   Replace `<filename>.py` with the name of the Python file containing the code.

5. The annotated frames showing the detected objects and the distance between them will be displayed using `cv2_imshow` if the distance exceeds the threshold.

That's it! You can now use this code to detect instances of human littering by analyzing video frames for the presence of hands and trash objects, and calculating the distance between them.
