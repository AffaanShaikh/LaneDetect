# LaneDetect
Road lane line detection pipeline implemented in Python using various image processing techniques and the OpenCV library.
It is designed to analyze images or video frames to identify and annotate lane lines on roads.

# Pipeline Steps:

Region of Interest (ROI): The script defines a region of interest on the road image. This step helps to focus only on the relevant part of the image where the lanes are expected to appear.

Color Correction: Applies color correction techniques like histogram equalization and Contrast Limited Adaptive Histogram Equalization (CLAHE) to enhance the contrast and visibility of the lanes.

Edge Detection: The Canny edge detection algorithm is used to find edges in the image. This is a crucial step in identifying lane lines.

Color Masking: Applies color masking to filter out specific colors (white and yellow) which are characteristic of lane lines. This helps in isolating the lanes from the rest of the image.

Hough Transform: The probabilistic Hough Transform is applied to detect straight lines in the image. This step identifies potential lane lines.

Filtering and Outlier Removal: The detected lines are filtered based on their slopes to remove outliers. This ensures that only lines with appropriate slopes are considered as lane lines.

Lane Line Fitting: The script fits two polynomial functions (one for the left lane and one for the right lane) to the detected lines.

Drawing Lanes: The lane lines are drawn on a blank image and overlaid on the original image.

# Output examples:

![test1](https://github.com/AffaanShaikh/LaneDetect/assets/130907730/278368ac-6e90-43c7-be96-5f2704922160)

![test4](https://github.com/AffaanShaikh/LaneDetect/assets/130907730/496e40e1-6c74-43b9-ad32-57f1508f1fff)

# Applications:
This lane detection script holds significant value in the realm of autonomous driving and advanced driver assistance systems (ADAS). It can serve as a foundational component for vehicles to autonomously navigate within lanes, contributing to enhanced road safety. Additionally, the script finds relevance in lane departure warning systems, alerting drivers when they veer off course. Beyond individual vehicles, it can be employed in traffic monitoring setups to assess lane discipline and traffic flow in real-time. Furthermore, the script's versatility makes it a valuable tool for research and development in the field of computer vision, providing a starting point for more complex lane detection algorithms and systems.
