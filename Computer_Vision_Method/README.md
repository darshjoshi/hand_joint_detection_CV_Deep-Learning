# Finger Joint Detection

Executed By: [Darsh Joshi](https://www.linkedin.com/in/darshjoshi/)

## Overview
This Python script detects finger joints in a given image using OpenCV and scikit-learn. It allows users to manually select a region of interest, applies various image processing techniques, detects contours and their centers, and uses KMeans clustering to identify the joint centers.

## Prerequisites
- Python 3.8 or higher 
- Tkinter (usually comes with Python)

## Installation
1. Clone the repository or download the script.
2. Install the required packages from the `requirements.txt` file:

```bash
pip install -r requirements.txt
```


## Usage
1. Run the script:

```bash
main.py
```
2. A file dialog will open. Select the image in which you want to detect finger joints.
3. Follow the on-screen instructions to process the image and detect joints.

## Functions
- `select_roi`: Manually select a region of interest in the image.
- `apply_gaussian_blur`, `apply_gamma_correction`, `apply_clahe`: Apply various image processing techniques.
- `apply_canny_edge_detector`: Detect edges in the image.
- `find_and_draw_contours`: Find and draw contours on the image.
- `find_joint_centers_and_cluster`: Identify joint centers using KMeans clustering.

## Output
- The script displays intermediate steps and the final result with detected joint centers.
- The coordinates of the joint centers are saved to `contour_centers.txt`.

## Customization
- You can adjust the parameters of the image processing functions and the KMeans clustering according to your needs.

## Contact
Your Name - [contact@darshjoshi.com](mailto:contact@darshjoshi.com)
