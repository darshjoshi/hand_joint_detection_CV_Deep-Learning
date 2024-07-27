import cv2
import numpy as np
import streamlit as st
from sklearn.cluster import KMeans

# Load preset images
preset_images = {
    "Image 1": "D:\Projects\Computer_Vision\9003175.jpg",
    "Image 2": "D:\Projects\Computer_Vision\9004175.jpg",
    "Image 3": "D:\Projects\Computer_Vision\9120358.jpg"
}

def display_image(title, image):
    """ Display an image with Streamlit. """
    st.image(image, caption=title, use_column_width=True, channels="BGR")

def apply_gaussian_blur(image):
    """ Apply Gaussian Blur to the image and return it. """
    return cv2.GaussianBlur(image, (9, 9), 0)

def apply_gamma_correction(image, gamma=0.55):
    """ Apply Gamma Correction to the image. """
    invGamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** invGamma) * 255 for i in np.arange(0, 256)]).astype("uint8")
    return cv2.LUT(image, table)

def apply_clahe(image):
    """ Apply CLAHE to the image and return it. """
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(32, 32))
    return clahe.apply(image)

def apply_canny_edge_detector(image):
    """ Apply Canny Edge Detector to the image and return it. """
    normalized_image = cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX)
    normalized_image = np.uint8(normalized_image)
    edges = cv2.Canny(normalized_image, 50, 200)
    return edges

def find_and_draw_contours(original_image, edge_image):
    """ Find contours and draw them on a copy of the original image. """
    contours, _ = cv2.findContours(edge_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    contours_image = cv2.cvtColor(original_image, cv2.COLOR_GRAY2BGR)
    cv2.drawContours(contours_image, contours, -1, (255, 255, 0), 2)
    return contours_image, contours

def find_contour_centers(contours):
    """ Find the center points of contours. """
    centers = []
    for contour in contours:
        M = cv2.moments(contour)
        if M["m00"] != 0:
            cX = int(M["m10"] / M["m00"])
            cY = int(M["m01"] / M["m00"])
            centers.append([cX, cY])
    return np.array(centers)

def find_joint_centers_and_cluster(centers, image):
    """ Cluster the contour centers and draw them on the image. """
    cluster_centers = None
    if len(centers) > 0:
        kmeans = KMeans(n_clusters=12, random_state=42).fit(centers)
        cluster_centers = kmeans.cluster_centers_

        for center in cluster_centers:
            cv2.circle(image, tuple(center.astype(int)), 5, (0, 0, 255), -1)  # Red color
    return cluster_centers

st.title("Finger Joint Detection from X-Rays")

# Select an image from preset options
selected_image_name = st.selectbox("Choose an image from the given 3 images", list(preset_images.keys()))
selected_image_path = preset_images[selected_image_name]

# Load the selected image
original_image = cv2.imread(selected_image_path, cv2.IMREAD_GRAYSCALE)

# Display the selected image
display_image("Original Image", cv2.cvtColor(original_image, cv2.COLOR_GRAY2BGR))

# ROI selection using sliders
height, width = original_image.shape
x1 = st.slider("Left Crop", 0, width, 0)
y1 = st.slider("Top Crop", 0, height, 0)
x2 = st.slider("Right Crop", 0, width, width)
y2 = st.slider("Bottom Crop", 0, height, height)

if x2 > x1 and y2 > y1:
    cropped_image = original_image[y1:y2, x1:x2]
    display_image("Cropped Area", cv2.cvtColor(cropped_image, cv2.COLOR_GRAY2BGR))

    # Image processing steps
    blurred_image = apply_gaussian_blur(cropped_image)
    gamma_corrected_image = apply_gamma_correction(blurred_image, gamma=1.5)
    clahe_image = apply_clahe(gamma_corrected_image)
    edges_image = apply_canny_edge_detector(clahe_image)
    contours_image, contours = find_and_draw_contours(cropped_image, edges_image)

    display_image("Contours Image", contours_image)

    contour_centers = find_contour_centers(contours)
    adjusted_centers = contour_centers + [x1, y1]  # Adjust center positions
    cluster_centers = find_joint_centers_and_cluster(adjusted_centers, cv2.cvtColor(original_image.copy(), cv2.COLOR_GRAY2BGR))

    if cluster_centers is not None:
        joint_centers_image = cv2.cvtColor(original_image.copy(), cv2.COLOR_GRAY2BGR)
        for center in cluster_centers:
            cv2.circle(joint_centers_image, (int(center[0]), int(center[1])), 5, (0, 0, 255), -1)
        display_image("Joint Centers on Original Image", joint_centers_image)

        # Save the cluster centers to a text file
        with open("contour_centers.txt", "w") as file:
            for i, center in enumerate(cluster_centers):
                file.write(f"point_{i+1}: {{\"coords\": ({int(center[0])}, {int(center[1])})}}\n")

        st.write("Cluster centers saved to contour_centers.txt")
