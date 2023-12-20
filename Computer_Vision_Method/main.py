import cv2
import numpy as np
import tkinter as tk
from tkinter import filedialog
from sklearn.cluster import KMeans
import argparse

def display_image(title, image):
    """ Display an image with OpenCV and wait for the ESC key to close. """
    cv2.imshow(title, image)
    key = cv2.waitKey(0)  # Wait for any key press
    cv2.destroyAllWindows()
    return key

def select_roi(image):
    """ Select ROI on the image and return it along with its coordinates. """
    roi = cv2.selectROI(image)
    cv2.destroyAllWindows()

    if roi[2] > 0 and roi[3] > 0:  # ROI should have a non-zero width and height
        return image[roi[1]:roi[1]+roi[3], roi[0]:roi[0]+roi[2]], roi
    else:
        print("No ROI selected. Try again.")
        return None, None

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
    contours_image = original_image.copy()
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
            cv2.circle(image, tuple(center.astype(int)), 5, (255, 255, 255), -1)
    return cluster_centers

def main(image_path=None):
    # If no image path is provided, open a file dialog to choose an image
    if image_path is None:
        root = tk.Tk()
        root.withdraw()
        image_path = filedialog.askopenfilename()

    # Reading the image in grayscale
    original_image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if original_image is None:
        raise FileNotFoundError(f"Cannot load the image at path: {image_path}")

    # Image processing steps
    cropped_image, roi_coords = select_roi(original_image)
    if cropped_image is not None:
        x, y, _, _ = roi_coords  # Coordinates of the ROI in the original image
        key = display_image("Cropped Area", cropped_image)

        if key == 27:  # ESC key
            blurred_image = apply_gaussian_blur(cropped_image)
            gamma_corrected_image = apply_gamma_correction(blurred_image, gamma=1.5)
            clahe_image = apply_clahe(gamma_corrected_image)
            edges_image = apply_canny_edge_detector(clahe_image)
            contours_image, contours = find_and_draw_contours(cropped_image, edges_image)

            if key == 27:
                contour_centers = find_contour_centers(contours)
                adjusted_centers = contour_centers + [x, y]  # Adjust center positions
                cluster_centers = find_joint_centers_and_cluster(adjusted_centers, original_image.copy())

                if cluster_centers is not None:
                    joint_centers_image = original_image.copy()
                    for center in cluster_centers:
                        cv2.circle(joint_centers_image, (int(center[0]), int(center[1])), 5, (255, 255, 255), -1)
                    display_image("Joint Centers on Original Image", joint_centers_image)

                    # Save the cluster centers to a text file
                    with open("contour_centers.txt", "w") as file:
                        for i, center in enumerate(cluster_centers):
                            file.write(f"point_{i+1}: {{\"coords\": ({int(center[0])}, {int(center[1])})}}\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Finger Joint Detection Script")
    parser.add_argument("-i", "--image", help="Path to the input image file", required=False)
    args = parser.parse_args()

    main(image_path=args.image)
