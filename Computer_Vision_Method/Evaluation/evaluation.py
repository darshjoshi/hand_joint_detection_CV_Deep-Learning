import re
import numpy as np
import matplotlib.pyplot as plt
import argparse

def extract_reference_coords(file_path):
    """ Extract reference coordinates from the specified file. """
    with open(file_path, 'r') as file:
        file_content = file.read()
    return re.findall(r'\w+ (\d+) (\d+) \d+\.\d+', file_content)

def extract_predicted_coords(file_path):
    """ Extract predicted coordinates from the specified file. """
    with open(file_path, 'r') as file:
        file_content = file.read()
    return re.findall(r'point_\d+: {"coords": \((\d+), (\d+)\)}', file_content)

def find_closest_matches(predicted_coords, reference_coords):
    """ Find the closest reference coordinate for each predicted coordinate. """
    if not reference_coords:
        raise ValueError("Reference coordinates are empty.")
    closest_matches = {}
    for i, coord in enumerate(predicted_coords):
        closest = min(reference_coords, key=lambda x: ((x[0] - coord[0])**2 + (x[1] - coord[1])**2)**0.5)
        closest_matches[f"point_{i+1}"] = closest
    return closest_matches

def calculate_accuracy_metrics(predicted_coords, closest_coords):
    """ Calculate Mean Squared Error (MSE) and Root Mean Squared Error (RMSE). """
    predicted_coords_array = np.array(predicted_coords)
    closest_coords_array = np.array([closest_coords[f"point_{i+1}"] for i in range(len(predicted_coords))])
    mse = np.mean((predicted_coords_array - closest_coords_array)**2)
    rmse = np.sqrt(mse)
    return mse, rmse

def plot_coordinates(predicted_coords, closest_matches):
    """ Plot the predicted coordinates and their closest matches. """
    predicted_x, predicted_y = zip(*predicted_coords)
    closest_x, closest_y = zip(*[closest_matches[f"point_{i+1}"] for i in range(len(predicted_coords))])

    plt.figure(figsize=(12, 8))
    plt.scatter(predicted_x, predicted_y, color='blue', label='Predicted Coordinates')
    plt.scatter(closest_x, closest_y, color='red', marker='x', label='Closest Matches')
    plt.title('Predicted vs. Reference Coordinates')
    plt.xlabel('X Coordinate')
    plt.ylabel('Y Coordinate')
    plt.legend()
    plt.show()

def process_coordinates(reference_file_path, predicted_file_path):
    """ Process the coordinates and compute accuracy metrics. """
    reference_coords = extract_reference_coords(reference_file_path)
    predicted_coords = extract_predicted_coords(predicted_file_path)

    reference_coords = [(int(x), int(y)) for x, y in reference_coords]
    predicted_coords = [(int(x), int(y)) for x, y in predicted_coords]

    closest_matches = find_closest_matches(predicted_coords, reference_coords)
    mse, rmse = calculate_accuracy_metrics(predicted_coords, closest_matches)

    print(f'Mean Squared Error (MSE): {mse}')
    print(f'Root Mean Squared Error (RMSE): {rmse}')
    plot_coordinates(predicted_coords, closest_matches)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Evaluate Predicted Coordinates")
    parser.add_argument("--ref", help="Path to the reference coordinates file", required=True)
    parser.add_argument("--pred", help="Path to the predicted coordinates file", required=True)
    args = parser.parse_args()

    process_coordinates(args.ref, args.pred)
