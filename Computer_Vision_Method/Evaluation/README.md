# Coordinate Evaluation Tool

## Overview
This tool evaluates predicted coordinates against reference coordinates. It calculates Mean Squared Error (MSE) and Root Mean Squared Error (RMSE) and visualizes the predicted coordinates alongside their closest reference coordinates.

## Prerequisites
- Python 3.8 and higher
- numpy
- matplotlib

## Installation
1. Clone the repository or download the script.
2. Install the required packages from the `requirements.txt` file:
```bash
pip install -r requirements.txt
```


## Usage
Run the script with the paths to the reference and predicted coordinates files:

```bash
python evaluation.py --ref [path_to_reference_file] --pred [path_to_predicted_file]
```


## Output
- Prints the MSE and RMSE values.
- Displays a scatter plot showing the predicted and closest reference coordinates.
