# DICOM Image Processing Tool
Executed by - Mihika Mishra

## Overview
In the data preprocessing phase ,one MATLAB script was implemented to efficiently prepare our X-ray image dataset for analysis. The primary goals were to convert DICOM files to the more accessible JPEG format and systematically organize the corresponding labeling information.

This MATLAB tool processes DICOM images by converting them to JPG format and renaming associated label files. It is designed for medical image datasets, particularly those stored in a structured folder hierarchy.

## Prerequisites
- MATLAB (R2023b)

Before running the script, set the source, output, and label directories. These can be passed as arguments to the main function:

source_directory = '[Path to DICOM images]';
output_directory = '[Path for output JPG images]';
label_directory = '[Path for label files]';

## Usage
Run the script in MATLAB:

convertDicomInFoldersStartingWith9(source_directory, output_directory, label_directory);

## Features

Converts DICOM images to JPG format.
Renames label files to match converted image names.
Handles a specific folder structure (folders starting with '9' and containing 'v06' subfolders).
Contributing

## Contact
  Email - mihikamishra5@gmail.com 
