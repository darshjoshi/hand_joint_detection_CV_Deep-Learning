# DICOM to JPG Conversion Tool

This MATLAB tool converts DICOM images to JPG format, focusing on folders with specific naming conventions.

## Prerequisites

- MATLAB installed on your system.

## Configuration

Set the source and output directories in your MATLAB script or through a configuration file:
source_directory = '[your-source-directory]';
output_directory = '[your-output-directory]';

## Features

Converts DICOM images in specified folders to JPG format.
Targets folders with specific naming conventions ('9*').
Provides error messages if target folders are not found.


# DICOM Image Processing Tool

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
   
