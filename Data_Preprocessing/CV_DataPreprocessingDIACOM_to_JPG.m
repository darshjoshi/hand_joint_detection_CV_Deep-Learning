% Main function to convert DICOM files in specified folders to JPG format
source_directory = (file path to source_directory)
output_directory =  (file path to Output_directory)

function convertDicomInFoldersStartingWith9(source_dir, output_dir)
    % Read directories starting with '9' in the source directory
    folders = dir(fullfile(source_dir, '9*'));
    
    for i = 1:numel(folders)
        % Define the current folder path
        current_folder = fullfile(folders(i).folder, folders(i).name, 'v06');
        
        % Check if the 'v06' folder exists
        if exist(current_folder, 'dir') == 7
            % Convert DICOM to JPG in the current folder
            convertDicomToJpgInFolder(current_folder, output_dir, folders(i).name);
        else
            % Display a message if the 'v06' folder is not found
            disp(['v06 folder not found in ', folders(i).name]);
        end
    end
end

% Function to convert a DICOM file to JPG format in a specified folder
function convertDicomToJpgInFolder(folder, output_dir, folder_name)
    % Define the DICOM file path
    dicom_filename = fullfile(folder, '001');
    
    % Create a JPG filename using the folder name
    jpg_filename = fullfile(output_dir, [folder_name, '.jpg']);
    
    % Convert DICOM to JPG
    convertDicomToJpg(dicom_filename, jpg_filename);
end

% Core function to handle the DICOM to JPG conversion
function convertDicomToJpg(dicomFilename, jpgFilename)
    % Read the DICOM image
    dicomImage = dicomread(dicomFilename);
    
    % Adjust the image contrast
    adjustedImage = imadjust(dicomImage);

    % Convert to uint8 if necessary
    if isa(adjustedImage, 'uint16')
        adjustedImage = uint8(255 * mat2gray(adjustedImage));
    end

    % Write the adjusted image to a JPG file
    imwrite(adjustedImage, jpgFilename, 'jpg');
end
