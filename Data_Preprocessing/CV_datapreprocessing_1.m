source_directory = '/Users/mihikamishra/Documents/cv_folder/x_ray_data';
output_directory = '/Users/mihikamishra/Documents/cv_folder/output_folder';

convertDicomInFoldersStartingWith9(source_directory, output_directory);

function convertDicomInFoldersStartingWith9(source_dir, output_dir)
    folders = dir(fullfile(source_dir, '9*'));
    
    for i = 1:numel(folders)
        current_folder = fullfile(folders(i).folder, folders(i).name, 'v06');
        
        if exist(current_folder, 'dir') == 7
            convertDicomToJpgInFolder(current_folder, output_dir, folders(i).name);
        else
            disp(['v06 folder not found in ', folders(i).name]);
        end
    end
end

function convertDicomToJpgInFolder(folder, output_dir, folder_name)
    dicom_filename = fullfile(folder, '001');
    
    % Create a unique filename using the folder name
    jpg_filename = fullfile(output_dir, [folder_name, '.jpg']);
    
    convertDicomToJpg(dicom_filename, jpg_filename);
end

function convertDicomToJpg(dicomFilename, jpgFilename)
    dicomImage = dicomread(dicomFilename);
    adjustedImage = imadjust(dicomImage);

    if isa(adjustedImage, 'uint16')
        adjustedImage = uint8(255 * mat2gray(adjustedImage));
    end

    imwrite(adjustedImage, jpgFilename, 'jpg');
end
