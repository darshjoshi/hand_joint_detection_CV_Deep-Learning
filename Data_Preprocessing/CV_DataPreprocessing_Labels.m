source_directory = (file path to source_directory)
output_directory =  (file path to Output_directory
label_directory =  (file path to label_directory


convertDicomInFoldersStartingWith9(source_directory, output_directory, label_directory);

function convertDicomInFoldersStartingWith9(source_dir, output_dir, label_dir)
    folders = dir(fullfile(source_dir, '9*'));
    
    for i = 1:numel(folders)
        current_folder = fullfile(folders(i).folder, folders(i).name, 'v06');
        
        if exist(current_folder, 'dir') == 7
            image_filename = convertDicomToJpgInFolder(current_folder, output_dir, folders(i).name);
            renameLabelFile(label_dir, output_dir, folders(i).name, image_filename);
        else
            disp(['v06 folder not found in ', folders(i).name]);
        end
    end
end

function image_filename = convertDicomToJpgInFolder(folder, output_dir, folder_name)
    dicom_filename = fullfile(folder, '001');
    image_filename = fullfile(output_dir, [folder_name, '.jpg']);
    
    convertDicomToJpg(dicom_filename, image_filename);
    return;
end

function convertDicomToJpg(dicomFilename, jpgFilename)
    dicomImage = dicomread(dicomFilename);
    adjustedImage = imadjust(dicomImage);

    if isa(adjustedImage, 'uint16')
        adjustedImage = uint8(255 * mat2gray(adjustedImage));
    end

    imwrite(adjustedImage, jpgFilename, 'jpg');
end

function renameLabelFile(label_dir, output_dir, folder_name, image_filename)
    % Construct the label file path based on the folder name and label directory
    label_filename = fullfile(label_dir, [folder_name, '_v06.txt']); % Modify this line based on your label file naming convention
    
    if exist(label_filename, 'file') == 2
        [~, name, ~] = fileparts(image_filename);
        new_label_filename = fullfile(output_dir, [name, '_v06.txt']);
        
        movefile(label_filename, new_label_filename);
    else
        disp(['Label file not found for ', folder_name]);
    end
end

