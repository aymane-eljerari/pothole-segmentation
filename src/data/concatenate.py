import os
import shutil


def concatenate_data(path):
    '''
    (String)        path: local data directort
    '''
    
    data_path = '/home/aymane/School/pothole-localization/data/segmentation/raw'
    folders = ['training', 'testing', 'validation']
    subfolders = ['rgb/', 'label/']
    count = 0

    # Check if directory exists
    if not os.path.exists(data_path):
        os.makedirs(data_path)

    # Create sub directories
    if not next(os.scandir(data_path), None):
        for subfolder in subfolders:
            os.makedirs(os.path.join(data_path, subfolder))
        print("Created segmentation data directorty.")
    
    # Copy all files into input and labels
    for subfolder in subfolders:
        for folder in folders:
            for file_name in os.listdir(os.path.join(path, folder, subfolder)):
                count += 1
                source = os.path.join(path, folder, subfolder, file_name)
                destination = os.path.join(data_path, subfolder)
                if os.path.isfile(source):
                    shutil.copy(source, destination)
                    old_name = os.path.join(destination, file_name)
                    new_name = os.path.join(destination, str(count)+'.png')
                    os.rename(old_name, new_name)
    
    print("Files organized successfully.")

if __name__ == '__main__':
    path = "/Users/aymane/Downloads/pothole600/"
    concatenate_data(path)
