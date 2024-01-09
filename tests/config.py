import os

def get_folder_names(directory, image_extensions=(".jpg", ".jpeg", ".png")):
    non_empty_image_folders = []    
    for entry in os.listdir(directory):
        full_path = os.path.join(directory, entry)      
        if os.path.isdir(full_path):
            # Check if the directory contains at least one image
            contains_image = any(file.lower().endswith(image_extensions) for file in os.listdir(full_path))
            if contains_image:
                non_empty_image_folders.append(entry)

    return non_empty_image_folders


def assert_images(directory):
    count = 0
    categories = get_folder_names(directory)
    cat = [i for i in range(len(categories))]
    for c in categories:
        path = os.path.join(directory, c)
        for file in os.listdir(path):
            if file.lower().endswith(".jpg"):
                count += 1
    return count, cat
