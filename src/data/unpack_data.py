import zipfile
import os
import sys
# Set the working directory to src root
os.chdir(os.path.dirname(sys.path[0]))
sys.path.append(os.path.normcase(os.getcwd()))

def unpack_raw_data():
    # Pull the raw and processed data from the data folder
    os.system('dvc pull')
    # Unzip the raw data
    with zipfile.ZipFile("../data/raw/Training.zip", 'r') as zip_ref:
        zip_ref.extractall("../data/raw/")

    with zipfile.ZipFile("../data/raw/Testing.zip", 'r') as zip_ref:
        zip_ref.extractall("../data/raw/")


if __name__ == "__main__":
    unpack_raw_data()

