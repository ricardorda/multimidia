import numpy as np
from skimage.feature import local_binary_pattern
from PIL import Image
import os
import pandas as pd
import imghdr
import matplotlib.pyplot as plt


NRI_UNIFORM_FEATURE_NUMBER = 59

# Setting up the train and test directories
train_directory = 'G:\\temp' #Diretorio que voces decompacataram a RYDLES.
lbp_extractor = 'nri_uniform'

# Setting up the resulting matrices directories
feature_matrix_train_path = 'Feature Matrix Train'

class LocalBinaryPatterns:
    def __init__(self, numPoints, radius):
        self.numPoints = numPoints
        self.radius = radius

    # LBP Feature Extractor from Skimage
    def describe_lbp_method_ag(self, image):
        lbpU = local_binary_pattern(image, self.numPoints, self.radius, method='nri_uniform')
        hist0, nbins0 = np.histogram(np.uint8(lbpU), bins=range(60), normed=True)

        #Exibe no console o vetor de características:
        print(hist0)

        return hist0

# Function to load an image from a path
def open_img(filename):
    img = Image.open(filename)
    return img

# Verify if a given image is using a valid format
def verify_valid_img(path):
    possible_formats = ['png','jpg','jpeg','tiff','bmp','gif']
    if imghdr.what(path) in possible_formats:
        return True
    else:
        return False

# Feature extraction call
def feature_extraction(image, lbp_extractor):
    lbp = LocalBinaryPatterns(8, 2) #Paramametros do LBP.
    image_matrix = np.array(image.convert('L'))
    img_features = lbp.describe_lbp_method_ag(image_matrix)

    return img_features.tolist()

def create_columns(column_number, property):
    columns = []
    for i in range(0, column_number):
        columns.append(str(i))

    columns.append(property)
    return columns

# Function to create the training feature matrix, it has the expected class for each sample
def create_feature_matrix_train(train_directory, lbp_extractor):
    # Variable to store the data_rows
    rows_list = []

    print("Started feature extraction for the training dataset")

    # Iterate over subdirectories in training folder (1 folder for each class)
    for dir in os.listdir(train_directory):

        print("Estou em", dir);

        # This is the path to each subdirectory
        sub_directory = train_directory + '\\' + dir

        # Retrieve the files for the given subdirectory
        training_filelist = os.listdir(sub_directory)

        # Iterate over all the files in the class folder
        for file in training_filelist:
            file_path = sub_directory + '\\' + file

            if verify_valid_img(file_path):
                print("Processing: "+file_path)

                image = open_img(file_path)
                img_features = feature_extraction(image, lbp_extractor)

                # The name of the directory is the class
                img_features.append(dir)

                rows_list.append(img_features)
            else:
                print("The following file is not a valid image: "+file_path)

    # Creating a dataframe to store all the features
    columns = create_columns(NRI_UNIFORM_FEATURE_NUMBER, 'class')

    feature_matrix = pd.DataFrame(rows_list, columns=columns)

    print("Finished creating Training Feature Matrix")

    return feature_matrix






if not os.path.isdir(feature_matrix_train_path):
    print('Creating Directory: '+feature_matrix_train_path)
    os.mkdir(feature_matrix_train_path)


feature_matrix_train = create_feature_matrix_train(train_directory, lbp_extractor)
print("Saving Training Feature Matrix to CSV")
feature_matrix_train.to_csv(feature_matrix_train_path + '\\feature_matrix_train.csv', index=False)







