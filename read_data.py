import pandas as pd
import numpy as np
import scipy.io
from datetime import datetime, timedelta
from sklearn.utils import shuffle
import shutil
import os

## Parameters ##

age_bins = [[0,10],[11,20],[21,30],[31,40],[41,50],[51,60],[61,70],[70,120]]
# age_bins = range(0,120)

# Definition of parameters for gender recognition
gender_bins = ["male", "female"]
gender_sample_size = 10000

# Parameters for filtering faces only
min_face_score = 1.0

# Definition of columns for metadata Matlab files
columns_imbd = ["dob", "year_photo_taken", "full_path", "gender", "name", "face_location", "face_score", "second_face_score", "celeb_names", "celeb_id"]
columns_wiki = ["dob", "year_photo_taken", "full_path", "gender", "name", "face_location", "face_score", "second_face_score"]


## Functions ##

# This function converts metadata Matlab files to pandas dataframe
def convert_mat_dataframe(mat_file, columns, dict_key):
    mat = scipy.io.loadmat(mat_file)
    mat_data = mat[dict_key][0][0]
    df = pd.DataFrame()
    for index, item in enumerate(mat_data):
        if len(item[0]) == len(mat_data[0][0]): # Columns are only added to the dataframe if all the data is available
            df[columns[index]] = item[0]
    return(df)

# This function converts the Matlab serial date number to pandas datetime
def convert_date(Matlab_date):
    try:
        python_dt = datetime.fromordinal(int(Matlab_date)) + timedelta(days=Matlab_date%1) - timedelta(days = 366)
    except OverflowError:
        python_dt = None
    return python_dt

# This function converts the numpy arrays in the metadata Matlab files to its first element
def select_first_element(array):
    return array[0]

# This function calculates the age of the person the year the photo was taken
def calculate_age(df):
    age_list = []
    for index, row in df.iterrows():
        # Since data of photo is not known precisely, it is assumed to be July 1st
        date_photo = datetime(year = row["year_photo_taken"], month = 7, day = 1)
        date_of_birth = row["dob_py"]
        if date_of_birth.month < date_photo.month:
            age = int(date_photo.year - date_of_birth.year)
        else:
            age = int(date_photo.year - date_of_birth.year - 1)
        age_list.append(age)
    return age_list

# This function clarifies the gender in the daraframe (1.0 is replaced by "male" and 0.0 is replaced by "female")
def clarify_gender(gender):
    if gender == 1.0:
        gender = "male"
    else:
        gender = "female"
    return gender

# This function removes face score below a defined threshold
def filter_portraits(df):
    old_size = len(df)
    df = df[df["face_score"] >= min_face_score ]
    df = df[df["second_face_score"].isnull()]
    new_size = len(df)
    print(old_size - new_size, "rows filtered due to face score")
    return df

# This function selects a random sample of pictures for each bin:
def select_samples_gender(df):
    np.random.seed(2)
    df2 = pd.DataFrame()
    new_df = pd.DataFrame()
    for item in gender_bins:
        df2 = df[df["gender"] == item]
        df2 = shuffle(df2)
        df2 = df2.iloc[0:gender_sample_size]
        print(len(df2), "rows selected for gender:", item)
        new_df = pd.concat([new_df, df2])
    return new_df

# This function renames the full path of images 
def absolute_path_images(path, folder):
    full_path = str(folder+path)
    print(full_path)
    return full_path

# This function moves selected training images into the training folder
def move_training_images(df, folder, feature):
    try:
        os.mkdir("./train/")
        for category in df[feature].unique():
            os.mkdir("./train/"+category+"/")
    except OSError:
        print("Training directory already exists")
    with open('./memory.txt', 'w') as memory_file:
        for index, row in df.iterrows():
            category = row[feature]
            old_path = folder+row["full_path"]
            new_path = "./train/"+category+"/"+row["full_path"].split("/")[-1]
            memory_file.write(old_path+" , "+new_path+"\n")
            try:
                shutil.copy(old_path, new_path)
            except:
                print("File missing :", old_path)
        print(len(df), "selected training images moved to traning folder")



## Main ##

# Converting metadata Matlab files to pandas dataframe
imdb_metadata = convert_mat_dataframe("./Data/imdb/imdb.mat", columns_imbd, "imdb")
wiki_metadata = convert_mat_dataframe("./Data/wiki/wiki.mat", columns_wiki, "wiki")

# Converting date of birth from Matlab serial date number to pandas datetime
imdb_metadata["dob_py"] = imdb_metadata["dob"].apply(convert_date)
wiki_metadata["dob_py"] = wiki_metadata["dob"].apply(convert_date)
old_size_imdb = len(imdb_metadata)
old_size_wiki = len(wiki_metadata)

# Removing rows with invalid date of birth
imdb_metadata = imdb_metadata[imdb_metadata["dob_py"].notna()]
new_size_imdb = len(imdb_metadata)
print("IMDB data:", old_size_imdb - new_size_imdb, "rows removed due to invalid date of birth")
wiki_metadata = wiki_metadata[wiki_metadata["dob_py"].notna()]
new_size_wiki = len(wiki_metadata)
print("Wikipedia data:", old_size_wiki - new_size_wiki, "rows removed due to invalid date of birth")

# Removing unnecessary numpy array level
imdb_metadata["name"] = imdb_metadata["name"].apply(select_first_element)
imdb_metadata["full_path"] = imdb_metadata["full_path"].apply(select_first_element)
imdb_metadata["face_location"] = imdb_metadata["face_location"].apply(select_first_element)

# Calculating age when photo was taken
imdb_metadata["age"] = calculate_age(imdb_metadata)
wiki_metadata["age"] = calculate_age(wiki_metadata)

# Clarify gender class
imdb_metadata["gender"] = imdb_metadata["gender"].apply(clarify_gender)
wiki_metadata["gender"] = wiki_metadata["gender"].apply(clarify_gender)

# Filter protraits only based on face_score and second_face_score
imdb_metadata = filter_portraits(imdb_metadata)

# Select random sample (without replacement) from data for each gender
imdb_train = select_samples_gender(imdb_metadata)

# Rename full path of images
#imdb_train["full_path"] = imdb_train["full_path"].apply(absolute_path_images, args = ("./Data/imdb/",))

# Move training images in training folder
move_training_images(imdb_train, "./Data/imdb/", "gender")
