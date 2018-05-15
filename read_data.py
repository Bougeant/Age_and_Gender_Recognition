import pandas as pd
import numpy as np
import scipy.io
from datetime import datetime, timedelta

# Converting metadata Matlab files to pandas dataframe
mat = scipy.io.loadmat("Data/imdb/imdb.mat")
mat_data = mat["imdb"][0][0]

columns_imbd = ["dob", "year_photo_taken", "full_path", "gender", "name", "face_location", "face_score", "second_face_score", "celeb_names", "celeb_id"]
columns_wiki = ["dob", "year_photo_taken", "full_path", "gender", "name", "face_location", "face_score", "second_face_score"]


data = pd.DataFrame()
for index, item in enumerate(mat_data):
	print("hello")
	print(len(item[0]))
	print(item[0])
	if len(item[0]) == len(mat_data[0][0]):
		data[columns_imbd[index]] = item[0]

print(data)



# Converting date of birth from Matlab serial date number to pandas datetime
def convert_dob(dob):
	try:
		python_dob = datetime.fromordinal(int(dob)) + timedelta(days=dob%1) - timedelta(days = 366)
	except OverflowError:
		python_dob = None
	return python_dob
data["dob_py"] = data["dob"].apply(convert_dob)
