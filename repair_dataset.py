from PIL import Image
import time

with open('./misclassified_images.txt', 'r') as misclassified_images:
    contents = misclassified_images.readlines()

for file in contents:
	file_list = file.split(" ")
	file_path = list(filter(None, file_list))[1]
	print(file_path)
	image = Image.open(file_path)
	image.show()
	time.sleep(3)
		
