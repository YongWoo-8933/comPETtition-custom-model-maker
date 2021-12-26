import glob
import numpy as np
import os

from PIL import Image
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.preprocessing.image import load_img

aug = ImageDataGenerator(
	rotation_range=20,
	zoom_range=0.15,
	width_shift_range=0.2,
	height_shift_range=0.2,
	shear_range=0.15,
	horizontal_flip=True,
	fill_mode="nearest"
	)

# 파일의 현재 실행위치를 찾고, 폴더 및 dataset의 경로를 저장
path = os.path.abspath(__file__)
path = path.replace('c','C',1)
file_dir_path = os.path.dirname(path)
dataset_path = os.path.join(file_dir_path, 'dataset')
dataset_images_path = os.path.join(dataset_path, 'images')
aug_images_path = os.path.join(dataset_path, 'image_aug')

# 원본 class 폴더 list
image_classes_dir_list = [ x[1] for x in os.walk(dataset_images_path) ]
image_classes_dir_list = image_classes_dir_list[0]

for class_dir in image_classes_dir_list :
    # origin class 폴더 경로 저장, 파일 긁어 리스트화
	origin_dir_path = os.path.join( dataset_images_path, class_dir )
	filename_list = glob.glob( os.path.join( origin_dir_path, '*.jpg' ) )
	
    # 폴더 list를 기반으로 image_aug에 폴더생성, target(new) 경로저장
	new_dir_path = os.path.join( aug_images_path, class_dir ) 
	if os.path.exists( new_dir_path ) : 
		print ('Error: Existing directory. ' +  new_dir_path)
		break
	else : 
		os.makedirs( new_dir_path )

	# augmentation 실행 후 target dir에 저장
	for file in filename_list : 
		image = img_to_array ( Image.open(file) )
		image = np.expand_dims( image, axis=0 )
		aug_image_list = aug.flow(
									image, 
									batch_size = 1, 
									save_to_dir = new_dir_path,
									save_prefix="image", 
									save_format="png"
									)
		
		target_nums = 5      # 새로만들 aug_img 갯수 
		nums = 0
		
		for i in aug_image_list:
			nums += 1
			if nums == target_nums :
				break