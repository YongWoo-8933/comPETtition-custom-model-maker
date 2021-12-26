from genericpath import exists
import glob
import os
import shutil
import numpy as np

# 이미지 분류 비율 설정
training_ratio = 0.8
validation_ratio = 0.1
testing_ratio = 1 - training_ratio - validation_ratio


# 나뉘어진 array를 바탕으로 training, testing, validation으로 파일을 분류
def makeImageSet( form, class_dir ) : 
    _path = os.path.join( dataset_path, form )

    if not os.path.exists( _path ) :
        os.mkdir( _path )
    else : 
        pass


    if form == 'training' :
        _array = training_array
        _path = os.path.join( _path, class_dir )
    elif form == 'testing' :
        _array = testing_array
        _path = os.path.join( _path, 'test' )
    elif form == 'validation' :
        _array = validation_array
        _path = os.path.join( _path, class_dir )
    else :
        print('Error : form is not exist')
        return False

    if not os.path.exists( _path ) :
        os.mkdir( _path )
    else : 
        pass

    
    for file in _array :
        file_name_location = file.rfind('\\')
        file_name = file[(file_name_location+1):]
        new_file_path = os.path.join( _path, file_name )

        if os.path.exists( new_file_path ) : 
            print( 'Error : Existing file : {}'.format(new_file_path) )
        else :
            shutil.copy( file, new_file_path )


# 파일의 현재 실행위치를 찾고, 폴더 및 dataset의 경로를 저장
path = os.path.abspath(__file__)
path = path.replace('c','C', 1)
file_dir_path = os.path.dirname(path)
dataset_path = os.path.join(file_dir_path, 'dataset')
dataset_images_path = os.path.join(dataset_path, 'images')

# train - test split
image_classes_dir_list = [ x[1] for x in os.walk(dataset_images_path) ]
image_classes_dir_list = image_classes_dir_list[0]
image_names_array = np.array( image_classes_dir_list )


for class_dir in image_classes_dir_list :
    # origin class 폴더 경로 저장, 파일 긁어 리스트화
    origin_dir_path = os.path.join( dataset_images_path, class_dir )
    filename_list = glob.glob( os.path.join( origin_dir_path, '*.jpg' ) )
    filename_array = np.array( filename_list )

    # testing file 랜덤 샘플링
    testing_file_nums = int( testing_ratio * len( filename_list ) )
    testing_array = np.random.choice( filename_array , testing_file_nums, replace=False)

    training_and_validation_array = np.setdiff1d(filename_array, testing_array)

    # validation file 랜덤 샘플링
    new_validation_ratio = validation_ratio / ( training_ratio + validation_ratio )
    validation_file_nums = int( new_validation_ratio * len( training_and_validation_array ) )
    validation_array = np.random.choice( training_and_validation_array , validation_file_nums, replace=False)

    # training file 자동 결정
    training_array = np.setdiff1d(training_and_validation_array, validation_array)

    # dir 형성
    makeImageSet( 'training', class_dir )
    makeImageSet( 'testing', class_dir )
    makeImageSet( 'validation', class_dir )


