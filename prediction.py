import os
import tensorflow as tf #tf 2.0.0
import numpy as np
import pandas as pd
from tensorflow.keras import Model 
from tensorflow.keras.layers import Dense, Dropout, Flatten,GlobalAveragePooling2D
from keras.preprocessing.image import ImageDataGenerator


# 파일의 실행경로 및 예측할 이미지 경로, 결과를 저장할 경로 설정
path = os.path.abspath(__file__)
path = path.replace('c','C', 1)
file_dir_path = os.path.dirname(path)

pred_image_dir_path = os.path.join( file_dir_path, 'pred_images' )
pred_result_dir_path = os.path.join( file_dir_path, 'pred_results' )



# generator / img size / 클래스 등 기본사항 설정
img_h = 300
img_w = 300
num_classes = 4
classes = [ 'cutest',
            'prettyCute',
            'soso',
            'ugly' ]

SEED = 1234
tf.random.set_seed( SEED ) 

test_data_generator = ImageDataGenerator( rescale=1./299 )
test_generator = test_data_generator.flow_from_directory( pred_image_dir_path,
                                                          target_size=(300, 300),
                                                          batch_size=1,
                                                          shuffle=False,
                                                          class_mode=None,
                                                          seed=SEED )


# 평가하려는 모델 선정 및 예측 진행
model_dir_name = 'model_1'
model_name = 'saved_model'
model_dir_path = os.path.join( file_dir_path, 'saved_model' )
model_dir_path = os.path.join( model_dir_path, model_dir_name )

model_weights_dir_name = 8
model_weights_name = 'cp-val_acc-0.700000.ckpt'
model_weights_path = os.path.join( file_dir_path, 'checkpoints' )
model_weights_path = os.path.join( model_weights_path, '{dir_name}\\{model_name}'
                                                .format( dir_name=model_weights_dir_name, model_name=model_weights_name ) ) # 원하는 가중치 이름 넣기

model = tf.keras.models.load_model( model_dir_path )
model.load_weights(model_weights_path)

prediction_generator = model.predict( test_generator,
                                      steps=test_generator.n//test_generator.batch_size,
                                      verbose=1 )
                                      
test_generator.reset()


# 결과 저장 형식 지정, 내보내기
predictions = np.array( [k for k in prediction_generator] )
filenames = test_generator.filenames 

i = 0
for arr in predictions :
      j = 0
      for prob in arr :
            predictions[i,j] = round(prob,4)
            j += 1
      i += 1

results=pd.DataFrame( {"filename"    : filenames,
                       "Cutest"      : predictions[:,0],
                       "Pretty-cute" : predictions[:,1],
                       "Soso"        : predictions[:,2],
                       "Ugly"        : predictions[:,3], 
                       "Score"       : predictions[:,0] * 100 + predictions[:,1] * 70 + predictions[:,2] * 40 + predictions[:,3] * 10
                       } )

results.to_csv( 
      pred_result_dir_path + '\\' + 
      "dir{model_dir_name}-{model_name}-result.csv"
      .format( 
            model_dir_name = model_weights_dir_name, 
            model_name=model_weights_name ), index=False )