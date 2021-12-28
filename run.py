# -*- coding: utf-8 -*-
import os
import tensorflow as tf #tf 2.0.0
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow_hub as hub
from tensorflow.keras import Model 
from tensorflow.keras.layers import DepthwiseConv2D, Conv2D, Dense, MaxPooling2D, Dropout, Flatten,GlobalAveragePooling2D, Activation
from tensorflow.keras.models import Sequential
from tensorflow.keras.callbacks import ReduceLROnPlateau
from keras.preprocessing.image import ImageDataGenerator
from tensorflow.python.keras.layers.normalization_v2 import BatchNormalization


# 배치, 클래스 등 기본 설정
Batch_size = 16
img_h = 260
img_w = 260
num_classes = 4
classes = [ 'cutest', # 0
            'prettyCute', # 1
            'soso', # 2
            'ugly', # 3
           ]

SEED = 1234
tf.random.set_seed(SEED) 


# ImageDataGenerator (in-place augmentation)
train_data_gen = ImageDataGenerator(rotation_range=40,
                                    width_shift_range=0.3,
                                    height_shift_range=0.3,
                                    zoom_range=0.4,
                                    horizontal_flip=True,
                                    vertical_flip=True,
                                    fill_mode='nearest',
                                    cval=0,
                                    rescale=1./(img_h-1))

valid_data_gen = ImageDataGenerator(rescale=1./(img_h-1))

test_data_gen = ImageDataGenerator(rescale=1./(img_h-1))


# 파일의 현재 실행위치를 찾고, 경로저장
path = os.path.abspath(__file__)
path = path.replace('c','C', 1)
file_dir_path = os.path.dirname(path)
dataset_path = os.path.join( file_dir_path, 'dataset' )

# 기존 이미지 경로저장
training_path = os.path.join( dataset_path, 'training' )
validation_path = os.path.join( dataset_path, 'validation' )
testing_path = os.path.join( dataset_path, 'testing' )

# # training에 사용된 이미지 경로 생성, 저장
# trained_dir_path = os.path.join(file_dir_path, 'trained')
# trained_training_path = os.path.join( trained_dir_path, 'training')
# trained_validation_path = os.path.join( trained_dir_path, 'validation')
# trained_testing_path = os.path.join( trained_dir_path, 'testing')
# if os.path.exists( trained_training_path ) :
#       print('Error : trained_training_path exists')
# elif os.path.exists( trained_validation_path ) :
#       print('Error : trained_validation_path exists')
# elif os.path.exists( trained_testing_path ) :
#       print('Error : trained_testing_path exists')
# else :
#       os.makedirs( trained_training_path )
#       os.makedirs( trained_validation_path )
#       os.makedirs( trained_testing_path )

# checkpoint 경로저장
checkpoint_dir_path = os.path.join( file_dir_path, 'checkpoints' )
if not os.path.exists(checkpoint_dir_path) :
      os.makedirs(checkpoint_dir_path)
checkpoint_dir_list = os.listdir( checkpoint_dir_path )
checkpoint_dir_num = len( checkpoint_dir_list )
checkpoint_dir = os.path.join( checkpoint_dir_path, '{}'.format( checkpoint_dir_num ) )
os.makedirs( checkpoint_dir )
checkpoint_path = os.path.join( checkpoint_dir, 'cp-val_acc-{val_accuracy:04f}.ckpt' )


# Testing 데이터 생성
test_gen = test_data_gen.flow_from_directory(testing_path,
                                             target_size=(img_w, img_h),
                                             batch_size=1,
                                             shuffle=False,
                                             class_mode=None,
                                             seed=SEED
                                             )

# Training 데이터 생성
train_gen = train_data_gen.flow_from_directory( training_path,
                                                target_size=(img_w, img_h),
                                                batch_size=Batch_size,
                                                classes=classes,
                                                class_mode='categorical',
                                                shuffle=True,
                                                seed=SEED
                                                ) 

# Validation 데이터 생성
valid_gen = valid_data_gen.flow_from_directory( validation_path,
                                                target_size=(img_w, img_h),
                                                batch_size=Batch_size, 
                                                classes=classes,
                                                class_mode='categorical',
                                                shuffle=True,
                                                seed=SEED
                                                )



# 맞춤형 모델
base_layer = hub.load("https://tfhub.dev/google/imagenet/efficientnet_v2_imagenet21k_ft1k_b2/classification/2")
base_layer = hub.KerasLayer("https://tfhub.dev/google/imagenet/efficientnet_v2_imagenet21k_ft1k_b2/feature_vector/2", trainable=False)
model = tf.keras.Sequential([
        base_layer,
        Dropout(0.2),
        Dense(units=64, activation='relu'),
        Dropout(0.2),
        Dense(units=4, activation='softmax')
    ])

model.build( input_shape=[None, img_w, img_h, 3] ) 

loss = tf.keras.losses.CategoricalCrossentropy()
optimizer = tf.keras.optimizers.Adam(learning_rate=0.00015)
model.compile(optimizer=optimizer, loss=loss, metrics= ['accuracy'])

model.summary()

# callback 함수 설정 : lrr, cp
lrr = ReduceLROnPlateau(monitor='val_accuracy', 
                        factor=0.5, 
                        patience=5, 
                        verbose=1, 
                        min_lr=0.00001 )

cp_callback = tf.keras.callbacks.ModelCheckpoint( filepath=checkpoint_path,
                                                  monitor='val_accuracy',
                                                  save_best_only = True,
                                                  mode = 'max',
                                                  save_weights_only = True,
                                                  verbose=1 )

model.save_weights( checkpoint_path.format(val_accuracy=0) )

callbacks = [ lrr, cp_callback ]

# model fit_generator
STEP_SIZE_TRAIN=train_gen.n//train_gen.batch_size
STEP_SIZE_VALID=valid_gen.n//valid_gen.batch_size
STEP_SIZE_TEST=test_gen.n//test_gen.batch_size

transfer_learning_history = model.fit( train_gen,
                                       steps_per_epoch=STEP_SIZE_TRAIN,
                                       validation_data=valid_gen,
                                       validation_steps=STEP_SIZE_VALID,
                                       epochs=30,
                                       callbacks=callbacks,
                                       class_weight=None
)

# save model
saved_model_path = os.path.join( file_dir_path, 'saved_model' )
model_num = len( os.listdir( saved_model_path ) )
saved_model_name = os.path.join( saved_model_path, 'model_' + str(model_num) )
model.save( saved_model_name )

# 학습중 가장 높은 val_accuracy를 보였던 weights로 평가진행
weight_list = os.listdir( checkpoint_dir )
best_weight_name = weight_list[-1].replace( '.index', '' )
best_weight = os.path.join( checkpoint_dir, best_weight_name )
model.load_weights(best_weight)

# model evaluate with valid set
model.evaluate(valid_gen, steps=STEP_SIZE_VALID, verbose=1)

test_gen.reset()
pred = model.predict( test_gen,
                      steps=STEP_SIZE_TEST,
                      verbose=1)

predicted_class_indices = np.argmax(pred,axis=1)
filenames = test_gen.filenames 
answer_list = []
for filename in filenames : 
      if filename.find( 'cutest' ) >= 0 :
            answer_list.append( 0 )
      elif filename.find( 'prettyCute' ) >= 0 :
            answer_list.append( 1 )
      elif filename.find( 'soso' ) >= 0 :
            answer_list.append( 2 )
      elif filename.find( 'ugly' ) >= 0 :
            answer_list.append( 3 )

results = pd.DataFrame( {"Id" : filenames,
                         "prediction" : predicted_class_indices,
                         "answer" : answer_list
                        } )

correct = 0
for i, pred in enumerate ( predicted_class_indices ):
      if pred == answer_list[i] : correct += 1
      elif abs( pred - answer_list[i] ) == 1 : correct += 0.3
      elif abs( pred - answer_list[i] ) == 3 : correct -= 0.5
          
model_accuracy = round( ( correct / len( answer_list ) ) * 100, 2)
                        
results.to_csv("result.csv",index=False)
accuracy_file_name = os.path.join( saved_model_name, "{cp}_accuracy_{acc}.txt".format( cp = best_weight_name, acc = model_accuracy ) )
f = open( accuracy_file_name, 'w' )
f.close()