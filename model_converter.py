import os
import tensorflow as tf #tf 2.0.0
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# 파일의 현재 실행위치를 찾고, 경로저장
path = os.path.abspath(__file__)
path = path.replace('c','C', 1)
file_dir_path = os.path.dirname(path)
saved_model_path = os.path.join( file_dir_path, 'saved_model' )
checkpoints_path = os.path.join( file_dir_path, 'checkpoints' )

# 바꿀 모델 정보
class tensor_model :
    def __init__( self, model_dir_name, cp_dir_name=None, weights_name=None ) :
        self.model_path = os.path.join( saved_model_path, model_dir_name )
        if cp_dir_name!=None and weights_name!=None : 
            self.weights_path = os.path.join( os.path.join( checkpoints_path, str(cp_dir_name) ), weights_name )
        else : self.weights_path = None

        print( '--선택한 model경로-- \n{}\n--해당 bestweights경로--\n{}\n'.format(self.model_path, self.weights_path) )

model = tensor_model( 'model_1' )
