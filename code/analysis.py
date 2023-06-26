import numpy as np
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve
import tensorflow as tf
tf.config.threading.set_inter_op_parallelism_threads(20)
tf.config.threading.set_intra_op_parallelism_threads(20)
tf.random.set_seed(8008)
np.random.seed(808)

from data_mgmt import make_2D_datasets
from data_mgmt import make_generators
from models import cnn2D
from models import cnn2D_deep
from models import res2D
from models3D import cnn3D
from models3D import res3D
from models3D import mc2
from models3D import hara

# point this at 'small_local' for testing, or 'data' for full runs
root_dir = 'data'
# this is fixed
img_hw = 160


# primary method to begin training models. uncomment to train/save
def main():
   
    ## 2D Models ##
    #-------------#
    n_epoch = 10
    batch_size = 32
    
    #cnn_2D = cnn2D(n_epoch, batch_size, img_hw, root_dir, True)
    #cnn_2D.save()
    #cnn_2D_deep = cnn2D_deep(n_epoch, batch_size, img_hw, root_dir, True)
    #cnn_2D_deep.save()
    #n_epoch = 2
    res_2D = res2D(n_epoch, batch_size, img_hw, root_dir, True)
    #res_2D.save()

    
    ## 3D Models ##
    #-------------#
    #n_epoch = 2
    #batch_size = 10
    #n_frames = 16 # if you change this, nuke data/3D directory
    
    #cnn_3D = cnn3D(n_epoch, batch_size, img_hw, n_frames, root_dir, True)
    #cnn_3D.save()
    #hara_m = hara(n_epoch, batch_size, img_hw, n_frames, root_dir, True)
    #hara_m.save()
    #res_3D = res3D(n_epoch, batch_size, img_hw, n_frames, root_dir, True)
    #res_3D.save()
    #mc2_m = mc2(n_epoch, batch_size, img_hw, n_frames, root_dir, True)
    #mc2_m.save()


# secondary method to continue training saved models
def continue_training():
    #n_epoch = 2
    batch_size = 32
    #n_frames = 16 # if you change this, nuke data/3D directory
    
    
    mobilenet_redux = tf.keras.models.load_model('../saved_models/MobileNetV2')
    
    [training_generator, testing_generator] = make_2D_datasets(batch_size, img_hw, root_dir)
    
    # y_pred = tf.nn.softmax(cnn3D_redux.predict(testing_generator))[:,1] 
    y_pred = mobilenet_redux.predict(testing_generator)
    
    y_test = []
    
    # 2D data
    for x,y in testing_generator:
        y_test = y_test+y.numpy().tolist()
    
    # 3D data
    #for x,y in testing_generator:
    #    y_test = y_test+y.tolist()
        

    
    auc = roc_auc_score(y_test, y_pred)
    false_pos, true_pos, thresholds = roc_curve(y_test, y_pred)
    
    roc_data = np.array([false_pos, true_pos, thresholds]).T
    
    

if __name__ == '__main__':
    main()
