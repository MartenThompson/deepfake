import numpy as np
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve
import tensorflow as tf
tf.config.threading.set_inter_op_parallelism_threads(20)
tf.config.threading.set_intra_op_parallelism_threads(20)
tf.random.set_seed(8008)
np.random.seed(808)
from tensorflow import keras
from tensorflow.keras import layers
from data_mgmt import make_generators
from models import modelContainer
from models import modelContainer2


def pad_depth(x, desired_channels):
    y = tf.keras.backend.zeros_like(x)
    new_channels = desired_channels - x.shape.as_list()[-1]
    y = y[..., :new_channels]
    return tf.keras.backend.concatenate([x, y], axis=-1)

def hara(n_epoch, batch_size, img_hw, n_frames, root_dir, trace=False):
    
    [training_generator, testing_generator] = make_generators(batch_size, img_hw, n_frames, root_dir)
 
    input_shape = (n_frames, img_hw, img_hw, 3) # does not include batch size

    inputs = keras.Input(shape=input_shape)
    x = layers.experimental.preprocessing.Rescaling(scale=1./127.5, offset=-1)(inputs) # or just 1./255
    
    conv1_out = layers.Conv3D(64, kernel_size=(7,7,7), strides=(1,2,2), activation='relu')(x)
    
    # Conv 2_1
    x = layers.Conv3D(64, kernel_size=(3,3,3), strides=(1,1,1), padding='same', activation=None)(conv1_out)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.activations.relu(x)
    x = layers.Conv3D(64, kernel_size=(3,3,3), strides=(1,1,1), padding='same', activation=None)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.activations.relu(x)
    block21_out = layers.add([x, conv1_out])
    
    x = layers.Conv3D(64, kernel_size=(3,3,3), strides=(1,1,1), padding='same', activation=None)(block21_out)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.activations.relu(x)
    x = layers.Conv3D(64, kernel_size=(3,3,3), strides=(1,1,1), padding='same', activation=None)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.activations.relu(x)
    block22_out = layers.add([x, block21_out])

    # Conv 3_1
    x = layers.Conv3D(128, kernel_size=(3,3,3), strides=(2,2,2), padding='same', activation=None)(block22_out)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.activations.relu(x)
    x = layers.Conv3D(128, kernel_size=(3,3,3), strides=(1,1,1), padding='same', activation=None)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.activations.relu(x)
    
    block22_out_down = layers.Conv3D(64, kernel_size=(3,3,3), strides=(2,2,2), padding='same', activation=None)(block22_out)
    block22_out_pad = pad_depth(block22_out_down,128)
    block31_out = layers.add([x, block22_out_pad])
    
    x = layers.Conv3D(128, kernel_size=(3,3,3), strides=(1,1,1), padding='same', activation=None)(block31_out)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.activations.relu(x)
    x = layers.Conv3D(128, kernel_size=(3,3,3), strides=(1,1,1), padding='same', activation=None)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.activations.relu(x)
    block32_out = layers.add([x, block31_out])
    
    # Conv 4_1
    x = layers.Conv3D(256, kernel_size=(3,3,3), strides=(2,2,2), padding='same', activation=None)(block32_out)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.activations.relu(x)
    x = layers.Conv3D(256, kernel_size=(3,3,3), strides=(1,1,1), padding='same', activation=None)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.activations.relu(x)
    
    block32_out_down = layers.Conv3D(128, kernel_size=(3,3,3), strides=(2,2,2), padding='same', activation=None)(block32_out)
    block32_out_pad = pad_depth(block32_out_down,256)
    block41_out = layers.add([x, block32_out_pad])
    
    x = layers.Conv3D(256, kernel_size=(3,3,3), strides=(1,1,1), padding='same', activation=None)(block41_out)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.activations.relu(x)
    x = layers.Conv3D(256, kernel_size=(3,3,3), strides=(1,1,1), padding='same', activation=None)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.activations.relu(x)
    block42_out = layers.add([x, block41_out])
    
    # Conv 5_1
    #x = layers.Conv3D(512, kernel_size=(3,3,3), strides=(2,2,2), padding='same', activation=None)(block42_out)
    #x = tf.keras.layers.BatchNormalization()(x)
    #x = tf.keras.activations.relu(x)
    #x = layers.Conv3D(512, kernel_size=(3,3,3), strides=(1,1,1), padding='same', activation=None)(x)
    #x = tf.keras.layers.BatchNormalization()(x)
    #x = tf.keras.activations.relu(x)
    
    #block42_out = layers.Conv3D(256, kernel_size=(3,3,3), strides=(2,2,2), padding='same', activation=None)(block42_out)
    #block42_out = pad_depth(block42_out,512)
    #block51_out = layers.add([x, block42_out])
    
    #x = layers.Conv3D(512, kernel_size=(3,3,3), strides=(1,1,1), padding='same', activation=None)(block51_out)
    #x = tf.keras.layers.BatchNormalization()(x)
    #x = tf.keras.activations.relu(x)
    #x = layers.Conv3D(512, kernel_size=(3,3,3), strides=(1,1,1), padding='same', activation=None)(x)
    #x = tf.keras.layers.BatchNormalization()(x)
    #x = tf.keras.activations.relu(x)
    #block52_out = layers.add([x, block51_out])
    x = tf.keras.layers.AveragePooling3D(pool_size=(3,3,3), padding='same')(block42_out)
    
    x = layers.Flatten()(x)
    
    outputs = layers.Dense(2, activation='softmax')(x)
    
    model = keras.Model(inputs=inputs, outputs=outputs, name='hara')
    #print(model.summary())
    #keras.utils.plot_model(model, 'hara.png', show_shapes=False, show_layer_names=False, rankdir='TB')
    
    model.compile(
        loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        #optimizer=keras.optimizers.RMSprop(),
        optimizer=tf.keras.optimizers.SGD(learning_rate=0.01, momentum=0.9),
        metrics=['accuracy'],
    )
    
    if trace:
        print('Training Hara')
    history = model.fit(training_generator, validation_data=testing_generator, epochs=n_epoch, verbose=1) # 1 for prog bar
    

    y_pred = tf.nn.softmax(model.predict(testing_generator))[:,1] # sklearn just takes prob of larger label
    
    y_test = []
    for x,y in testing_generator:
        y_test = y_test+y.tolist()
    
    auc = roc_auc_score(y_test, y_pred)
    false_pos, true_pos, thresholds = roc_curve(y_test, y_pred)
    
    roc_data = np.array([false_pos, true_pos, thresholds]).T
    
    m = modelContainer2('hara_model', model, history, roc_data, auc)
    
    return(m)


def cnn3D(n_epoch, batch_size, img_hw, n_frames, root_dir, trace=False):
    '''
    Input: 4D tensor (3D volumes of frames w/ 3 channels)
    Network: 
        Dense       2 nodes
        Dense       128 nodes
        Flatten
        MaxPool3D   (2, 2, 2) pool
        Conv3D      2 filters, (3,3,3) kernel, ReLU
        Rescaling   [-1,1]
    '''
    
    [training_generator, testing_generator] = make_generators(batch_size, img_hw, n_frames, root_dir)
 
    input_shape = (n_frames, img_hw, img_hw, 3) # does not include batch size

    inputs = keras.Input(shape=input_shape)
    x = layers.experimental.preprocessing.Rescaling(scale=1./127.5, offset=-1)(inputs) # or just 1./255
    x = layers.Conv3D(8, kernel_size=(3,3,3), activation='relu')(x)
    x = layers.Conv3D(8, kernel_size=(3,3,3), activation='relu')(x)
    x = layers.Conv3D(8, kernel_size=(3,3,3), activation='relu')(x)
    x = layers.MaxPool3D(pool_size=(2,4,4))(x)
    x = layers.Flatten()(x)
    x = layers.Dropout(0.5)(x)
    x = layers.Dense(64, activation='relu')(x)
    x = layers.Dropout(0.5)(x)
    outputs = layers.Dense(2)(x)
    
    model = keras.Model(inputs=inputs, outputs=outputs, name='cnn_3d')
    # print(model.summary())
    
    model.compile(
        loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        optimizer=keras.optimizers.RMSprop(),
        metrics=['accuracy'],
    )
    
    if trace:
        print('Training CNN 3D')
    history = model.fit(training_generator, epochs=n_epoch, verbose=2) # 1 for prog bar
    
    if trace:
        print('Testing CNN 3D')
    test_accr = model.evaluate(testing_generator, verbose=2) 

    y_pred = tf.nn.softmax(model.predict(testing_generator))[:,1] # sklearn just takes prob of larger label
    
    y_test = []
    for x,y in testing_generator:
        y_test = y_test+y.tolist()
    
    auc = roc_auc_score(y_test, y_pred)
    false_pos, true_pos, thresholds = roc_curve(y_test, y_pred)
    
    roc_data = np.array([false_pos, true_pos, thresholds]).T
    
    m = modelContainer('cnn3D_model', model, history, test_accr, roc_data, auc)
    
    return(m)




def res3D(n_epoch, batch_size, img_hw, n_frames, root_dir, trace=False):
    
    [training_generator, testing_generator] = make_generators(batch_size, img_hw, n_frames, root_dir)
    
    input_shape = (n_frames, img_hw, img_hw, 3)
    
    inputs = keras.Input(shape=input_shape)
    x = layers.experimental.preprocessing.Rescaling(scale=1./127.5, offset=-1)(inputs)
    x = layers.Conv3D(32, 3, activation='relu')(x)
    x = layers.Conv3D(64, 3, activation='relu')(x)
    block_1_output = layers.MaxPooling3D(3)(x)
    
    x = layers.Conv3D(64, 3, activation='relu', padding='same')(block_1_output)
    x = layers.Conv3D(64, 3, activation='relu', padding='same')(x)
    block_2_output = layers.add([x, block_1_output])
    
    x = layers.Conv3D(64, 3, activation='relu', padding='same')(block_2_output)
    x = layers.Conv3D(64, 3, activation='relu', padding='same')(x)
    block_3_output = layers.add([x, block_2_output])
    
    x = layers.Conv3D(64, 3, activation='relu')(block_3_output)
    x = layers.GlobalAveragePooling3D()(x)
    x = layers.Dense(256, activation='relu')(x)
    x = layers.Dropout(0.5)(x)
    outputs = layers.Dense(2)(x)
    
    model = keras.Model(inputs, outputs, name='resnet2D')
    #model.summary()
    
    model.compile(
        loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        optimizer=keras.optimizers.RMSprop(),
        metrics=["accuracy"],
    )
     
    if trace:
        print('Training ResNet 3D')
    history = model.fit(training_generator, epochs=n_epoch, verbose=2)
    
    if trace:
        print('Testing ResNet 3D')
    test_accr = model.evaluate(testing_generator, verbose=2) 
    
    y_pred = tf.nn.softmax(model.predict(testing_generator))[:,1] # sklearn just takes prob of larger label
    
    y_test = []
    for x,y in testing_generator:
        y_test = y_test+y.tolist()
    
    auc = roc_auc_score(y_test, y_pred)
    false_pos, true_pos, thresholds = roc_curve(y_test, y_pred)
    
    roc_data = np.array([false_pos, true_pos, thresholds]).T
    
    m = modelContainer('res3D_model', model, history, test_accr, roc_data, auc)
    
    return(m)



def mc2(n_epoch, batch_size, img_hw, n_frames, root_dir, trace=False):
    
    [training_generator, testing_generator] = make_generators(batch_size, img_hw, n_frames, root_dir)
 
    input_shape = (n_frames, img_hw, img_hw, 3) # does not include batch size
    flatten_time_dim = n_frames - (5-1) - (3-1) # result of Conv3Ds 
    
    inputs = keras.Input(shape=input_shape)
    x = layers.experimental.preprocessing.Rescaling(scale=1./127.5, offset=-1)(inputs) # or just 1./255
    x = layers.Conv3D(8, kernel_size=(5,5,5), activation='relu')(x)
    x = layers.Conv3D(8, kernel_size=(3,3,3), activation='relu')(x)
    
    x = layers.MaxPool3D(pool_size=(flatten_time_dim, 2, 2))(x)
    x = tf.squeeze(x, axis=1)
    
    x = layers.Conv2D(32, 3, activation='relu')(x)
    x = layers.Conv2D(32, 3, activation='relu')(x)
    x = layers.MaxPooling2D()(x)
    x = layers.Flatten()(x)
    x = layers.Dropout(0.5)(x)
    x = layers.Dense(64, activation='relu')(x)
    x = layers.Dropout(0.5)(x)
    outputs = layers.Dense(2)(x)
    
    model = keras.Model(inputs=inputs, outputs=outputs, name='cnn_3d')
    # print(model.summary())
    
    model.compile(
        loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        optimizer=keras.optimizers.RMSprop(),
        metrics=['accuracy'],
    )
    
    if trace:
        print('Training MC2')
    history = model.fit(training_generator, epochs=n_epoch, verbose=2) # 1 for prog bar
    
    if trace:
        print('Testing MC2')
    test_accr = model.evaluate(testing_generator, verbose=2) 

    y_pred = tf.nn.softmax(model.predict(testing_generator))[:,1] # sklearn just takes prob of larger label
    
    y_test = []
    for x,y in testing_generator:
        y_test = y_test+y.tolist()
    
    auc = roc_auc_score(y_test, y_pred)
    false_pos, true_pos, thresholds = roc_curve(y_test, y_pred)
    
    roc_data = np.array([false_pos, true_pos, thresholds]).T
    
    m = modelContainer('mc2_model', model, history, test_accr, roc_data, auc)
    
    return(m)


