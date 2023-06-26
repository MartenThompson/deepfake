import numpy as np
import os
from PIL import Image 
import tensorflow as tf
#tf.config.threading.set_inter_op_parallelism_threads(20)
#tf.config.threading.set_intra_op_parallelism_threads(20)
#tf.random.set_seed(8008)
#np.random.seed(808)


#import keras
from tensorflow.keras.utils import Sequence
import numpy.random as rnd

import shutil


def copy_train_test(root, alpha, trace):
    # get list of all folders in root/original/fake
        # copy alpha% of them into root/runtime/training/fake
        # copy remaining into root/runtime/test/fake

    # get list of all folders in root/original/real
        # copy alpha% of them into root/runtime/training/real
        # copy remaining into root/runtime/test/real
        
    rnd.seed(320)

    og_fake_dirs = np.array(os.listdir('../'+root+'/original/fake'))
    og_real_dirs = np.array(os.listdir('../'+root+'/original/real'))
    n_f = len(og_fake_dirs)
    n_r = len(og_real_dirs)
    
    indices_fake = np.arange(n_f)
    train_indices_fake = rnd.choice(indices_fake, int(alpha*n_f), replace=False)
    test_indices_fake = np.delete(indices_fake, train_indices_fake)
    runtime_train_fake_dirs = og_fake_dirs[train_indices_fake]
    runtime_test_fake_dirs = og_fake_dirs[test_indices_fake]
    
    indices_real = np.arange(n_r)
    train_indices_real = rnd.choice(indices_real, int(alpha*n_r), replace=False)
    test_indices_real = np.delete(indices_real, train_indices_real)
    runtime_train_real_dirs = og_real_dirs[train_indices_real]
    runtime_test_real_dirs = og_real_dirs[test_indices_real]
    
    if trace:
        print('Train fake:', runtime_train_fake_dirs)
        print('Test fake:', runtime_test_fake_dirs)
        
        print('Train real:', runtime_train_real_dirs)
        print('Test real:', runtime_test_real_dirs)
    
    print('Copying train fake...')
    # Copy fake (train)
    for i in range(len(runtime_train_fake_dirs)):
        src = '../'+root+'/original/fake/' + runtime_train_fake_dirs[i]
        dest = '../'+root+'/runtime/train/fake/'  + runtime_train_fake_dirs[i]
        shutil.copytree(src, dest) 
    
    print('Complete')
    print('Copying test fake...')
    # fake (test)
    for i in range(len(runtime_test_fake_dirs)):
        src = '../'+root+'/original/fake/' + runtime_test_fake_dirs[i]
        dest = '../'+root+'/runtime/test/fake/'  + runtime_test_fake_dirs[i]
        shutil.copytree(src, dest) 
    
    print('Complete')
    print('Copying train real...')    
    # real (train)
    for i in range(len(runtime_train_real_dirs)):
        src = '../'+root+'/original/real/' + runtime_train_real_dirs[i]
        dest = '../'+root+'/runtime/train/real/'  + runtime_train_real_dirs[i]
        shutil.copytree(src, dest) 
    
    print('Complete')
    print('Copying test real...')
    # real (test)
    for i in range(len(runtime_test_real_dirs)):
        src = '../'+root+'/original/real/' + runtime_test_real_dirs[i]
        dest = '../'+root+'/runtime/test/real/'  + runtime_test_real_dirs[i]
        shutil.copytree(src, dest) 

    print('Complete')
    

def make_2D_datasets(batch_size, img_hw, root_dir):
    
    if False == os.path.isdir('../'+root_dir+'/runtime/train'):
        copy_train_test(root_dir, 0.8, True)
    
    train_ds = tf.keras.preprocessing.image_dataset_from_directory(
        directory='../'+root_dir+'/runtime/train',
        color_mode='rgb',
        image_size=(img_hw, img_hw),
        batch_size=batch_size)
    
    test_ds = tf.keras.preprocessing.image_dataset_from_directory(
      directory='../'+root_dir+'/runtime/test',
      color_mode='rgb',
      image_size=(img_hw, img_hw),
      batch_size=batch_size)
    

    train_ds = train_ds.prefetch(buffer_size=1)
    test_ds = test_ds.prefetch(buffer_size=1)
    
    return([train_ds, test_ds])


def save_npy(root_dir, train_test, fake_real, img_hw, movie_length, create_3D=False):
    
    # should only be True once
    if create_3D:
        os.makedirs('../'+root_dir+'/3D')
        os.makedirs('../'+root_dir+'/3D/data')    
    
    if 'fake' == fake_real:
        lab = 0
    else:
        lab = 1
    
    labels = {}
    

    
    dirs = np.array(os.listdir('../'+root_dir+'/runtime/'+train_test+'/'+fake_real))
    
    for folder in dirs:
        all_images = np.array(os.listdir('../'+root_dir+'/runtime/'+train_test+'/'+fake_real+'/'+folder))
        n_images = len(all_images)
        
        n_movies = int(n_images/movie_length)
        
        itr = 0
        for m in range(n_movies):
            movie = np.empty([movie_length, img_hw, img_hw, 3])
            movie_name = folder + str(m)
            labels[movie_name] = lab
            
            for i in range(movie_length):
                movie[i,:,:,:] = Image.open('../'+root_dir+'/runtime/'+train_test+'/'+fake_real+'/'+folder +'/' + all_images[itr])
                itr+=1
                
            #np.save('../'+root_dir+'/runtime/'+train_test+'/'+fake_real+'/3D/' + movie_name, movie)
            np.save('../'+root_dir+'/3D/data/'+ movie_name, movie)

    
    return(labels)
    

def make_3D_npy(root_dir, img_hw, movie_length):
    
    # artifact of earlier model building; this method expects such a file org
    if False == os.path.isdir('../'+root_dir+'/runtime/train'):
        copy_train_test(root_dir, 0.8, True)
    
    
    if False == os.path.isdir('../'+root_dir+'/3D'):
        # hasn't been run yet
        
        print('packing 3D arrays...')
        train_fake_labels = save_npy(root_dir, 'train', 'fake', img_hw, movie_length, True)
        print('1/4')
        train_real_labels = save_npy(root_dir, 'train', 'real', img_hw, movie_length)        
        print('2/4')
        test_fake_labels = save_npy(root_dir, 'test', 'fake', img_hw, movie_length)
        print('3/4')
        test_real_labels = save_npy(root_dir, 'test', 'real', img_hw, movie_length)
        print('4/4')        
        # 'asdadsffsdaf0':0 or 1
        all_labels = {**train_fake_labels, **train_real_labels, **test_fake_labels, **test_real_labels}
        
        partition = {}
        partition['train'] = list(train_real_labels.keys()) + list(train_fake_labels.keys())
        partition['test'] = list(test_real_labels.keys()) + list(test_fake_labels.keys())              
        
        # save labels and parition        
        np.save('../'+root_dir+'/3D/partition.npy', partition) 
        np.save('../'+root_dir+'/3D/labels.npy', all_labels) 
        
        return([partition, all_labels])
        
    else:
        print('reading partition and labels from existing data')
        partition = np.load('../'+root_dir+'/3D/partition.npy', allow_pickle='TRUE').item()
        labels = np.load('../'+root_dir+'/3D/labels.npy', allow_pickle='TRUE').item()
        return([partition, labels])
    
    # fin.




#class DataGenerator(keras.utils.Sequence):
class DataGenerator(Sequence):
    'Generates data for Keras'
    
    def __init__(self, root_dir, list_IDs, labels, batch_size=32, dim=(16,160,160,3), n_channels=3, n_classes=2, shuffle=True):
        'Initialization'
        self.root_dir = root_dir
        self.dim = dim
        self.batch_size = batch_size
        self.labels = labels
        self.list_IDs = list_IDs
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.shuffle = shuffle
        self.on_epoch_end()
        

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(len(self.list_IDs) / self.batch_size))

    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]

        # Find list of IDs
        list_IDs_temp = [self.list_IDs[k] for k in indexes]

        # Generate data
        X, y = self.__data_generation(list_IDs_temp)

        return X, y

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(len(self.list_IDs))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def __data_generation(self, list_IDs_temp):
        'Generates data containing batch_size samples' # X : (n_samples, *dim, n_channels)
        # Initialization
        #X = np.empty((self.batch_size, *self.dim, self.n_channels))
        X = np.empty((self.batch_size, *self.dim))
        y = np.empty((self.batch_size), dtype=int)

        # Generate data
        for i, ID in enumerate(list_IDs_temp):
            # Store sample
            #X[i,] = np.load('data/' + ID + '.npy')
            X[i,] = np.load('../' + self.root_dir + '/3D/data/' + ID + '.npy')

            # Store class
            y[i] = np.array(self.labels[ID])

        #return X, keras.utils.to_categorical(y, num_classes=self.n_classes)
        return X, y


def make_generators(batch_size, img_hw, n_frames, root_dir):
    params = {'dim': (n_frames, img_hw, img_hw, 3), # last dim number of channels
          'batch_size': batch_size,
          'n_classes': 2,
          'shuffle': True}

    [partition, labels] = make_3D_npy(root_dir, img_hw, n_frames)

    training_generator = DataGenerator(root_dir, partition['train'], labels, **params)
    testing_generator = DataGenerator(root_dir, partition['test'], labels, **params)
    
    return([training_generator, testing_generator])








class DataGenerator224(Sequence):
    'Generates data for Keras'
    
    def __init__(self, root_dir, list_IDs, labels, batch_size=32, dim=(16,224,224,3), n_channels=3, n_classes=2, shuffle=True):
        'Initialization'
        self.root_dir = root_dir
        self.dim = dim
        self.batch_size = batch_size
        self.labels = labels
        self.list_IDs = list_IDs
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.shuffle = shuffle
        self.on_epoch_end()
        

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(len(self.list_IDs) / self.batch_size))

    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]

        # Find list of IDs
        list_IDs_temp = [self.list_IDs[k] for k in indexes]

        # Generate data
        X, y = self.__data_generation(list_IDs_temp)

        return X, y

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(len(self.list_IDs))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def __data_generation(self, list_IDs_temp):
        'Generates data containing batch_size samples' # X : (n_samples, *dim, n_channels)
        # Initialization
        #X = np.empty((self.batch_size, *self.dim, self.n_channels))
        X = np.empty((self.batch_size, *self.dim))
        y = np.empty((self.batch_size), dtype=int)

        # Generate data
        for i, ID in enumerate(list_IDs_temp):
            # Store sample
            #X[i,] = np.load('data/' + ID + '.npy')
            #x = np.load('../' + self.root_dir + '/3D/data/' + ID + '.npy')
            X[i,] = tf.image.resize(np.load('../' + self.root_dir + '/3D/data/' + ID + '.npy'), [224,224])
            
            # Store class
            y[i] = np.array(self.labels[ID])

        #return X, keras.utils.to_categorical(y, num_classes=self.n_classes)
        return X, y







def make_generators224(batch_size, img_hw, n_frames, root_dir):
    params = {'dim': (n_frames, 224, 224, 3), # last dim number of channels
          'batch_size': batch_size,
          'n_classes': 2,
          'shuffle': True}

    [partition, labels] = make_3D_npy(root_dir, img_hw, n_frames)

    training_generator = DataGenerator224(root_dir, partition['train'], labels, **params)
    testing_generator = DataGenerator224(root_dir, partition['test'], labels, **params)
    
    return([training_generator, testing_generator])



'''
params = {'dim': (16,160,160,3),
          'batch_size': 3,
          'n_classes': 2,
          'shuffle': True}

# Datasets
root_dir = 'small_local'
[partition, labels] = make_3D_npy(root_dir, 160, 16)

# Generators
training_generator = DataGenerator(root_dir, partition['train'], labels, **params)
#validation_generator = DataGenerator(partition['validation'], labels, **params)

itr = 0
for x,y in training_generator:
    print('y',type(y), y.shape)
    print('x',type(x), x.shape)
'''

# Design model
#model = Sequential()
#[...] # Architecture
#model.compile()

# Train model on dataset
#model.fit_generator(generator=training_generator,
#                    validation_data=training_generator,
#                    use_multiprocessing=True,
#                    workers=6)





#x = np.load('temp.npy')






