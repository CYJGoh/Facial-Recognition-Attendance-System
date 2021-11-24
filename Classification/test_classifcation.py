import matplotlib.pyplot as plt
import numpy as np
import os
import tensorflow as tf
import tensorflow_addons as tfa
import tensorflow_hub as hub
from  tensorflow_addons.optimizers import MultiOptimizer
from tensorflow.keras import layers, models, losses, optimizers, metrics, regularizers, applications, Model

#Avoid out of memory errors
from tensorflow.keras.preprocessing import image_dataset_from_directory
config = tf.compat.v1.ConfigProto(gpu_options = 
                         tf.compat.v1.GPUOptions(per_process_gpu_memory_fraction=0.8)
# device_count = {'GPU': 1}
)
config.gpu_options.allow_growth = True
session = tf.compat.v1.Session(config=config)
tf.compat.v1.keras.backend.set_session(session)

#path to classification data folder
dir = "C:\\Users\\101232163\\Downloads\\Face_recognition_dataset\\classification_data"


test_dir = os.path.join(dir, 'test_data') #test data folder path
val_dir = os.path.join(dir, 'val_data') #validation data folder path

BATCH_SIZE = 32



#create the model without any weights, this is used when needing to load weights of a previous model
def create_model():
    base_m = applications.MobileNetV2(input_shape=IMG_SHAPE,
                                      include_top=False,
                                      alpha = ALPHA) #create mobilenetv2 model without weights
    base_m.trainable = False
    global_average_pooling_layer = layers.GlobalAveragePooling2D() #Global average pooling layer to convert ouput of base to a vector
    prediction_layer = layers.Dense(4000) #the classification layer has 4000 neurons corresponding to each class
    
    inputs = tf.keras.Input(shape=IMG_SHAPE) #input layer
    layer = preprocess_input(inputs) 
    layer = base_m(layer, training=False)
    global_avg = global_average_pooling_layer(layer)
    droput_layer = layers.Dropout(0.2)(global_avg)
    outputs = prediction_layer(droput_layer)
    model = Model(inputs, outputs)
    
    #compile with adam optimizer with lr set to 0.0001
    model.compile(optimizer=optimizers.Adam(0.0001),
              loss=losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])
    
    return model

#used to evaluate the model on test and validation data, shows the loss and accuracy
def model_evaluate(mdl):
    val_loss, val_acc = mdl.evaluate(val_ds)
    test_loss, test_acc = mdl.evaluate(test_ds)
    print("validation dataset loss: {:.2f}".format(val_loss))
    print("validation dataset accuracy: {:.2f}".format(val_acc))
    
    print("test dataset loss: {:.2f}".format(test_loss))
    print("test dataset accuracy: {:.2f}".format(test_acc))
    

#create model 1
IMG_SIZE = (160, 160) #Input resolution of network
ALPHA = 1.0 #Width multiplier
IMG_SHAPE = IMG_SIZE + (3,) #RGB hence add 3


test_ds = image_dataset_from_directory(test_dir,
                                       shuffle=True,
                                       batch_size=BATCH_SIZE,
                                       image_size=IMG_SIZE)

val_ds = image_dataset_from_directory(val_dir,
                                      shuffle=True,
                                      batch_size=BATCH_SIZE,
                                      image_size=IMG_SIZE)



preprocess_input = applications.mobilenet_v2.preprocess_input

model1 = create_model()
#Load the saved weights of the trained model, parameter is the path to the weights of the model
model1.load_weights('C:\\Users\\101232163\Desktop\\Final\\model1\\model_top_124_base_diff_lr_re_low')
model_evaluate(model1)

del test_ds, val_ds

#create model 2
IMG_SIZE = (224, 224) #Input resolution of network
ALPHA = 1.4 #Width multiplier
IMG_SHAPE = IMG_SIZE + (3,) #RGB hence add 3


test_ds = image_dataset_from_directory(test_dir,
                                       shuffle=True,
                                       batch_size=BATCH_SIZE,
                                       image_size=IMG_SIZE)

val_ds = image_dataset_from_directory(val_dir,
                                      shuffle=True,
                                      batch_size=BATCH_SIZE,
                                      image_size=IMG_SIZE)


preprocess_input = applications.mobilenet_v2.preprocess_input

model2 = create_model()
#Load the saved weights of the trained model, parameter is the path to the weights of the model
model2.load_weights('C:\\Users\\101232163\Desktop\\Final\\model2\\model_top_134_base_diff_lr_lowered_low_1e')
model_evaluate(model2)