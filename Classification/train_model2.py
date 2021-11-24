#This python file is used to train model 2 of classification approach discussed in report

import matplotlib.pyplot as plt
import numpy as np
import os
import tensorflow as tf
import tensorflow_addons as tfa
import tensorflow_hub as hub
from  tensorflow_addons.optimizers import MultiOptimizer
from tensorflow.keras import layers, models, losses, optimizers, metrics, regularizers, applications, Model


#Avoid out of memory errors by allowing growth
from tensorflow.keras.preprocessing import image_dataset_from_directory
config = tf.compat.v1.ConfigProto(gpu_options = 
                         tf.compat.v1.GPUOptions(per_process_gpu_memory_fraction=0.8)
# device_count = {'GPU': 1}
)
config.gpu_options.allow_growth = True
session = tf.compat.v1.Session(config=config)
tf.compat.v1.keras.backend.set_session(session)

log_dir = "C:\\Users\\101232163\\Downloads\\Assignment_2\\tensorboard_logs134_4000_224_1_4\\"
dir = "C:\\Users\\101232163\\Downloads\\Face_recognition_dataset\\classification_data"

train_dir = os.path.join(dir, 'train_data')
test_dir = os.path.join(dir, 'test_data')
val_dir = os.path.join(dir, 'val_data')

BATCH_SIZE = 32 #set batch size
IMG_SIZE = (224, 224) #Input resolution of network
ALPHA = 1.4 #Width multiplier of network
IMG_SHAPE = IMG_SIZE + (3,) #RGB image hence add 3 to size to get image shape
TOTAL_EPOCHS = 0 #Used to count toal number of epochs

train_ds = image_dataset_from_directory(train_dir,
                                        shuffle=True,
                                        batch_size=BATCH_SIZE,
                                        image_size=IMG_SIZE)

test_ds = image_dataset_from_directory(test_dir,
                                       shuffle=True,
                                       batch_size=BATCH_SIZE,
                                       image_size=IMG_SIZE)

val_ds = image_dataset_from_directory(val_dir,
                                      shuffle=True,
                                      batch_size=BATCH_SIZE,
                                      image_size=IMG_SIZE)



#Scale the input pixel values in range of -1 and 1
preprocess_input = applications.mobilenet_v2.preprocess_input

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

#Same as earlier but instead of not loading any weight it creates a model with mobilnetV2 pre trained on imagenet
def create_model_with_imagenet():
    base_m = applications.MobileNetV2(input_shape=IMG_SHAPE,
                                      include_top=False,
                                      weights='imagenet',
                                      alpha = ALPHA)
    base_m.trainable = False
    global_average_pooling_layer = layers.GlobalAveragePooling2D()
    prediction_layer = layers.Dense(4000)
    
    inputs = tf.keras.Input(shape=IMG_SHAPE)
    layer = preprocess_input(inputs)
    layer = base_m(layer, training=False)
    global_avg = global_average_pooling_layer(layer)
    droput_layer = layers.Dropout(0.2)(global_avg)
    outputs = prediction_layer(droput_layer)
    model = Model(inputs, outputs)
    
    model.compile(optimizer=optimizers.Adam(0.0001),
              loss=losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])
    return model

#Used to freeze layers in a give model, note that it is not inclusize of end meaning if start=0 and end=10 it only freezes upto 9
def freeze_between(start, end, mdl):
    for layer in mdl.layers[start:end]:
        layer.trainable =  False

#freezes all layers before the value for layer_no
def freeze_before(layer_no, mdl):
    for layer in mdl.layers[:layer_no]:
        layer.trainable =  False
    
#callbacks for tensorboard logs
def get_callbacks(name):
    return [tf.keras.callbacks.TensorBoard(log_dir=log_dir + name)]


#used to train layers of model after the given layer number with given epochs, name is the name to store tensorboard logs
#previous history is the latest history of the model, if model wasn't trained before, use model.fit() as done normally
#mdl is the model to be trained
#lr is the learning rate
def train_after_layer_base(mdl, layer_no, epochs, lr, previous_history, name):
    global TOTAL_EPOCHS
    mdl_base = mdl.get_layer('mobilenetv2_1.40_224')
    mdl_base.trainable =True
    freeze_between(0, layer_no, mdl_base)
    mdl.compile(optimizer=optimizers.Adam(lr),
              loss=losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])
    TOTAL_EPOCHS += epochs    
    history = mdl.fit(train_ds,
                      epochs=TOTAL_EPOCHS,
                      initial_epoch=previous_history.epoch[-1],
                      validation_data=val_ds,
                      callbacks=get_callbacks(name))
    return history

#same as before but instead of history it takes in the total number of epochs the model has been trained for till now
def train_after_layer_base_int(mdl, layer_no, epochs, lr, previous, name):
    global TOTAL_EPOCHS
    mdl_base = mdl.get_layer('mobilenetv2_1.40_224')
    mdl_base.trainable =True
    freeze_between(0, layer_no, mdl_base)
    mdl.compile(optimizer=optimizers.Adam(lr),
              loss=losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])
    TOTAL_EPOCHS += epochs    
    history = mdl.fit(train_ds,
                      epochs=TOTAL_EPOCHS,
                      initial_epoch=previous,
                      validation_data=val_ds,
                      callbacks=get_callbacks(name))
    return history

#unlike the train_after_layer methods, the compile_and_fit methods do not freeze layers of the given model, freezing has to be
#using the earlier freeze functions.
#It also does not take in lr instead takes in the optimzer as a whole
def compile_and_fit(mdl, epochs, previous_history, optimizer, name):
    global TOTAL_EPOCHS
    mdl.compile(optimizer=optimizer,
              loss=losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])
    TOTAL_EPOCHS += epochs    
    history = mdl.fit(train_ds,
                           epochs=TOTAL_EPOCHS,
                           initial_epoch=previous_history.epoch[-1],
                           validation_data=val_ds,
                           callbacks=get_callbacks(name))
    return history 
#same as before but uses total number of epochs the model has been trained until now instead of history
def compile_and_fit_int(mdl, epochs, previous_epoch, optimizer, name):
    global TOTAL_EPOCHS
    mdl.compile(optimizer=optimizer,
              loss=losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])
    TOTAL_EPOCHS += epochs    
    history = mdl.fit(train_ds,
                           epochs=TOTAL_EPOCHS,
                           initial_epoch=previous_epoch,
                           validation_data=val_ds,
                           callbacks=get_callbacks(name))
    return history 

#Evaluates the model on the test and validation dataset, gives the accuracy and loss
def model_evaluate(mdl):
    val_loss, val_acc = mdl.evaluate(val_ds)
    test_loss, test_acc = mdl.evaluate(test_ds)
    print("validation dataset loss: {:.2f}".format(val_loss))
    print("validation dataset accuracy: {:.2f}".format(val_acc))
    
    print("test dataset loss: {:.2f}".format(test_loss))
    print("test dataset accuracy: {:.2f}".format(test_acc))

#Train top first
model = create_model_with_imagenet()
TOTAL_EPOCHS = 5
name = "model_with_top_trained"
history_1 = model.fit(train_ds,
                      epochs=TOTAL_EPOCHS,
                      validation_data=val_ds,
                      callbacks=get_callbacks(name))

model_evaluate(model)
model.save('saved_models134_224_1_4/'+name) #save the model at the given loaction with given name

#train first 54 layers and top with a lower lr than before
name = "model_top_54_base"

history_2 = train_after_layer_base(model, 100, 5, 0.00001, history_1, name)

model_evaluate(model)
model.save('saved_models134_224_1_4/'+name)


#Fine tune till 134 layers
base_m = model.get_layer('mobilenetv2_1.40_224')
base_m.trainable = True
name = "model_top_134_base_diff_lr"
freeze_before(20, base_m) #refreeze first 20 layers
optimizers_and_layers = [(optimizers.Adam(0.0001), model.layers[-3:]), #The layers on top of the base
                         (optimizers.Adam(0.00006), base_m.layers[100:155]), #Final 54 layers
                         (optimizers.Adam(0.00004), base_m.layers[50:100]), #middle 50
                         (optimizers.Adam(0.00001), base_m.layers[20:50])] #30 layers after layer number 19

optimizer = tfa.optimizers.MultiOptimizer(optimizers_and_layers)
history_3 = compile_and_fit(model, 9, history_2, optimizer, name)

model_evaluate(model)
model.save_weights('./checkpoints134_224_1_4/'+name)

#Above with layer lr's lowered
name = "model_top_134_base_diff_lr_lowered"
optimizers_and_layers = [(optimizers.Adam(0.00005), model.layers[-3:]), #The layers on top of the base
                         (optimizers.Adam(0.00002), base_m.layers[100:155]), #Final 54 layers
                         (optimizers.Adam(0.00001), base_m.layers[50:100]), #middle 50
                         (optimizers.Adam(0.000008), base_m.layers[20:50])] #30 layers after layer number 19

optimizer = tfa.optimizers.MultiOptimizer(optimizers_and_layers)
history_5 = compile_and_fit_int(model, 3, 19, optimizer, name)

model_evaluate(model)
model.save_weights('./checkpoints134_224_1_4/'+name)


#ABove with slighlty higher learning rates
name = "model_top_134_base_diff_lr_lowered_low_1e"

optimizers_and_layers = [(optimizers.Adam(0.00001), model.layers[-3:]), #The layers on top of the base
                         (optimizers.Adam(0.000001), base_m.layers[20:155])] #Final 134 layers

optimizer = tfa.optimizers.MultiOptimizer(optimizers_and_layers)
history_6 = compile_and_fit_int(model, 1, 22, optimizer, name)

model_evaluate(model)
model.save_weights('./checkpoints134_224_1_4/'+name)

#Repeat of above for another epoch
""" #Running this part did not improve anymore and loss increase
name = "model_top_134_base_diff_lr_lowered_low_2e"

optimizers_and_layers = [(optimizers.Adam(0.00001), model.layers[-3:]), #The layers on top of the base
                         (optimizers.Adam(0.000001), base_m.layers[20:155])] #Final 54 layers

optimizer = tfa.optimizers.MultiOptimizer(optimizers_and_layers)
history_7 = compile_and_fit_int(model, 1, 23, optimizer, name)

model_evaluate(model)
model.save_weights('./checkpoints134_224_1_4/'+name)
"""
