# Import standard dependencies
import cv2
import os
import random
import numpy as np
from matplotlib import pyplot as plt

# Import tensorflow dependencies - Functional API
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Layer, Conv2D, Dense, MaxPooling2D, Input, Flatten
from tensorflow.keras.metrics import Precision, Recall, BinaryAccuracy, AUC
import tensorflow as tf
from tensorflow.keras import layers, models, losses, optimizers, metrics, regularizers, applications, Model



# Avoid OOM errors by setting GPU Memory Consumption Growth
gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus: 
    tf.config.experimental.set_memory_growth(gpu, True)
os.environ["CUDA_VISIBLE_DEVICES"]= "0"
IMG_SIZE = (224, 224)
IMG_SHAPE = IMG_SIZE + (3,)
ALPHA = 1.4

def preprocess(file_path):
    
    # Read in image from file path
    byte_img = tf.io.read_file(file_path)
    # Load in the image 
    img = tf.io.decode_jpeg(byte_img)
    
    # Preprocessing steps - resizing the image to be 224x224x3
    img = tf.image.resize(img, IMG_SIZE)

    # Return image
    return img

def preprocess_twin(input_img, validation_img, label):
    return(preprocess(input_img), preprocess(validation_img), label)

def create_model(weights):
    preprocess_input = applications.mobilenet_v2.preprocess_input
    base_m = applications.MobileNetV2(input_shape=IMG_SHAPE,
                                      include_top=False,
                                      alpha = ALPHA)
    global_average_pooling_layer = layers.GlobalAveragePooling2D()
    dense_layer = layers.Dense(4000, activation='softmax')
    inputs = tf.keras.Input(shape=IMG_SHAPE)
    x = preprocess_input(inputs)
    x = base_m(x, training=False)
    x = global_average_pooling_layer(x)
    x = layers.Dropout(0.2)(x)
    outputs = dense_layer(x)
    model = Model(inputs, outputs)    
    model.load_weights(weights)
        
    return model

def make_embedding(weights): 
    preprocess_input = applications.mobilenet_v2.preprocess_input

    model = create_model(weights)
    base_m = model.get_layer('mobilenetv2_1.40_224')
    global_average_pooling_layer = layers.GlobalAveragePooling2D()
    inputs = tf.keras.Input(shape=IMG_SHAPE)
    x = preprocess_input(inputs)
    x = base_m(x, training=False)
    outputs = global_average_pooling_layer(x)
    new_model = Model(inputs, outputs, name='embedding')
    
    return new_model
    
    

# Siamese L1 Distance class
class L1Dist(Layer):
    
    # Init method - inheritance
    def __init__(self, **kwargs):
        super().__init__()
       
    # Magic happens here - similarity calculation
    def call(self, input_embedding, validation_embedding):
        return tf.math.abs(input_embedding - validation_embedding)
    
# Siamese L2 Distance class (Euclidean)
class L2Dist(Layer):
    
    # Init method - inheritance
    def __init__(self, **kwargs):
        super().__init__()
       
    # Magic happens here - similarity calculation    
    def call(self, input_embedding, validation_embedding):
        distance = tf.math.square(input_embedding - validation_embedding)
        return tf.math.sqrt(distance)

# Siamese Cosine Similarity class
class CosDist(Layer):
    
    # Init method - inheritance
    def __init__(self, **kwargs):
        super().__init__()
       
    # Magic happens here - similarity calculation    
    def call(self, input_embedding, validation_embedding):
        numerator = tf.math.multiply(input_embedding, validation_embedding)
        sum_input = tf.math.square(input_embedding)
        sum_validation = tf.math.square(validation_embedding)
        denominator = round(tf.math.multiply(tf.math.sqrt(sum_input), tf.math.sqrt(sum_validation)))
        return numerator/denominator

def make_siamese_model(): 
    
    # Anchor image input in the network
    input_image = Input(name='input_img', shape=IMG_SHAPE)
    
    # Validation image in the network 
    validation_image = Input(name='validation_img', shape=IMG_SHAPE)
    
    # Combine siamese distance components
    siamese_layer = L2Dist()
    siamese_layer._name = 'distance'
    distances = siamese_layer(embedding(input_image), embedding(validation_image))
    
    # Classification layer 
    classifier = Dense(1, activation='sigmoid')(distances)
    
    return Model(inputs=[input_image, validation_image], outputs=classifier, name='SiameseNetwork')

#create embedding with weights from saved model, the parameter should be path to weights of saved model
#embedding = make_embedding('C:\\Users\\101232163\\Desktop\\Final\\model2\\model_top_134_base_diff_lr_lowered_low_1e')
#siamese_model = make_siamese_model()

siamese_model = tf.keras.models.load_model('siamesemodel_classification_embedding_10eps_50%_L1.h5', 
                                            custom_objects={'L1Dist':L1Dist, 'BinaryCrossentropy':tf.losses.BinaryCrossentropy})

# Verification data
"""
verify_data = tf.data.TextLineDataset("verification_pairs_val.txt")

anc, pos_neg, lbl = [], [], []
count = 0
for element in verify_data:
    count += 1
    print(count)
    element_split = tf.strings.split(element, sep=" ")
    anc.append(element_split[0])
    pos_neg.append(element_split[1])
    lbl.append(int(element_split[2]))

anc = tf.data.Dataset.from_tensor_slices(anc)
pos_neg = tf.data.Dataset.from_tensor_slices(pos_neg)
lbl = tf.data.Dataset.from_tensor_slices(lbl)

verify_data = tf.data.Dataset.zip((anc, pos_neg, lbl))
tf.data.experimental.save(verify_data, path="verify_data")
print(verify_data.element_spec)"""

verify_data = tf.data.experimental.load(path="verify_data", element_spec=(tf.TensorSpec(shape=(), dtype=tf.string, name=None), 
                                                                        tf.TensorSpec(shape=(), dtype=tf.string, name=None), 
                                                                        tf.TensorSpec(shape=(), dtype=tf.int32, name=None)))
verify_data = verify_data.map(preprocess_twin)
verify_data = verify_data.cache()
verify_data = verify_data.batch(15)

test_input, test_val, y_true = verify_data.as_numpy_iterator().next()
print(test_input[0].shape)
print(test_val[0].shape)


r = Recall()
p = Precision()
a = BinaryAccuracy()
auc = AUC()

"""y_hat = siamese_model.predict([test_input, test_val])
yhat_ravel = y_hat.ravel()
print(y_hat)
print(yhat_ravel)
print(y_true)
r.update_state(y_true, yhat_ravel)
p.update_state(y_true,yhat_ravel)
a.update_state(y_true, yhat_ravel) """

count = 0
y_true, y_pred = [], []

for test_input, test_val, y in verify_data.as_numpy_iterator():
    count += 1
    print(count)
    yhat = siamese_model.predict([test_input, test_val])
    yhat_ravel = yhat.ravel()
    r.update_state(y, yhat)
    p.update_state(y, yhat)
    a.update_state(y, yhat)
    auc.update_state(y, yhat)

    for i in range(len(y)):
        y_true.append(y[i])
        y_pred.append(yhat_ravel[i])
    
print("Recall:", r.result().numpy(), "Precision:", p.result().numpy(), "Accuracy:", a.result().numpy(), "AUC:", auc.result().numpy())

from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt

fpr_keras, tpr_keras, thresholds_keras = roc_curve(y_true, y_pred)
auc_keras = auc(fpr_keras, tpr_keras)

plt.plot(fpr_keras, tpr_keras, label='Keras (area = {:.3f})'.format(auc_keras))
plt.xlabel('False positive rate')
plt.ylabel('True positive rate')
plt.title('ROC curve')
plt.legend(loc='best')
plt.show()