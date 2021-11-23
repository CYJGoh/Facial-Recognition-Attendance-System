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

# Avoid OOM errors by setting GPU Memory Consumption Growth
gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus: 
    tf.config.experimental.set_memory_growth(gpu, True)
os.environ["CUDA_VISIBLE_DEVICES"]= "0"

def preprocess(file_path):
    
    # Read in image from file path
    byte_img = tf.io.read_file(file_path)
    # Load in the image 
    img = tf.io.decode_jpeg(byte_img)
    
    # Preprocessing steps - resizing the image to be 100x100x3
    img = tf.image.resize(img, (105,105))
    # Scale image to be between 0 and 1 
    img = img / 255.0

    # Return image
    return img

def preprocess_twin(input_img, validation_img, label):
    return(preprocess(input_img), preprocess(validation_img), label)

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
        #distance = tf.math.square(input_embedding - validation_embedding)
        return tf.math.sqrt(tf.math.square(input_embedding - validation_embedding))

# Load model
siamese_model = tf.keras.models.load_model('models\siamesemodel_5eps_50%_L1.h5', 
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
verify_data = verify_data.batch(16)

r = Recall()
p = Precision()
a = BinaryAccuracy()
auc = AUC()

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
