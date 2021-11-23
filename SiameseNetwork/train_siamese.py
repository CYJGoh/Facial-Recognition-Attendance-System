# Import standard dependencies
import cv2
import os
import random
import numpy as np
from matplotlib import pyplot as plt

# Import tensorflow dependencies - Functional API
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Layer, Conv2D, Dense, MaxPooling2D, Input, Flatten, Lambda
from tensorflow.keras.metrics import Precision, Recall
import tensorflow as tf
from tensorflow.python.framework import dtypes
import tensorflow_datasets as tfds

# Avoid OOM errors by setting GPU Memory Consumption Growth
gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus: 
    tf.config.experimental.set_memory_growth(gpu, True)
os.environ["CUDA_VISIBLE_DEVICES"]= "0"

# Setup paths
POS_PATH = os.path.join('data', 'positive')
NEG_PATH = os.path.join('data', 'negative')
ANC_PATH = os.path.join('data', 'anchor')

"""anchor = tf.data.Dataset.list_files(ANC_PATH+'/*/*.jpg', shuffle = False)
positive = tf.data.Dataset.list_files(POS_PATH+'/*/*.jpg', shuffle = False)
negative = tf.data.Dataset.list_files(NEG_PATH+'/*/*.jpg')

anchor.map(preprocess)
positive.map(preprocess)
negative.map(preprocess)

# Label preparation

anc_neg, neg = [], []
count = 0
negative_iter = negative.as_numpy_iterator()

for img in anchor.as_numpy_iterator():
    negative_img = negative_iter.next()
    img_split = tf.strings.split(img, sep="\\")
    negative_img_split = tf.strings.split(negative_img, sep="\\")
    if (img_split[2] != negative_img_split[2]):
        anc_neg.append(img)
        neg.append(negative_img)
    
    count += 1
    print(count)

anc_neg = tf.data.Dataset.from_tensor_slices(anc_neg)
neg = tf.data.Dataset.from_tensor_slices(neg)

positives = tf.data.Dataset.zip((anchor, positive, tf.data.Dataset.from_tensor_slices(tf.ones(len(anchor)))))
negatives = tf.data.Dataset.zip((anc_neg, neg, tf.data.Dataset.from_tensor_slices(tf.zeros(len(anc_neg)))))
data = positives.concatenate(negatives)

tf.data.experimental.save(data, path="label_2")

npData = []
data_numpy = tfds.as_numpy(data)
for i, j, k in data_numpy:
    npData.append([tf.compat.as_str_any(i), tf.compat.as_str_any(j), float(k)])

data_numpy = np.array(npData)
np.random.shuffle(data_numpy)

a, b, c = [], [], []

for i in range(len(data_numpy)):
    a.append(data_numpy[i][0])
    b.append(data_numpy[i][1])
    c.append(data_numpy[i][2])

a, b, c = np.array(a), np.array(b), np.array(c)

data1 = tf.data.Dataset.from_tensor_slices(a)
data2 = tf.data.Dataset.from_tensor_slices(b)
c = tf.convert_to_tensor(c,dtype=tf.float32)
data3 = tf.data.Dataset.from_tensor_slices(c)
data = tf.data.Dataset.zip((data1, data2, data3))

tf.data.experimental.save(data, path="label_1")
"""

print("--> LOADING DATA LABEL")
data = tf.data.experimental.load(path="label_1_50", element_spec=(tf.TensorSpec(shape=(), dtype=tf.string, name=None), 
                                                            tf.TensorSpec(shape=(), dtype=tf.string, name=None), 
                                                            tf.TensorSpec(shape=(), dtype=tf.float32, name=None)))

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

print("--> SET UP TRAINING DATA INTO BATCHES")
# Build dataloader pipeline
data = data.map(preprocess_twin)
#data = data.shuffle(buffer_size=10000, reshuffle_each_iteration = True)

# Training partition (take full dataset as training data)
train_data = data.take(round(len(data)))
train_data = train_data.batch(32)
train_data = train_data.prefetch(8)

# Testing partition
"""test_data = data.skip(round(len(data)*.7))
test_data = test_data.take(round(len(data)*.3))
test_data.take(300)
test_data = test_data.batch(16)
test_data = test_data.prefetch(8)"""

def make_embedding(): 
    inp = Input(shape=(105,105,3), name='input_image')
    
    # First block
    c1 = Conv2D(64, (10,10), activation='relu')(inp)
    m1 = MaxPooling2D(64, (2,2), padding='same')(c1)
    
    # Second block
    c2 = Conv2D(128, (7,7), activation='relu')(m1)
    m2 = MaxPooling2D(64, (2,2), padding='same')(c2)
    
    # Third block 
    c3 = Conv2D(128, (4,4), activation='relu')(m2)
    m3 = MaxPooling2D(64, (2,2), padding='same')(c3)
    
    # Final embedding block
    c4 = Conv2D(256, (4,4), activation='relu')(m3)
    f1 = Flatten()(c4)
    d1 = Dense(4096, activation='relu')(f1)
    
    
    return Model(inputs=[inp], outputs=[d1], name='embedding')

# Siamese L1 Distance class (Manhattan)
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

def L2DistFunc(vects):
    input_embedding, validation_embedding = vects
    return tf.math.sqrt(tf.math.square(input_embedding - validation_embedding))

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
        denominator = tf.math.round(tf.math.multiply(tf.math.sqrt(sum_input), tf.math.sqrt(sum_validation)))
        return numerator/denominator

    

# Siamese model using L2Dist
def make_siamese_model(): 
    
    # Anchor image input in the network
    input_image = Input(name='input_img', shape=(105,105,3))
    
    # Validation image in the network 
    validation_image = Input(name='validation_img', shape=(105,105,3))
    
    # Combine siamese distance components
    #siamese_layer = ChebDist()
    #siamese_layer._name = 'distance'
    #distances = siamese_layer(embedding(input_image), embedding(validation_image))
    
    distances = tf.keras.layers.Lambda(L2DistFunc)([embedding(input_image), embedding(validation_image)])
    #distances.name = "distance"
    # Classification layer 
    classifier = Dense(1, activation='sigmoid')(distances)
    
    return Model(inputs=[input_image, validation_image], outputs=classifier, name='SiameseNetwork')

print("--> CREATING MODEL NETWORK")
embedding = make_embedding()
siamese_model = make_siamese_model()
binary_cross_loss = tf.losses.BinaryCrossentropy()
opt = tf.keras.optimizers.Adam(1e-4) # 0.0001

print("--> SETTING UP CHECKPOINTS")
checkpoint_dir = 'training_checkpoints'
checkpoint_prefix = os.path.join(checkpoint_dir, 'ckpt')
checkpoint = tf.train.Checkpoint(opt=opt, siamese_model=siamese_model)

@tf.function
def train_step(batch):
    
    # Record all of our operations 
    with tf.GradientTape() as tape:     
        # Get anchor and positive/negative image
        X = batch[:2]
        # Get label
        y = batch[2]
        
        # Forward pass
        yhat = siamese_model(X, training=True)
        # Calculate loss
        loss = binary_cross_loss(y, yhat)
    print(loss)
        
    # Calculate gradients
    grad = tape.gradient(loss, siamese_model.trainable_variables)
    
    # Calculate updated weights and apply to siamese model
    opt.apply_gradients(zip(grad, siamese_model.trainable_variables))
        
    # Return loss
    return loss

def train(data, EPOCHS):
    # Loop through epochs
    for epoch in range(1, EPOCHS+1):
        print('\n Epoch {}/{}'.format(epoch, EPOCHS))
        progbar = tf.keras.utils.Progbar(len(data))
        
        # Creating a metric object 
        r = Recall()
        p = Precision()
        
        # Loop through each batch
        for idx, batch in enumerate(data):
            # Run train step here
            loss = train_step(batch)
            yhat = siamese_model.predict(batch[:2])
            r.update_state(batch[2], yhat)
            p.update_state(batch[2], yhat) 
            progbar.update(idx+1)
        print("Loss:", loss.numpy(), "Recall:", r.result().numpy(), "Precision:", p.result().numpy())
        
        # Save checkpoints
        if epoch % 1 == 0: 
            checkpoint.save(file_prefix=checkpoint_prefix)

EPOCHS = 10

print("--> TRAINING STARTS")
train(train_data, EPOCHS)

print("--> SAVING MODEL")
# Save weights
siamese_model.save("siamesemodel_10eps_50%_L2DistFunc.h5")
siamese_model.save_weights("siamesemodel_10eps_50%_L2DistFunc")
