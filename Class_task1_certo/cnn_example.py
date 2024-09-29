import numpy as np
import matplotlib.pyplot as plt
import sklearn
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow import keras
from keras import layers, models
from keras.layers import RandomBrightness, RandomZoom, RandomTranslation, RandomContrast
from keras.callbacks import EarlyStopping, ReduceLROnPlateau
from keras.utils import to_categorical
from keras.preprocessing.image import ImageDataGenerator
import keras.backend as K

from sklearn.utils import shuffle

np.random.seed = 515005

x = np.load('Xtrain_Classification1.npy')
X_t = np.load('Xtest_Classification1.npy')
y = np.load('ytrain_Classification1.npy')
x = np.reshape(x, (6254, 28, 28, 3))
X_t = np.reshape(X_t, (1764, 28, 28, 3))
x = x.astype('float32')/ 255.0
y = to_categorical(y,num_classes=2)
class_0_count = np.sum(y[:, 0])  # Class 0
class_1_count = np.sum(y[:, 1])  # Class 1

print("Number of elements in class 0 in y:", class_0_count)
print("Number of elements in class 1 in y:", class_1_count)
x_train, x_temp, y_train, y_temp = train_test_split(x, y, train_size=0.8, random_state=42)

class FalsePositives(tf.keras.metrics.Metric):
    def __init__(self, name='false_positives', **kwargs):
        super(FalsePositives, self).__init__(name=name, **kwargs)
        self.false_positives = self.add_weight(name='fp', initializer='zeros')

    def update_state(self, y_true, y_pred, sample_weight=None):
        y_true_binary = K.argmax(y_true, axis=-1)
        y_pred_binary = K.argmax(y_pred, axis=-1)
        false_positives = K.sum(K.cast(K.equal(y_true_binary, 0) & K.equal(y_pred_binary, 1), 'float'))
        self.false_positives.assign_add(false_positives)

    def result(self):
        return self.false_positives

class FalseNegatives(tf.keras.metrics.Metric):
    def __init__(self, name='false_negatives', **kwargs):
        super(FalseNegatives, self).__init__(name=name, **kwargs)
        self.false_negatives = self.add_weight(name='fn', initializer='zeros')

    def update_state(self, y_true, y_pred, sample_weight=None):
        y_true_binary = K.argmax(y_true, axis=-1)
        y_pred_binary = K.argmax(y_pred, axis=-1)
        false_negatives = K.sum(K.cast(K.equal(y_true_binary, 1) & K.equal(y_pred_binary, 0), 'float'))
        self.false_negatives.assign_add(false_negatives)

    def result(self):
        return self.false_negatives
class TrueNegatives(tf.keras.metrics.Metric):
    def __init__(self, name='true_negatives', **kwargs):
        super(TrueNegatives, self).__init__(name=name, **kwargs)
        self.true_negatives = self.add_weight(name='tn', initializer='zeros')

    def update_state(self, y_true, y_pred, sample_weight=None):
        y_true_binary = K.argmax(y_true, axis=-1)
        y_pred_binary = K.argmax(y_pred, axis=-1)
        true_negatives = K.sum(K.cast(K.equal(y_true_binary, 0) & K.equal(y_pred_binary, 0), 'float'))
        self.true_negatives.assign_add(true_negatives)

    def result(self):
        return self.true_negatives

class TruePositives(tf.keras.metrics.Metric):
    def __init__(self, name='true_positives', **kwargs):
        super(TruePositives, self).__init__(name=name, **kwargs)
        self.true_positives = self.add_weight(name='tp', initializer='zeros')

    def update_state(self, y_true, y_pred, sample_weight=None):
        y_true_binary = K.argmax(y_true, axis=-1)
        y_pred_binary = K.argmax(y_pred, axis=-1)
        true_positives = K.sum(K.cast(K.equal(y_true_binary, 1) & K.equal(y_pred_binary, 1), 'float'))
        self.true_positives.assign_add(true_positives)

    def result(self):
        return self.true_positives








  



class Weighted_BCE_Loss(keras.losses.Loss):
    def __init__(self, weight_zero = 0.15, weight_one = 0.85):
        super().__init__()
        self.weight_zero = weight_zero
        self.weight_one = weight_one
    def call(self, y_true, y_pred):        
        bin_crossentropy = K.binary_crossentropy(y_true, y_pred)
    
        
        weights = y_true * self.weight_one + (1. - y_true) * self.weight_zero
        weighted_bin_crossentropy = weights * bin_crossentropy 

        return keras.backend.mean(weighted_bin_crossentropy)

def weighted_bincrossentropy(true, pred, weight_zero = 0.25, weight_one = 1):
    """
    Calculates weighted binary cross entropy. The weights are fixed.
        
    This can be useful for unbalanced catagories.
    
    Adjust the weights here depending on what is required.
    
    For example if there are 10x as many positive classes as negative classes,
        if you adjust weight_zero = 1.0, weight_one = 0.1, then false positives 
        will be penalize 10 times as much as false negatives.

    """
  
   
    bin_crossentropy = keras.backend.binary_crossentropy(true, pred)
    
    
    weights = true * weight_one + (1. - true) * weight_zero
    weighted_bin_crossentropy = weights * bin_crossentropy 

    return K.mean(weighted_bin_crossentropy)

model = keras.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 3)),
    layers.BatchNormalization(),
    layers.MaxPooling2D(pool_size=(2, 2)),
    layers.Dropout(0.05),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.BatchNormalization(),
    layers.MaxPooling2D(pool_size=(2, 2)),
    layers.Dropout(0.05),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.BatchNormalization(),
    layers.MaxPooling2D(pool_size=(2, 2)),
    layers.Dropout(0.05),
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.15),
    layers.Dense(2, activation='softmax')
])







 


class TrainBalancedAccuracyCallback(tf.keras.callbacks.Callback):
    def __init__(self):
        super(TrainBalancedAccuracyCallback, self).__init__()
        

    def on_epoch_end(self, epoch, logs={}):
       

        train_sensitivity = logs['tp'] / (logs['tp'] + logs['fn'])
        train_specificity = logs['tn'] / (logs['tn'] + logs['fp'])
        logs['train_sensitivity'] = train_sensitivity
        logs['train_specificity'] = train_specificity
        logs['train_balacc'] = (train_sensitivity + train_specificity) / 2
        print(' train_balacc', logs['train_balacc'])


class ValBalancedAccuracyCallback(tf.keras.callbacks.Callback):
    def __init__(self):
        super(ValBalancedAccuracyCallback, self).__init__()
      

    def on_epoch_end(self, epoch, logs={}):
        
        
        val_sensitivity = logs['val_tp'] / (logs['val_tp'] + logs['val_fn'])
        val_specificity = logs['val_tn'] / (logs['val_tn'] + logs['val_fp'])
        logs['val_sensitivity'] = val_sensitivity
        logs['val_specificity'] = val_specificity
        logs['val_balacc'] = (val_sensitivity + val_specificity) / 2
        print(' val_balacc', logs['val_balacc'])
        
METRICS = [
    TruePositives(name='tp'),
    FalsePositives(name='fp'),
    TrueNegatives(name='tn'),
    FalseNegatives(name='fn'),
    keras.metrics.Precision(name='precision'),
    keras.metrics.Recall(name='recall'),
    keras.metrics.AUC(name='auc'),
    keras.metrics.AUC(name='prc', curve='PR'),
    
    
]
        

        
lr_scheduler = ReduceLROnPlateau(factor=0.8, patience=5, min_lr=1e-7, verbose=1)


optimizer = keras.optimizers.legacy.Adam(learning_rate=0.001)
model.compile(optimizer=optimizer, loss=Weighted_BCE_Loss(weight_zero=0.16, weight_one=0.84), metrics=METRICS)
datagen = ImageDataGenerator(
    rotation_range=40,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)
datagen.fit(x_train)
datagen.fit(x_temp)
train_generator = datagen.flow(x_train, y_train, batch_size=32)
val_generator = datagen.flow(x_temp,y_temp, batch_size=32)

train_balanced_accuracy_callback = TrainBalancedAccuracyCallback()
val_balanced_accuracy_callback = ValBalancedAccuracyCallback()

model.fit(train_generator, epochs=100, validation_data=val_generator, batch_size= 32,
          callbacks=[lr_scheduler, train_balanced_accuracy_callback, val_balanced_accuracy_callback], verbose = 0)
val_balacc_history = []

# Train your model and record validation balanced accuracy for each epoch
for epoch in range(100):
    history = model.fit(x_train, y_train, epochs=1, validation_data=(x_temp, y_temp), verbose=0)
    val_balacc = history.history['val_balacc'][0]
    val_balacc_history.append(val_balacc)

# Create an array of epoch numbers
epochs = range(1, 100 + 1)

# Plot the validation balanced accuracy
plt.plot(epochs, val_balacc_history, 'b', label='Validation Balanced Accuracy')
plt.title('Validation Balanced Accuracy Over Epochs')
plt.xlabel('Epochs')
plt.ylabel('Validation Balanced Accuracy')
plt.legend()
plt.show()

X_t = X_t.astype('float32') / 255.0
y_t = model.predict(X_t)
print(y_t)
y_t = (y_t[:, 1] > 0.5).astype(int)   
y_t = np.reshape(y_t, (-1, 1))

np.save('ytest_Classification1.npy', y_t)
print(y_t)


