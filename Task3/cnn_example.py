import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from tensorflow import keras
from keras import layers
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import EarlyStopping, ReduceLROnPlateau
from keras.utils import to_categorical

X = np.load('Xtrain_Classification1.npy')
X_t = np.load('Xtest_Classification1.npy')
y = np.load('ytrain_Classification1.npy')
X = np.reshape(X, (6254, 28, 28, 3))
X_t = np.reshape(X_t, (1764, 28, 28, 3))

X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.2, random_state=42)
X_train = X_train.astype('float32') / 255.0
X_temp = X_temp.astype('float32') / 255.0


y_train = to_categorical(y_train, num_classes=2)
y_temp = to_categorical(y_temp, num_classes=2)

METRICS = [
    keras.metrics.TruePositives(name='tp'),
    keras.metrics.FalsePositives(name='fp'),
    keras.metrics.TrueNegatives(name='tn'),
    keras.metrics.FalseNegatives(name='fn'),
    keras.metrics.Precision(name='precision'),
    keras.metrics.Recall(name='recall'),
    keras.metrics.AUC(name='auc'),
    keras.metrics.AUC(name='prc', curve='PR'),
]

class TrainBalancedAccuracyCallback(keras.callbacks.Callback):
    def __init__(self, **kargs):
        super(TrainBalancedAccuracyCallback, self).__init__(**kargs)

    def on_epoch_end(self,epoch, logs={}):
        train_sensitivity = logs['tp'] / (logs['tp'] + logs['fn'] + keras.backend.epsilon())
        train_specificity = logs['tn'] / (logs['tn'] + logs['fp'] + keras.backend.epsilon())
        logs['train_sensitivity'] = train_sensitivity
        logs['train_specificity'] = train_specificity
        logs['train_balacc'] = (train_sensitivity + train_specificity) / 2
        print('train_balacc', logs['train_balacc'])

class ValBalancedAccuracyCallback(keras.callbacks.Callback):
    def __init__(self, **kargs):
        super(ValBalancedAccuracyCallback, self).__init__(**kargs)

    def on_epoch_end(self,epoch, logs={}):
        val_sensitivity = logs['val_tp'] / (logs['val_tp'] + logs['val_fn'] + keras.backend.epsilon())
        val_specificity = logs['val_tn'] / (logs['val_tn'] + logs['val_fp'] + keras.backend.epsilon())
        logs['val_sensitivity'] = val_sensitivity
        logs['val_specificity'] = val_specificity
        logs['val_balacc'] = (val_sensitivity + val_specificity) / 2
        print('val_balacc', logs['val_balacc'])


model = keras.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 3)),
    layers.BatchNormalization(),
    layers.MaxPooling2D(pool_size=(2, 2)),
    
    layers.Conv2D(32, (3, 3), activation='relu'),
    layers.BatchNormalization(),
    layers.MaxPooling2D(pool_size=(2, 2)),
    
    layers.Conv2D(32, (3, 3), activation='relu'),
    layers.BatchNormalization(),
    layers.MaxPooling2D(pool_size=(2, 2)),
    
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.3),
    
    layers.Dense(2, activation='softmax')
])

optimizer = keras.optimizers.Adam(learning_rate=0.001)
model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=METRICS)

datagen = ImageDataGenerator(
    rotation_range=40,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

lr_scheduler = ReduceLROnPlateau(factor=0.5, patience=5, min_lr=1e-7, verbose=1)
early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

train_generator = datagen.flow(X_train, y_train, batch_size=20)

model.fit(train_generator, epochs=50, validation_data=(X_temp, y_temp), 
          callbacks=[early_stopping, lr_scheduler, TrainBalancedAccuracyCallback(), ValBalancedAccuracyCallback()])


model.predict(X_temp)