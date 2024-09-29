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
from imblearn.over_sampling import RandomOverSampler
from sklearn.utils import shuffle
from keras.utils import custom_object_scope

np.random.seed = 515005

x = np.load('Xtrain_Classification2.npy')
X_t = np.load('Xtest_Classification2.npy')
y = np.load('ytrain_Classification2.npy')

x = x.reshape(x.shape[0], 28, 28, 3)
X_t = X_t.reshape(X_t.shape[0], 28, 28, 3)

x = x.astype('float32')/ 255.0
y = to_categorical(y,num_classes=6)
print(y)
y_flattened = y.argmax(axis=1) 
class_counts = np.bincount(y_flattened)
for class_label, count in enumerate(class_counts):
    print(f"Class {class_label}: {count} samples")
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



model = keras.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 3)),
    layers.BatchNormalization(),
    layers.MaxPooling2D(pool_size=(2, 2)),
    layers.Dropout(0.05),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.BatchNormalization(),
    layers.MaxPooling2D(pool_size=(2, 2)),
    layers.Dropout(0.05),
    layers.Conv2D(128, (3, 3), activation='relu'),
    layers.BatchNormalization(),
    layers.MaxPooling2D(pool_size=(2, 2)),
    layers.Dropout(0.05),
    layers.Flatten(),
    layers.Dense(512, activation='relu'),
    layers.Dropout(0.15),
    layers.Dense(6, activation='softmax')
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


@tf.keras.saving.register_keras_serializable(name="WeightedCategoricalCrossentropy")
class WeightedCategoricalCrossentropy:
    def __init__(self, weights, label_smoothing=0.0, axis=-1, name="weighted_categorical_crossentropy"):
        super().__init__()
        self.weights = weights
        self.label_smoothing = label_smoothing
        self.name = name

    def __call__(self, y_true, y_pred, axis=-1):
        y_pred = tf.convert_to_tensor(y_pred)
        y_true = tf.cast(y_true, y_pred.dtype)
        self.label_smoothing = tf.convert_to_tensor(self.label_smoothing, dtype=y_pred.dtype)

        

        def _smooth_labels():
            num_classes = tf.cast(tf.shape(y_true)[-1], y_pred.dtype)
            return y_true * (1.0 - self.label_smoothing) + (self.label_smoothing / num_classes)

        y_true = tf.__internal__.smart_cond.smart_cond(self.label_smoothing, _smooth_labels, lambda: y_true)

        return tf.reduce_mean(tf.keras.losses.categorical_crossentropy(y_true, y_pred))

    def get_config(self):
        return {"name": self.name, "weights": self.weights}

    @classmethod
    def from_config(cls, config):
        return cls(**config)

        
lr_scheduler = ReduceLROnPlateau(factor=0.8, patience=5, min_lr=1e-7, verbose=1)
tf.keras.saving.get_custom_objects().clear()
with custom_object_scope({"WeightedCategoricalCrossentropy": WeightedCategoricalCrossentropy}):
    total_samples = 5362 + 890 + 116 + 2305 + 990 + 966
    weights = [total_samples / (class_samples * 6) for class_samples in [5362, 890, 116, 2305, 990, 966]]
    print(weights)
    wcce = WeightedCategoricalCrossentropy(weights)
    
     
optimizer = keras.optimizers.legacy.Adam(learning_rate=0.001)
model.compile(optimizer=optimizer, loss=wcce, metrics=METRICS)


datagen = ImageDataGenerator(
    rotation_range=40,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)
minority_classes = [1,2, 4, 5]  
augmented_x = []
augmented_y = []

for class_label in minority_classes:
    class_indices = np.where(y_train.argmax(axis=1) == class_label)[0]
    for index in class_indices:
        augmented_images = []
        original_image = x_train[index]
        augmented_images.append(original_image) 
        
        for _ in range(30):
            augmented_image = datagen.random_transform(original_image)
            augmented_images.append(augmented_image)
        augmented_x.extend(augmented_images)
        augmented_y.extend([y_train[index]] * len(augmented_images))

augmented_x = np.array(augmented_x)
augmented_y = np.array(augmented_y)


x_combined = np.vstack((x_train, augmented_x))
y_combined = np.vstack((y_train, augmented_y))


x_combined = x_combined.reshape(x_combined.shape[0], -1)


oversampler = RandomOverSampler(sampling_strategy="auto", random_state=42)
x_oversampled, y_oversampled = oversampler.fit_resample(x_combined, y_combined)


x_oversampled = x_oversampled.reshape(x_oversampled.shape[0], 28, 28, 3)


x_oversampled, y_oversampled = shuffle(x_oversampled, y_oversampled, random_state=42)

y_flattened = y_oversampled.argmax(axis=1) 
class_counts = np.bincount(y_flattened)
for class_label, count in enumerate(class_counts):
    print(f"Class {class_label}: {count} samples")
train_generator = datagen.flow(x_oversampled, y_oversampled, batch_size=32)
val_generator = datagen.flow(x_temp,y_temp, batch_size=32)

train_balanced_accuracy_callback = TrainBalancedAccuracyCallback()
val_balanced_accuracy_callback = ValBalancedAccuracyCallback()

model.fit(train_generator, epochs=100, validation_data=val_generator, batch_size= 32,
          callbacks=[lr_scheduler, train_balanced_accuracy_callback, val_balanced_accuracy_callback], verbose= 0 )


X_t = X_t.astype('float32') / 255.0
y_t = model.predict(X_t)
print(y_t)
y_x = y_t.argmax(axis=1)


y_x = y_x.reshape(-1, 1)

print(y_x.shape)
print(y_x)

np.save("ytest_Classification2.npy", y_x)