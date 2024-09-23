import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
from keras import layers, models, optimizers, losses, metrics, callbacks
from IPython.display import clear_output
import matplotlib.pyplot as plt
import tensorflow as tf

# Enable mixed precision training
tf.keras.mixed_precision.set_global_policy('mixed_float16')

# GPU memory growth
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(e)

# Enhanced data augmentation
datagen = tf.keras.preprocessing.image.ImageDataGenerator(
    rescale=1./255,
    validation_split=0.2,
    rotation_range=30,
    width_shift_range=0.2,
    height_shift_range=0.2,
    horizontal_flip=True,
    vertical_flip=True,
    zoom_range=0.2,
    shear_range=0.2,
    brightness_range=[0.8, 1.2]
)

img_height, img_width = 512, 512
batch_size = 32

def create_dataset(data_dir, subset):
    dataset = tf.keras.utils.image_dataset_from_directory(
        data_dir,
        validation_split=0.2,
        subset=subset,
        seed=123,
        image_size=(img_height, img_width),
        batch_size=batch_size
    )
    
    # Use data augmentation
    data_augmentation = tf.keras.Sequential([
        tf.keras.layers.RandomFlip("horizontal"),
        tf.keras.layers.RandomRotation(0.2),
    ])
    
    dataset = dataset.map(
        lambda x, y: (data_augmentation(x, training=True), y),
        num_parallel_calls=tf.data.AUTOTUNE
    )
    
    return dataset.prefetch(buffer_size=tf.data.AUTOTUNE)

# Create a more complex model
def create_model(input_shape):
    base_model = tf.keras.applications.EfficientNetB3(
        include_top=False,
        weights='imagenet',
        input_shape=input_shape
    )
    
    # Fine-tune from this layer onwards
    fine_tune_at = len(base_model.layers) - 30
    for layer in base_model.layers[:fine_tune_at]:
        layer.trainable = False
    
    model = models.Sequential([
        base_model,
        layers.GlobalAveragePooling2D(),
        layers.BatchNormalization(),
        layers.Dense(512, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(256, activation='relu'),
        layers.Dropout(0.3),
        layers.Dense(1, activation='sigmoid')
    ])
    return model

model = create_model((img_height, img_width, 3))

# Compile with learning rate schedule
initial_learning_rate = 1e-4
lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
    initial_learning_rate,
    decay_steps=1000,
    decay_rate=0.9,
    staircase=True
)

model.compile(
    optimizer=optimizers.Adam(learning_rate=lr_schedule),
    loss=losses.BinaryCrossentropy(),
    metrics=[
        metrics.BinaryAccuracy(),
        metrics.Precision(),
        metrics.Recall(),
        metrics.AUC()
    ]
)

class PlotLearning(callbacks.Callback):
    def on_train_begin(self, logs={}):
        self.metrics = {}
        for metric in logs:
            self.metrics[metric] = []
        self.fig = plt.figure(figsize=(12, 4))
        
    def on_epoch_end(self, epoch, logs={}):
        # Update metrics
        for metric in logs:
            if metric in self.metrics:
                self.metrics[metric].append(logs.get(metric))
            else:
                self.metrics[metric] = [logs.get(metric)]
        
        # Clear the previous plot
        clear_output(wait=True)
        
        # Plot training and validation metrics
        plt.clf()
        plt.subplot(1, 2, 1)
        plt.plot(self.metrics['binary_accuracy'], label='Train Accuracy')
        plt.plot(self.metrics['val_binary_accuracy'], label='Validation Accuracy')
        plt.title('Model Accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.legend()
        
        plt.subplot(1, 2, 2)
        plt.plot(self.metrics['loss'], label='Train Loss')
        plt.plot(self.metrics['val_loss'], label='Validation Loss')
        plt.title('Model Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        
        plt.tight_layout()
        plt.show()

# Add the PlotLearning callback to your list of callbacks
checkpoint = callbacks.ModelCheckpoint(
    'src/model/blade_ball_best.keras',
    save_best_only=True,
    monitor='val_binary_accuracy',
    mode='max'
)
early_stopping = callbacks.EarlyStopping(
    monitor='val_loss',
    patience=10,
    restore_best_weights=True
)
reduce_lr = callbacks.ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.2,
    patience=5,
    min_lr=1e-6
)
plot_callback = PlotLearning()
callbacks_list = [checkpoint, early_stopping, reduce_lr, plot_callback]

train_dataset = create_dataset('data_rmbg', "training")
validation_dataset = create_dataset('data_rmbg', "validation")

# Then use these datasets in your fit method
history = model.fit(
    train_dataset,
    epochs=20,
    validation_data=validation_dataset,
    callbacks=callbacks_list
)

# Save the final model
model.save('src/model/blade_ball_final.keras')

print("Model training is finished!")

# Evaluate on test set
test_generator = datagen.flow_from_directory(
    'data_rmbg',
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='binary',
    subset='validation',
    shuffle=False
)

test_loss, test_acc, test_precision, test_recall, test_auc = model.evaluate(test_generator)
print(f"Test Accuracy: {test_acc:.4f}")
print(f"Test Precision: {test_precision:.4f}")
print(f"Test Recall: {test_recall:.4f}")
print(f"Test AUC: {test_auc:.4f}")

print("Evaluation is complete!")