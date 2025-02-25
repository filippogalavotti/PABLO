# Import all of the necessary libraries 

import os, json
import numpy as np
import tensorflow as tf
import keras_tuner

# variables for model structure

"""

PARAMETRI COTRUTTIVI

IMPORTANTE! MODELS_TO_PASS < MAX_VALID_TRIALS << MAX_TRIALS

"""

NUM_SAMPLES_IN = 50
NUM_CHANNELS_IN = 8
NUM_CHANNELS_OUT = 20

MIN_RNN_INPUT_TIMESTEPS = 16

# variables for model training

TRIAL_PERCENTAGE = 0.50    # Percentuale del dataset da utilizzare nell'allenamento di scrematura
TRIAL_EPOCHS = 10          # Numero di epoche su cui allenare ciascun modello nell'allenamento di scrematura

MAX_VALID_TRIALS = 100     # Numero di modelli da allenare nell'allenamento di scrematura (numero di configurazioni di iperparametri valide)
MAX_TRIALS = 1000000       # Numero massimo di configurazioni di iperparametri da provare 

MODELS_TO_PASS = 10        # Numero di modelli (tra quelli valutati nell'allenamento di scrematura) da passare all'allenamento finale
MODELS_PERCENTAGE = 1      # Percentuale del dataset da utilizzare nell'allenamento finale
MODELS_EPOCHS = 20         # Numero di epoche su cui allenare ciascun modello nell'allenamento finale

SIZE_CONSTRAINT = 128   # KB

# Define helper functions

def make_serializable(obj):
    if isinstance(obj, dict):
        return {key: make_serializable(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [make_serializable(item) for item in obj]
    elif isinstance(obj, np.float32):  # or TensorFlow float types
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    return obj

def augment_features(feature, label):

    shift = tf.random.uniform(shape=[], minval=1, maxval=tf.shape(feature)[0], dtype=tf.int32)
    rolled_feature = tf.roll(feature, shift = shift, axis = 0)

    should_mirror = tf.random.uniform(shape=[], minval = 0, maxval = 1) < 0.5
    augmented_feature = tf.cond(should_mirror, lambda: tf.reverse(rolled_feature, axis = [0]), lambda: rolled_feature)

    return augmented_feature, label

def reset_weights(model):
    for layer in model.layers:
        if hasattr(layer, 'kernel') and layer.kernel is not None:
            # Use the shape of the kernel to initialize it
            shape = tf.shape(layer.kernel)
            initializer = tf.keras.initializers.get(layer.kernel_initializer)
            layer.kernel.assign(initializer(shape))
        if hasattr(layer, 'bias') and layer.bias is not None:
            # Use the shape of the bias to initialize it
            shape = tf.shape(layer.bias)
            initializer = tf.keras.initializers.get(layer.bias_initializer)
            layer.bias.assign(initializer(shape))

# defining custom callbacks

class RestoreBestValMSE(tf.keras.callbacks.Callback):
    def __init__(self):
        super(RestoreBestValMSE, self).__init__()
        self.best_weights = None
        self.best_val_mse = float('inf')  # Initialize to infinity

    def on_epoch_end(self, epoch, logs=None):
        # Check if the validation MSE improved in this epoch
        val_mse = logs.get('val_mse')  # Make sure to track 'val_mse' in model metrics
        if val_mse is not None and val_mse < self.best_val_mse:
            # If validation MSE improved, save current weights and update best_val_mse
            self.best_val_mse = val_mse
            self.best_weights = self.model.get_weights()  # Save the weights of the model

    def on_train_end(self, logs=None):
        # At the end of training, set the model's weights to the best recorded weights
        if self.best_weights is not None:
            self.model.set_weights(self.best_weights)
            print("\nRestored model's weights to those from epoch with minimum val_mse.")

# Define the EarlyStopping callback
early_stopping_callback_trials = tf.keras.callbacks.EarlyStopping(
    monitor='val_mse',          # Metric to monitor
    patience=5,                 # Number of epochs to wait for improvement
    verbose=1,                  # Verbosity mode (1 = progress messages)
    mode='min',                 # Use 'min' for loss metrics
    restore_best_weights=False  # Revert to the best weights after stopping
)

# Define ReduceLROnPlateau callback
reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
    monitor='val_mse',     # Metric to monitor
    factor=0.5,            # Factor to reduce the learning rate by
    patience=3,            # Number of epochs with no improvement before reducing
    min_lr=1e-12           # Lower bound on the learning rate
)

print("\n--- DATASETS UPLOAD ---")

print("> Uploading train_dataset")
uploaded_train_dataset = tf.data.Dataset.load("/leonardo/home/userexternal/fgalavot/project/package/FullDataset/train_dataset")

print("> Uploading test_dataset")
uploaded_test_dataset = tf.data.Dataset.load("/leonardo/home/userexternal/fgalavot/project/package/FullDataset/test_dataset")

print("\n--- APPLYING DATA AUGMENTATION ---")

uploaded_train_dataset = uploaded_train_dataset.map(augment_features)

print("\n--- APPLYING PREFETCHING ---")

uploaded_train_dataset = uploaded_train_dataset.prefetch(tf.data.AUTOTUNE)
uploaded_test_dataset = uploaded_test_dataset.prefetch(tf.data.AUTOTUNE)

print("\n--- PREPARING DATA SAMPLES ---")

train_sample_1 = uploaded_train_dataset.take(int(TRIAL_PERCENTAGE * len(uploaded_train_dataset)))
test_sample_1 = uploaded_test_dataset.take(int(TRIAL_PERCENTAGE * len(uploaded_test_dataset)))

train_sample_2 = uploaded_train_dataset.take(int(MODELS_PERCENTAGE * len(uploaded_train_dataset)))
test_sample_2 = uploaded_test_dataset.take(int(MODELS_PERCENTAGE * len(uploaded_test_dataset)))

print(f"\nCardinality of the train_sample_1: {train_sample_1.cardinality()}")
print(f"\nCardinality of the test_sample_1: {test_sample_1.cardinality()}")

print(f"\nCardinality of the train_sample_2: {train_sample_2.cardinality()}")
print(f"\nCardinality of the test_sample_2: {test_sample_2.cardinality()}")


"""

HERE WE DEFINE THE HYPERMODEL 

|     |   |   |   |
|     |   |   |   |
V     V   V   V   V

"""


print("\n--- SETTING UP THE HYPERMODEL ---")

class MyHyperModel(keras_tuner.HyperModel):
    def __init__(self):
        super().__init__()
        self.hyperparameters = keras_tuner.HyperParameters()  # Initialize HyperParameters
    
    def build(self, hp):
        self.hyperparameters = hp

        x_in = tf.keras.layers.Input(shape=(NUM_SAMPLES_IN, NUM_CHANNELS_IN))

        x = x_in

        # max_params = int(SIZE_CONSTRAINT * 256)

        # Conv Blocks
        num_conv_blocks = hp.Int('num_conv_blocks', min_value=1, max_value=3)

        for i in range(num_conv_blocks):

            sequence_length = x.shape[1]  # Get the sequence length dimension

            kernel_size = hp.Int(f'conv_kernel_size_1_{i}', min_value=3, max_value=5)
            dilation_rate = hp.Int(f'conv_dilation_rate_1_{i}', min_value=2, max_value=8)
            min_output_size = MIN_RNN_INPUT_TIMESTEPS if MIN_RNN_INPUT_TIMESTEPS > (kernel_size * dilation_rate) else (kernel_size * dilation_rate)

            if sequence_length < min_output_size:
                print(f"Stopping at conv block {i}.2 as sequence length {sequence_length} is less than {min_output_size}")
                break

            x = tf.keras.layers.Conv1D(
                filters=hp.Int(f'conv_filters_1_{i}', min_value=16, max_value=64, step=8),
                kernel_size=kernel_size,
                dilation_rate=dilation_rate,
                padding='same',
                use_bias=False,
                kernel_regularizer=tf.keras.regularizers.l2(hp.Float(f'l2_conv_1_{i}', min_value=1e-6, max_value=1e-2, sampling='log'))
            )(x)

            x = tf.keras.layers.BatchNormalization()(x)
            x = tf.keras.layers.ReLU()(x)

            # Check output shape after this layer
            sequence_length = x.shape[1]  # Get the sequence length dimension

            kernel_size = hp.Int(f'conv_kernel_size_2_{i}', min_value=3, max_value=5)
            dilation_rate = hp.Int(f'conv_dilation_rate_2_{i}', min_value=2, max_value=8)
            min_output_size = MIN_RNN_INPUT_TIMESTEPS if MIN_RNN_INPUT_TIMESTEPS > (kernel_size * dilation_rate) else (kernel_size * dilation_rate)

            # Stop if sequence length is smaller than the minimum allowed
            if sequence_length < min_output_size:
                print(f"Stopping at conv block {i}.2 as sequence length {sequence_length} is less than {min_output_size}")
                break

            x = tf.keras.layers.Conv1D(
                filters=hp.Int(f'conv_filters_2_{i}', min_value=16, max_value=64, step=8),
                kernel_size=kernel_size,
                dilation_rate=dilation_rate,
                padding='same',
                use_bias=False,
                kernel_regularizer=tf.keras.regularizers.l2(hp.Float(f'l2_conv_2_{i}', min_value=1e-6, max_value=1e-2, sampling='log'))
            )(x)

            x = tf.keras.layers.BatchNormalization()(x)
            x = tf.keras.layers.ReLU()(x)

            # Check output shape after this layer
            sequence_length = x.shape[1]  # Get the sequence length dimension

            kernel_size = hp.Int(f'conv_kernel_size_3_{i}', min_value=3, max_value=5)
            strides = hp.Int(f'conv_strides_3_{i}', min_value=2, max_value=4, step=1)
            pool_size = 2
            min_output_size = (MIN_RNN_INPUT_TIMESTEPS * strides * pool_size) if (MIN_RNN_INPUT_TIMESTEPS * strides * pool_size) > kernel_size else kernel_size

            # Stop if sequence length is smaller than the minimum allowed
            if sequence_length < min_output_size:
                print(f"Stopping at conv block {i}.3 as sequence length {sequence_length} is less than {min_output_size}")
                break

            x = tf.keras.layers.Conv1D(
                filters=hp.Int(f'conv_filters_3_{i}', min_value=16, max_value=64, step=8),
                kernel_size=kernel_size,
                strides=strides,
                padding='same',
                use_bias=False,
                kernel_regularizer=tf.keras.regularizers.l2(hp.Float(f'l2_conv_3_{i}', min_value=1e-6, max_value=1e-2, sampling='log'))
            )(x)

            # Adjust pooling layer based on sequence length
            if x.shape[1] > 1:
                x = tf.keras.layers.AveragePooling1D(pool_size=pool_size)(x)

            x = tf.keras.layers.BatchNormalization()(x)
            x = tf.keras.layers.ReLU()(x)

        # Recurrent Layers

        # Ensure the sequence includes the proper amount of timesteps
        if x.shape[1] >= MIN_RNN_INPUT_TIMESTEPS:
            # Hyperparameters for the GRU layers
            num_recurrent_units = hp.Int('num_recurrent_units', min_value=16, max_value=128, step=16)
            recurrent_dropout = hp.Float('recurrent_dropout', min_value=0.1, max_value=0.5, step=0.1)
            l2_recurrent = hp.Float('l2_recurrent', min_value=1e-6, max_value=1e-2, sampling='log')

            # First Bidirectional GRU layer
            x = tf.keras.layers.Bidirectional(
                tf.keras.layers.GRU(
                    units=num_recurrent_units,
                    return_sequences=True,  # Output full sequence to feed into the next layer
                    dropout=recurrent_dropout,
                    kernel_regularizer=tf.keras.regularizers.l2(l2_recurrent)
                )
            )(x)

            # Second Bidirectional GRU layer
            x = tf.keras.layers.Bidirectional(
                tf.keras.layers.GRU(
                    units=num_recurrent_units,
                    return_sequences=False,  # Process up to the final output
                    dropout=recurrent_dropout,
                    kernel_regularizer=tf.keras.regularizers.l2(l2_recurrent)
                )
            )(x)

        # Discard Invalid Models
        else:
            raise ValueError(f'Invalid Model: Input timesteps of RNN layers is {x.shape[1]} (minimum acceptable size is: {MIN_RNN_INPUT_TIMESTEPS})')

        # Flatten

        x = tf.keras.layers.Flatten()(x)

        # Dense Blocks
        num_dense_blocks = hp.Int('num_dense_blocks', min_value=1, max_value=2)

        for i in range(num_dense_blocks):
        
            x = tf.keras.layers.Dense(
                units = hp.Int(f'dense_units_{i}', min_value=16, max_value=64, step=8),
                use_bias = False,
                kernel_regularizer=tf.keras.regularizers.l2(hp.Float(f'l2_dense_{i}', min_value=1e-6, max_value=1e-2, sampling='log'))
            )(x)
            x = tf.keras.layers.BatchNormalization()(x)
            x = tf.keras.layers.ReLU()(x)
            x = tf.keras.layers.Dropout(
                rate=hp.Float(f'dense_dropout_rate_{i}', min_value=0.1, max_value=0.5, step=0.1)
            )(x)

        # Final Layer
        x_out = tf.keras.layers.Dense(
                units = NUM_CHANNELS_OUT,
                use_bias = False,
                kernel_regularizer=tf.keras.regularizers.l2(hp.Float(f'l2_dense_final', min_value=1e-6, max_value=1e-2, sampling='log'))
        )(x)

        model = tf.keras.Model(inputs=x_in, outputs=x_out)

        model.compile(optimizer=tf.keras.optimizers.Adam(hp.Float('learning_rate', min_value=1e-6, max_value=1e-2, sampling='log')),
                      loss='mse', metrics=['mse'])
        
        print(f"Model Size: {model.count_params() / 256}")
        
        return model
    

"""

HERE WE SETUP THE TUNER

|     |   |   |   |
|     |   |   |   |
V     V   V   V   V

"""

print("\n--- SETTING UP THE TUNER ---")

class MyTuner(keras_tuner.BayesianOptimization):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.valid_trials = 0  # Counter for valid trials
        self.total_trials = 0  # Counter for total trials attempted

    def run_trial(self, trial, *args, **kwargs):
        try:
            # Run the trial
            histories = super().run_trial(trial, *args, **kwargs)
            trial.score = np.min(histories[0].history.get(self.oracle.objective.name, [np.inf]))

            # Check the objective value for the trial
            print(f"Trial {trial.trial_id} - Objective value: {trial.score}")

            # Print a message only when a valid trial is trained
            print(f"Valid trial {self.valid_trials + 1} / {MAX_VALID_TRIALS}...")
            
            # Increment the valid trial count if no error occurs
            self.valid_trials += 1 
            trial.status = keras_tuner.engine.trial.TrialStatus.COMPLETED
        except Exception as e:

            # Check the objective value for the trial
            print(f"Trial {trial.trial_id} failed with exception: {e}")

            # Skip incrementing valid_trials on failure
            trial.status = keras_tuner.engine.trial.TrialStatus.FAILED

    def search(self, *args, **kwargs):
        while self.valid_trials < MAX_VALID_TRIALS and self.total_trials < MAX_TRIALS:
            trial = self.oracle.create_trial(f"trial_{self.total_trials}")  # QUI viene stampato "Search: Running Trial #i ---" 
            
            # Check if the trial status indicates the search should stop
            if trial.status == keras_tuner.engine.trial.TrialStatus.STOPPED:
                print(f"Trial {trial.trial_id} stopped with status: {trial.status}")
                continue

            # Run the trial
            self.run_trial(trial, *args, **kwargs)
            
            # Increment total trials
            self.total_trials += 1
        
        print(f"Search completed with {self.valid_trials} valid trials and {self.total_trials} total trials.")

tuner = MyTuner(
    MyHyperModel(),
    objective='val_mse',                                                                   # Objective to optimize
    max_model_size=int(SIZE_CONSTRAINT * 256),                                             # Size constraint of the model
    max_trials=MAX_TRIALS,
    directory='/leonardo/home/userexternal/fgalavot/project/package/L1Full',               # Directory to store results
    project_name='BayesianOptimization_search',                                            # Project name for tracking
)



"""

HERE WE SETUP THE SEARCH

"""

print("\n--- RUNNING THE SEARCH ---")

tuner.search(
    train_sample_1,
    epochs=TRIAL_EPOCHS,
    validation_data=test_sample_1,
    callbacks=[early_stopping_callback_trials, reduce_lr]
)

print("\n--- RETRIEVING THE BEST MODELS ---")
best_models = tuner.get_best_models(num_models=MODELS_TO_PASS)


print("\n--- PREVENTIVE SAVING OF BEST MODELS ---")
for i in range(MODELS_TO_PASS):
    best_models[i].summary()
    best_models[i].save(f'/leonardo/home/userexternal/fgalavot/project/package/L1Full/best_models_backup/model_{i}.keras')


"""

HERE WE IMPLEMENT SECOND TRAINING

"""


history = []

for i in range(MODELS_TO_PASS):

    # Reset the weights and biases

    print("\n--- RESETTING THE WEIGHTS ---")

    reset_weights(best_models[i])

    # Train the model

    print(f"\n--- TRAINING MODEL {i} ---")

    restore_best_val_mse_callback = RestoreBestValMSE()

    history.append(best_models[i].fit(
        train_sample_2,
        epochs = MODELS_EPOCHS,
        callbacks=[restore_best_val_mse_callback, reduce_lr],
        validation_data = test_sample_2,
    ))

    print(f"\n--- SAVING MODEL {i} ---")

    best_models[i].save(f'/leonardo/home/userexternal/fgalavot/project/package/L1Full/best_models/model_{i}.keras')

    print(f"\n--- SAVING MODEL {i}'S HISTORY ---")

    # Process history[i].history
    serializable_history = make_serializable(history[i].history)

    with open(f'/leonardo/home/userexternal/fgalavot/project/package/L1Full/best_models/model_{i}_history.json', 'w') as f:
        json.dump(serializable_history, f)


print("--- BENCHMARK MODEL ---")

def create_model(
    num_samples_in: int,
    num_channels_in: int,
    num_channels_out: int,
):
    
    model = tf.keras.Sequential()

    model.add(tf.keras.layers.Input(shape=(num_samples_in, num_channels_in)))

    # Block 0
    model.add(tf.keras.layers.Conv1D(16, 3, dilation_rate=2, padding='same', use_bias=False))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.ReLU())
    model.add(tf.keras.layers.Conv1D(16, 3, dilation_rate=2, padding='same', use_bias=False))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.ReLU())
    model.add(tf.keras.layers.Conv1D(16, 5, padding='same', use_bias=False))
    model.add(tf.keras.layers.AveragePooling1D(pool_size=2))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.ReLU())

    # Block 1
    model.add(tf.keras.layers.Conv1D(32, 3, dilation_rate=4, padding='same', use_bias=False))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.ReLU())
    model.add(tf.keras.layers.Conv1D(32, 3, dilation_rate=4, padding='same', use_bias=False))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.ReLU())
    model.add(tf.keras.layers.Conv1D(32, 5, strides=2, padding='same', use_bias=False))
    model.add(tf.keras.layers.AveragePooling1D(pool_size=2))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.ReLU())

    # Block 2
    # model.add(tf.keras.layers.Conv1D(64, 3, dilation_rate=8, padding='same', use_bias=False))
    # model.add(tf.keras.layers.BatchNormalization())
    # model.add(tf.keras.layers.ReLU())
    # model.add(tf.keras.layers.Conv1D(64, 3, dilation_rate=8, padding='same', use_bias=False))
    # model.add(tf.keras.layers.BatchNormalization())
    # model.add(tf.keras.layers.ReLU())
    # model.add(tf.keras.layers.Conv1D(64, 5, strides=4, padding='same', use_bias=False))
    # model.add(tf.keras.layers.AveragePooling1D(pool_size=2))
    # model.add(tf.keras.layers.BatchNormalization())
    # model.add(tf.keras.layers.ReLU())

    # Fully Connected Layers
    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dense(64, use_bias=False))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.ReLU())
    model.add(tf.keras.layers.Dropout(0.5))
    model.add(tf.keras.layers.Dense(32, use_bias=False))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.ReLU())
    model.add(tf.keras.layers.Dropout(0.5))
    model.add(tf.keras.layers.Dense(num_channels_out, use_bias=False))

    return model

marcello = create_model(
    num_samples_in=NUM_SAMPLES_IN,
    num_channels_in=NUM_CHANNELS_IN,
    num_channels_out=NUM_CHANNELS_OUT,
)

marcello.summary()

optimizer = tf.keras.optimizers.Adam()
loss_fn = tf.keras.losses.MeanSquaredError()
marcello.compile(optimizer=optimizer, loss=loss_fn, metrics=[tf.keras.metrics.MeanSquaredError()])

print("--- BENCHMARK MODEL'S TRAINING ---")

history = marcello.fit(
    train_sample_2,
    epochs = MODELS_EPOCHS,
    validation_data = test_sample_2
)

print("--- SAVING BENCHMARK MODEL ---")

marcello.save('/leonardo/home/userexternal/fgalavot/project/package/L1Full/best_models/marcello.keras')

print("--- SAVING BENCHMARK MODEL'S HISTORY ---")

with open('/leonardo/home/userexternal/fgalavot/project/package/L1Full/best_models/history_marcello.json', 'w') as f:
    json.dump(history.history, f)

print("--- PROGRAMMA CONCLUSO ---")