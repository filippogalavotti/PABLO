# Import all of the necessary libraries 

import os, json
import numpy as np
import tensorflow as tf

WINDOW_SIZE = 50
BUFFER_SIZE = 512
BATCH_SIZE = 64
SHIFT = 17

# Define list of directories for train and test datasets

train_directory_paths = [
    r"/home/bs_fgalavotti/project/dataset_v2_blocks/dataset_v2_blocks/health/left/alex_kovalev_standart_elbow_left/preproc_angles/train",
    r"/home/bs_fgalavotti/project/dataset_v2_blocks/dataset_v2_blocks/health/left/anna_makarova_standart_elbow_left/preproc_angles/train",
    r"/home/bs_fgalavotti/project/dataset_v2_blocks/dataset_v2_blocks/health/left/artem_snailbox_standart_elbow_left/preproc_angles/train",
    r"/home/bs_fgalavotti/project/dataset_v2_blocks/dataset_v2_blocks/health/left/matthew_antonov_standart_elbow_left/preproc_angles/train",
    r"/home/bs_fgalavotti/project/dataset_v2_blocks/dataset_v2_blocks/health/left/misha_korobok_standart_elbow_left/preproc_angles/train",
    r"/home/bs_fgalavotti/project/dataset_v2_blocks/dataset_v2_blocks/health/left/nikita_snailbox_standart_elbow_left/preproc_angles/train",
    r"/home/bs_fgalavotti/project/dataset_v2_blocks/dataset_v2_blocks/health/left/petya_chizhov_standart_elbow_left/preproc_angles/train",
    r"/home/bs_fgalavotti/project/dataset_v2_blocks/dataset_v2_blocks/health/left/polina_maksimova_standart_elbow_left/preproc_angles/train",
    r"/home/bs_fgalavotti/project/dataset_v2_blocks/dataset_v2_blocks/health/left/sema_duplin_standart_elbow_left/preproc_angles/train",

    r"/home/bs_fgalavotti/project/dataset_v2_blocks/dataset_v2_blocks/health/right/alex_kovalev_standart_elbow_right/preproc_angles/train",
    r"/home/bs_fgalavotti/project/dataset_v2_blocks/dataset_v2_blocks/health/right/andrew_snailbox_standart_elbow_right/preproc_angles/train",
    r"/home/bs_fgalavotti/project/dataset_v2_blocks/dataset_v2_blocks/health/right/anna_makarova_standart_elbow_right/preproc_angles/train",
    r"/home/bs_fgalavotti/project/dataset_v2_blocks/dataset_v2_blocks/health/right/artem_snailbox_standart_elbow_right/preproc_angles/train",
    r"/home/bs_fgalavotti/project/dataset_v2_blocks/dataset_v2_blocks/health/right/matthew_antonov_standart_elbow_right/preproc_angles/train",
    r"/home/bs_fgalavotti/project/dataset_v2_blocks/dataset_v2_blocks/health/right/matvey_gorbenko_standart_elbow_right/preproc_angles/train",
    r"/home/bs_fgalavotti/project/dataset_v2_blocks/dataset_v2_blocks/health/right/misha_korobok_standart_elbow_right/preproc_angles/train",
    r"/home/bs_fgalavotti/project/dataset_v2_blocks/dataset_v2_blocks/health/right/nikita_snailbox_standart_elbow_right/preproc_angles/train",
    r"/home/bs_fgalavotti/project/dataset_v2_blocks/dataset_v2_blocks/health/right/petya_chizhov_standart_elbow_right/preproc_angles/train",
    r"/home/bs_fgalavotti/project/dataset_v2_blocks/dataset_v2_blocks/health/right/polina_maksimova_standart_elbow_right/preproc_angles/train",
    r"/home/bs_fgalavotti/project/dataset_v2_blocks/dataset_v2_blocks/health/right/sema_duplin_standart_elbow_right/preproc_angles/train",
    
    r"/home/bs_fgalavotti/project/dataset_v2_blocks/dataset_v2_blocks/amputant/left/fedya_tropin_standart_elbow_left/preproc_angles/train",
    r"/home/bs_fgalavotti/project/dataset_v2_blocks/dataset_v2_blocks/amputant/left/valery_first_standart_elbow_left/preproc_angles/train",
    r"/home/bs_fgalavotti/project/dataset_v2_blocks/dataset_v2_blocks/amputant/left/valery_first_standart_elbow_left/preproc_angles/train_strong_activity"
]

test_directory_paths = [
    r"/home/bs_fgalavotti/project/dataset_v2_blocks/dataset_v2_blocks/health/left/alex_kovalev_standart_elbow_left/preproc_angles/test",
    r"/home/bs_fgalavotti/project/dataset_v2_blocks/dataset_v2_blocks/health/left/anna_makarova_standart_elbow_left/preproc_angles/test",
    r"/home/bs_fgalavotti/project/dataset_v2_blocks/dataset_v2_blocks/health/left/matthew_antonov_standart_elbow_left/preproc_angles/test",
    r"/home/bs_fgalavotti/project/dataset_v2_blocks/dataset_v2_blocks/health/left/misha_korobok_standart_elbow_left/preproc_angles/test",
    r"/home/bs_fgalavotti/project/dataset_v2_blocks/dataset_v2_blocks/health/left/petya_chizhov_standart_elbow_left/preproc_angles/test",
    r"/home/bs_fgalavotti/project/dataset_v2_blocks/dataset_v2_blocks/health/left/sema_duplin_standart_elbow_left/preproc_angles/test",

    r"/home/bs_fgalavotti/project/dataset_v2_blocks/dataset_v2_blocks/health/right/alex_kovalev_standart_elbow_right/preproc_angles/test",
    r"/home/bs_fgalavotti/project/dataset_v2_blocks/dataset_v2_blocks/health/right/anna_makarova_standart_elbow_right/preproc_angles/test",
    r"/home/bs_fgalavotti/project/dataset_v2_blocks/dataset_v2_blocks/health/right/misha_korobok_standart_elbow_right/preproc_angles/test",
    r"/home/bs_fgalavotti/project/dataset_v2_blocks/dataset_v2_blocks/health/right/petya_chizhov_standart_elbow_right/preproc_angles/test",
    r"/home/bs_fgalavotti/project/dataset_v2_blocks/dataset_v2_blocks/health/right/sema_duplin_standart_elbow_right/preproc_angles/test",

    r"/home/bs_fgalavotti/project/dataset_v2_blocks/dataset_v2_blocks/amputant/left/fedya_tropin_standart_elbow_left/preproc_angles/test",
]

# Define helper functions

def flatten_window(features_window, labels_window):
    # Batch the features and labels windows
    features_batch = features_window.batch(WINDOW_SIZE, drop_remainder=True)
    labels_batch = labels_window.batch(WINDOW_SIZE, drop_remainder=True)
    
    # Take only the label at the last time step of each window
    labels_batch = labels_batch.map(lambda x: x[-1])  # Get the last element from each window
    return tf.data.Dataset.zip(features_batch, labels_batch)

def augment_features(feature, label):

    shift = tf.random.uniform(shape=[], minval=1, maxval=tf.shape(feature)[0], dtype=tf.int32)
    rolled_feature = tf.roll(feature, shift = shift, axis = 0)

    should_mirror = tf.random.uniform(shape=[], minval = 0, maxval = 1) < 0.5
    augmented_feature = tf.cond(should_mirror, lambda: tf.reverse(rolled_feature, axis = [0]), lambda: rolled_feature)

    return augmented_feature, label

def directory_to_dataset(directory_path, training):
    datasets = []  # To collect datasets from multiple files
    for filename in os.listdir(directory_path):
        if filename.endswith('.npz'):
            # Construct the full file path
            file_path = os.path.join(directory_path, filename)

            print(f"\rProcessing: {file_path}", end="", flush=True)
            
            # Load the .npz file
            data = np.load(file_path)

            data_myo = data["data_myo"]
            data_angles = data["data_angles"]

            if training:

                raw_dataset = tf.data.Dataset.from_tensor_slices((data_myo, data_angles))

                # raw_dataset.map(augment_features)

                windowed_dataset = raw_dataset.window(WINDOW_SIZE, shift=SHIFT, drop_remainder=True)

                cardinality = windowed_dataset.cardinality()

                mapped_dataset = windowed_dataset.flat_map(flatten_window)

                mapped_dataset = mapped_dataset.apply(tf.data.experimental.assert_cardinality(cardinality))
                
                datasets.append(mapped_dataset)


            else:
                raw_dataset = tf.data.Dataset.from_tensor_slices((data_myo, data_angles))

                windowed_dataset = raw_dataset.window(WINDOW_SIZE, shift=SHIFT, drop_remainder=True)

                cardinality = windowed_dataset.cardinality()

                mapped_dataset = windowed_dataset.flat_map(flatten_window)

                mapped_dataset = mapped_dataset.apply(tf.data.experimental.assert_cardinality(cardinality))
                
                datasets.append(mapped_dataset)



    combined_dataset = datasets[0]

    for ds in datasets[1:]:
        combined_dataset = combined_dataset.concatenate(ds)

    return combined_dataset

def directory_paths_to_dataset(directory_paths, training):
    dataset = directory_to_dataset(directory_paths[0], training)
    for path in directory_paths[1:]:
        dataset = dataset.concatenate(directory_to_dataset(path, training))
    return dataset

# Create the train and test dataset

print("\n--- DATASETS CREATION ---")

train_dataset = directory_paths_to_dataset(train_directory_paths, training = True)

print(f"\nCardinality of the train_dataset: {train_dataset.cardinality()}")

test_dataset = directory_paths_to_dataset(test_directory_paths, training = False)

print(f"\nCardinality of the test_dataset: {test_dataset.cardinality()}")

print("\n--- PREPARING THE TRAINING AND TEST SAMPLES ---")

train_dataset = train_dataset.shuffle(BUFFER_SIZE).batch(BATCH_SIZE)
test_dataset = test_dataset.shuffle(BUFFER_SIZE).batch(BATCH_SIZE)

print("\n--- DATASETS DOWNLOAD ---")

print("> Downloading train_dataset")
train_dataset.save("/home/bs_fgalavotti/project/FullDataset/train_dataset")

print("> Downloading test_dataset")
test_dataset.save("/home/bs_fgalavotti/project/FullDataset/test_dataset")