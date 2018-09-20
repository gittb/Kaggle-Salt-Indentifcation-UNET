import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
import sys
import os
from data import *

data_gen_args = dict(rotation_range=.2,
                    horizontal_flip=True,
                    fill_mode='nearest')

gen = trainGenerator(1,'data/salt/train','images','masks',data_gen_args,save_to_dir = None)

path_tfrecords_train = os.path.join("train.tfrecords")
"""
def wrap_bytes(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def add_to_record(image, mask, record_path):





def convert(gen, record_path, record_path):
    img, mask = gen
    print("Converting: " + record_path)
    
    # Number of images. Used when printing the progress.
    
    # Open a TFRecordWriter for the output-file.
    with tf.python_io.TFRecordWriter(record_path) as writer:
        
        # Iterate over all the image-paths and class-labels.
        for i, (path, label) in enumerate(zip(image_paths, labels)):
            # Print the percentage-progress.
            print_progress(count=i, total=num_images-1)

            # Load the image-file using matplotlib's imread function.
            img = imread(path)
            
            # Convert the image to raw bytes.
            img_bytes = img.tostring()

            # Create a dict with the data we want to save in the
            # TFRecords file. You can add more relevant data here.
            data = \
                {
                    'image': wrap_bytes(img_bytes),
                    'label': wrap_int64(label)
                }

            # Wrap the data as TensorFlow Features.
            feature = tf.train.Features(feature=data)

            # Wrap again as a TensorFlow Example.
            example = tf.train.Example(features=feature)

            # Serialize the data.
            serialized = example.SerializeToString()
            
            # Write the serialized data to the TFRecords file.
            writer.write(serialized)
"""
c = 0
while True:
    _ = gen
    print(c)
    c += 1
