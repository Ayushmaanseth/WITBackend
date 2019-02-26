import urllib.request, urllib.parse, urllib.error
import os,subprocess
import pandas as pd
import numpy as np
import os
import tensorflow as tf
from tensorflow import data
import math
def upload_for_tensorboard(file_path,model_path=None):
    os.system('tensorboard --logdir . &')
    #os.system('docker run -p 8500:8500 --mount type=bind,source=%s,target=/models/my_model/ -e MODEL_NAME=my_model -t tensorflow/serving &' % model_path)
    what_if_tool_path = ('http://localhost:6006/#whatif&inferenceAddress1=%s&modelName1=my_model&examplesPath=%s' %(urllib.parse.quote('localhost:8500'), urllib.parse.quote(file_path)))
    return what_if_tool_path

def write_df_as_tfrecord(df, filename, columns=None):
    writer = tf.python_io.TFRecordWriter(filename)
    if columns == None:
        columns = df.columns.values.tolist()
    for index, row in df.iterrows():
        example = tf.train.Example()
        for col in columns:
            if df[col].dtype is np.dtype(np.int64):
                example.features.feature[col].int64_list.value.append(int(row[col]))
            elif df[col].dtype is np.dtype(np.float64):
                example.features.feature[col].float_list.value.append(row[col])
            elif row[col] == row[col]:
                example.features.feature[col].bytes_list.value.append(row[col].encode('utf-8'))
        writer.write(example.SerializeToString())
    writer.close()

def create_feature_spec(df, columns):
    feature_spec = {}
    for f in columns:
        if df[f].dtype is np.dtype(np.int64):
            feature_spec[f] = tf.FixedLenFeature(shape=(), dtype=tf.int64)
        elif df[f].dtype is np.dtype(np.float64):
            feature_spec[f] = tf.FixedLenFeature(shape=(), dtype=tf.float32)
        else:
            feature_spec[f] = tf.FixedLenFeature(shape=(), dtype=tf.string)
    return feature_spec

def parse_tf_example(example_proto, label, feature_spec):
    parsed_features = tf.parse_example(serialized=example_proto, features=feature_spec)
    target = parsed_features.pop(label)
    return parsed_features, target

# An input function for providing input to a model from tf.Examples from tf record files.
def tfrecords_input_fn(files_name_pattern, feature_spec, label, mode=tf.estimator.ModeKeys.EVAL,
                       num_epochs=None,
                       batch_size=64):
    shuffle = True if mode == tf.estimator.ModeKeys.TRAIN else False
    file_names = tf.matching_files(files_name_pattern)
    dataset = data.TFRecordDataset(filenames=file_names)

    if shuffle:
        dataset = dataset.shuffle(buffer_size=2 * batch_size + 1)

    dataset = dataset.batch(batch_size)
    dataset = dataset.map(lambda tf_example: parse_tf_example(tf_example, label, feature_spec))
    dataset = dataset.repeat(num_epochs)
    return dataset

# Creates simple numeric and categorical feature columns from a feature spec and a
# list of columns from that spec to use.
#
# NOTE: Models might perform better with some feature engineering such as bucketed
# numeric columns and hash-bucket/embedding columns for categorical features.
def create_feature_columns(columns, feature_spec):
    ret = []
    for col in columns:
        if feature_spec[col].dtype is tf.int64 or feature_spec[col].dtype is tf.float32:
            ret.append(tf.feature_column.numeric_column(col))
        else:
            ret.append(tf.feature_column.indicator_column(
                tf.feature_column.categorical_column_with_vocabulary_list(col, list(df[col].unique()))))
    return ret



def run_model(file_path,target,labels):
    model_name = 'trained_model'

    # USER: Set the name you want to give the tfrecord dataset file
    tfrecord_name = 'data.tfrecord'

    csv_path = file_path
    df = pd.read_csv(csv_path)
    csv_columns = df.columns.values.tolist()
    label_column = target
    target_zero_value = zero_value
    #df = pd.read_csv(csv_path, skipinitialspace=True)
    #df = df.reindex(np.random.permutation(df.index))
    #print(df.columns.tolist())
    formats = [int,float,double,np.int64,np.float64,np.float32,np.int32]
    if type(target_zero_value) not in formats:
        df[label_column] = np.where(df[label_column] == target_zero_value, 0, 1)

    



    #if label_column in csv_columns:
        #csv_columns.remove(label_column)

    #features_and_labels = csv_columns + [label_column]
    tfrecord_name = 'data.tfrecord'
    tfrecord_path = '/home/datasets/data.tfrecord'
    write_df_as_tfrecord(df,tfrecord_path)
    return tfrecord_path
