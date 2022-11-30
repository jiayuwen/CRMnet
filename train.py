import numpy as np
import datetime
import argparse
import os
import time
from model import *
from utils import *
from tensorflow.keras.optimizers import Adam


def get_args():
    parser = argparse.ArgumentParser(description="Yeast_2022")
    parser.add_argument('-e', '--experiment_media', default='complex_media', type=str,
                        help='experiment media')
    parser.add_argument('-m', '--models', default='trans_unet', type=str,
                        help='model architecture')
    args = parser.parse_args()
    return args

    
args = get_args()
print(args)
model_arch = args.models
experiment_media = args.experiment_media
path = ""

## can increase training speed but has portable issue on TPU "bfloat16" with tensorflow 2.8.0
# from tensorflow.keras import mixed_precision
# Equivalent to the two lines above
# mixed_precision.set_global_policy('mixed_float16')
# for TPU
# mixed_precision.set_global_policy('mixed_bfloat16')

if tf.config.list_physical_devices('GPU'):
    strategy = tf.distribute.MirroredStrategy()
    gpus = tf.config.list_logical_devices('GPU')
    print("All devices: ", gpus)
    n_hardwares = len(gpus)
else:  # Use the TPU Strategy
    ##TPU
    resolver = tf.distribute.cluster_resolver.TPUClusterResolver(tpu='local')
    tf.config.experimental_connect_to_cluster(resolver)

    # This is the TPU initialization code that has to be at the beginning.
    tf.tpu.experimental.initialize_tpu_system(resolver)
    strategy = tf.distribute.TPUStrategy(resolver)
    tpus = tf.config.list_logical_devices('TPU')
    print("All devices: ", tpus)
    n_hardwares = len(tpus)

## tensorboard profiler 
tf.profiler.experimental.server.start(6000)

if experiment_media == "complex_media":
    saved_tf_dataset_path = "preprocessed_data/complex_media/"
    model_path = "saved_model/complex_media_model/"
    result_path = "output/complex_media/"+model_arch
else:
    saved_tf_dataset_path = "preprocessed_data/defined_media/"
    model_path = "saved_model/defined_media_model/"
    result_path = "output/defined_media/"+model_arch

if not os.path.isdir(model_path):
    os.makedirs(model_path, exist_ok=True)
if not os.path.isdir(result_path):
    os.makedirs(result_path, exist_ok=True)

train_dataset = tf.data.experimental.load(os.path.join(saved_tf_dataset_path + "train_dataset"))
val_dataset = tf.data.experimental.load(os.path.join(saved_tf_dataset_path + "val_dataset"))

## Disable AutoShard.
options = tf.data.Options()
options.experimental_distribute.auto_shard_policy = tf.data.experimental.AutoShardPolicy.OFF
train_dataset = train_dataset.with_options(options)
val_dataset = val_dataset.with_options(options)

batch_size=1024*n_hardwares #number of TPU cores or GPUs 
train_dataset = train_dataset.batch(batch_size, drop_remainder=True)
val_dataset = val_dataset.batch(batch_size, drop_remainder=True)

train_dataset.cache()
val_dataset.cache()

train_dataset.prefetch(tf.data.AUTOTUNE)
val_dataset.prefetch(tf.data.AUTOTUNE)

with strategy.scope():
    r_square = tfa.metrics.r_square.RSquare(dtype=tf.float32, y_shape=(1,))
    rmse = tf.keras.metrics.RootMeanSquaredError()
    model = return_model(model_arch)
    model.compile(optimizer=Adam(), steps_per_execution=50, loss = tf.keras.losses.Huber(), metrics=[r_square,rmse])

model.summary()

scheduler = CosineScheduler(max_update=50, base_lr=0.001*n_hardwares, final_lr=0.001*n_hardwares, warmup_steps=10, warmup_begin_lr=0.0001*n_hardwares)
learning_rate = tf.keras.callbacks.LearningRateScheduler(scheduler)
early_stop = tf.keras.callbacks.EarlyStopping(monitor='val_r_square', patience=10, mode='max', restore_best_weights=True)

log_dir = path+"logs/fit/"+model_arch+"_"+ experiment_media+"_"+datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

tik = time.time()
history = model.fit(x=train_dataset, epochs=50, batch_size=batch_size, verbose=2,
                  validation_data=val_dataset, callbacks=[tensorboard_callback, early_stop, learning_rate])
tok = time.time()

model.save(model_path + model_arch)

result_dic = model.evaluate(val_dataset, batch_size=batch_size, return_dict=True)
result_dic["training_time"] = tok-tik
save_result(result_dic, result_path)


