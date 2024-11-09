import tensorflow as tf
import traceback
import cProfile
import pstats
import keras
import time
import os
# snakeviz ./STATS.prof

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ['CUDA_VISIBLE_DEVICES'] = '0'


def main():

    input_shape = (48, 48)

    print('\n\n\n\nStart')

    val_ds = keras.utils.image_dataset_from_directory(
        directory='DataSet/validation',
        labels="inferred",
        label_mode="int",
        class_names=['happy', 'not_happy'],
        color_mode="rgb",
        batch_size=64,
        image_size=input_shape,
        interpolation="bilinear",
        follow_links=False,
        crop_to_aspect_ratio=False,
        data_format=None,
        shuffle=True
    )

    print('Loaded ValData')

    model = keras.saving.load_model('NewModel_25.keras')
    start = time.perf_counter()
    with tf.device('/GPU:0'):
        model.evaluate(val_ds)
    end = time.perf_counter()
    print(f'NewModel_25.keras: {round((end-start), 5)}')

    model = keras.saving.load_model('NewModel_26.keras')
    start = time.perf_counter()
    model.evaluate(val_ds)
    end = time.perf_counter()
    print(f'NewModel_26.keras: {round((end-start), 5)}')

    model = keras.saving.load_model('NewModel_27.keras')
    start = time.perf_counter()
    model.evaluate(val_ds)
    end = time.perf_counter()
    print(f'NewModel_27.keras: {round((end-start), 5)}')

    model = keras.saving.load_model('NewModel_28.keras')
    start = time.perf_counter()
    model.evaluate(val_ds)
    end = time.perf_counter()
    print(f'NewModel_28.keras: {round((end-start), 5)}')

    model = keras.saving.load_model('NewModel_29.keras')
    start = time.perf_counter()
    model.evaluate(val_ds)
    end = time.perf_counter()
    print(f'NewModel_29.keras: {round((end-start), 5)}')

    model = keras.saving.load_model('NewModel_30.keras')
    start = time.perf_counter()
    model.evaluate(val_ds)
    end = time.perf_counter()
    print(f'NewModel_30.keras: {round((end-start), 5)}')

    model = keras.saving.load_model('NewModel_31.keras')
    start = time.perf_counter()
    model.evaluate(val_ds)
    end = time.perf_counter()
    print(f'NewModel_31.keras: {round((end-start), 5)}')

    model = keras.saving.load_model('NewModel_32.keras')
    start = time.perf_counter()
    model.evaluate(val_ds)
    end = time.perf_counter()
    print(f'NewModel_32.keras: {round((end-start), 5)}')

    model = keras.saving.load_model('NewModel_33.keras')
    start = time.perf_counter()
    model.evaluate(val_ds)
    end = time.perf_counter()
    print(f'NewModel_33.keras: {round((end-start), 5)}')


if __name__ == '__main__':

    with cProfile.Profile() as pr:
        try:
            main()
        except (KeyboardInterrupt, OverflowError):
            traceback.print_exc()

    stats = pstats.Stats(pr)
    stats.sort_stats(pstats.SortKey.TIME)
    stats.dump_stats(filename='STATS.prof')
