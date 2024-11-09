from tensorflow import data as tf_data
from model import make_model
import traceback
import cProfile
import pstats
import keras


def main():

    input_shape = (48, 48)

    train_ds = keras.utils.image_dataset_from_directory(
        directory='DataSet/train',
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
    )

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

    train_ds = train_ds.prefetch(tf_data.AUTOTUNE)
    val_ds = val_ds.prefetch(tf_data.AUTOTUNE)

    model = make_model(input_shape=input_shape + (3,), num_classes=2)
    epochs = 50

    model = keras.saving.load_model('NewModel_28.keras')

    callbacks = [
        keras.callbacks.ModelCheckpoint("NewModel_{epoch}.keras"),
    ]
    model.compile(
        optimizer=keras.optimizers.Adam(3e-4),
        loss=keras.losses.BinaryCrossentropy(from_logits=True),
        metrics=[keras.metrics.BinaryAccuracy(name="acc")]
    )

    model.summary()

    model.fit(
        train_ds,
        epochs=epochs,
        callbacks=callbacks,
        validation_data=val_ds,
    )


if __name__ == '__main__':

    with cProfile.Profile() as pr:
        try:
            main()
        except (KeyboardInterrupt, OverflowError):
            traceback.print_exc()

    stats = pstats.Stats(pr)
    stats.sort_stats(pstats.SortKey.TIME)
    stats.dump_stats(filename='STATS.prof')
