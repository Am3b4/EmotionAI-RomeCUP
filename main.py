from Dataaugmentation import random_noise
from colorama import Fore
from tqdm import tqdm

import traceback
import cProfile
import pstats
import shutil
import keras
import time
import os

# snakeviz ./STATS.prof

emotion_list = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']
emotion_dict = {0: 'happy', 1: 'not_happy'}

data_augmentation_layers = [
    keras.layers.RandomFlip("horizontal"),
    keras.layers.RandomFlip("vertical"),
    keras.layers.RandomRotation(0.1),
    keras.layers.RandomBrightness(0.1),
]

# snakeviz ./STATS.prof


def data_augmentation(images):
    for layer in data_augmentation_layers:
        images = layer(images)
    return images


def my_transform(path: str):

    dir_trans = {'RawDataSet': 'DataSet', 'images1': '', 'images2': '', 'test': 'validation', 'D:': 'D:/'}
    path_list = path.split('/')

    last_part = path_list[-1].split('\\')
    path_list[-1] = last_part[-1]
    path_list.append(last_part[0])

    for i, path_part in enumerate(path_list):
        if path_part in dir_trans.keys():
            path_list[i] = dir_trans[path_part]

    path_list.remove('')

    return '/'.join(path_list)


def my_transform2(path: str):

    dir_trans = {'train': '', 'validation': ''}
    path_list = path.split('/')

    last_part = path_list[-1].split('\\')
    path_list[-1] = last_part[-1]
    path_list.append(last_part[0])

    for i, path_part in enumerate(path_list):
        if path_part in dir_trans.keys():
            path_list[i] = dir_trans[path_part]

    path_list.remove('')

    return '/'.join(path_list)


def create_dataset():
    ORIG_BASE_DIR = r"D:/Python_Projects/EmotionAI/RawDataSet/"
    GOAL_BASE_DIR = r"D:/Python_Projects/EmotionAI/DataSet/"

    emotions = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']
    folders = ['images1', 'images2']

    for emotion in emotions:
        if not os.path.exists(f'{GOAL_BASE_DIR}{emotion}'):
            os.makedirs(f'{GOAL_BASE_DIR}{emotion}')
        if not os.path.exists(f'{GOAL_BASE_DIR}{emotion}/train'):
            os.makedirs(f'{GOAL_BASE_DIR}{emotion}/train')
        if not os.path.exists(f'{GOAL_BASE_DIR}{emotion}/validation'):
            os.makedirs(f'{GOAL_BASE_DIR}{emotion}/validation')

    all_dirs = []

    for folder in folders:
        sub_dirs = [item for item in os.listdir(f'{ORIG_BASE_DIR}{folder}') if (os.path.isdir(os.path.join(f'{ORIG_BASE_DIR}{folder}', item)) and item != 'images')]
        for sub_dir in sub_dirs:
            sub_sub_dirs = [f for f in os.listdir(f'{ORIG_BASE_DIR}{folder}/{sub_dir}') if os.path.isdir(os.path.join(f'{ORIG_BASE_DIR}{folder}/{sub_dir}', f))]
            for sub_sub_dir in sub_sub_dirs:
                all_dirs.append(os.path.join(f'{ORIG_BASE_DIR}{folder}/{sub_dir}', sub_sub_dir))

    for item in tqdm(all_dirs):
        for file in os.listdir(item):
            src_file = os.path.join(item, file)
            destination_file = os.path.join(my_transform(item), file)

            shutil.copy2(src_file, destination_file)


def redo_dataset():

    emotions = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']
    ORIG_DIR = f'DataSet'

    all_dirs = []

    for emotion in emotions:
        NOW_DIR = f'{ORIG_DIR}/{emotion}'
        sub_dirs = [item for item in os.listdir(NOW_DIR) if (os.path.isdir(os.path.join(NOW_DIR, item)))]
        for sub_dir in sub_dirs:
            all_dirs.append(os.path.join(NOW_DIR, sub_dir))

    for item in tqdm(all_dirs):
        for file in os.listdir(item):
            src_file = os.path.join(item, file)
            destination_file = os.path.join(my_transform2(item), file)
            shutil.copy2(src_file, destination_file)


def count_files(emotion: str, filePath: str):

    count = 0

    for root_dir, cur_dir, files in os.walk(fr'{filePath}/{emotion}'):
        count += len(files)
    return count


def de_equalizer():

    dir_dict = {}

    if not os.path.exists(f'DataSet/EQTMP'):
        os.mkdir(f'DataSet/EQTMP')

    for emotion in emotion_list:

        if not os.path.exists(f'DataSet/EQTMP/{emotion}'):
            os.mkdir(f'DataSet/EQTMP/{emotion}')

        if not os.path.exists(f'DataSet/{emotion}'):
            print(f'ERROR {emotion} folder missing')
            raise KeyboardInterrupt

        if os.path.exists(f'DataSet/{emotion}/tmp'):
            dir_dict[f'{emotion}'] = f'Dataset/{emotion}/tmp'

    for key, value in dir_dict.items():
        shutil.move(value, f'DataSet/EQTMP/{key}/')


def de_focuser():

    if not os.path.exists(f'DataSet/TMP'):
        print('Nothing to defocus')
        raise KeyboardInterrupt

    path_dir = f'DataSet/TMP'
    sub_dirs = [name for name in os.listdir(path_dir) if os.path.isdir(os.path.join(path_dir, name))]

    for emotion in sub_dirs:
        shutil.move(os.path.join(path_dir, emotion), f'DataSet/{emotion}')


def tmp_equalizer(filePath: str):

    emotions_count = {}

    my_emotion_list = [f for f in os.listdir(f"{filePath}")]

    for emotion in my_emotion_list:
        emotions_count[emotion] = count_files(emotion, filePath)

    for key, value in emotions_count.items():
        print(Fore.GREEN + f'{filePath}/{key}: {value}')

    print(Fore.RESET)
    max_count = max(emotions_count.values())

    for key, value in emotions_count.items():

        time.sleep(1)

        path_dir = f'{filePath}/{key}'
        count = value
        i = 0
        bar = tqdm(total=(max_count-count), desc=f'{filePath}/{key}: ')

        fileList = [f for f in os.listdir(path_dir) if os.path.isfile(os.path.join(path_dir, f))]

        if not os.path.exists(f'{filePath}/{key}/tmp'):
            os.mkdir(f'{filePath}/{key}/tmp')

        while count < max_count:

            if i >= value:
                i = 0
            elif i == value-1:
                i = 0
            else:
                i += 1

            random_noise(path_dir, fileList[i])
            bar.update(1)
            count += 1


def tmp_focus(focus_emotion: str, middle: ''):

    emotions = emotion_list

    if not os.path.exists(fr"DataSet/{middle}not_{focus_emotion}"):
        os.mkdir(fr"DataSet/{middle}not_{focus_emotion}")

    for emotion in emotions:
        if not emotion == focus_emotion:
            shutil.move(f"DataSet/{middle}{emotion}", f'DataSet/{middle}not_{focus_emotion}')


def train_val_split(n):

    if not os.path.exists(f'DataSet/train'):
        os.mkdir(f'DataSet/train')
    if not os.path.exists('DataSet/validation'):
        os.mkdir(f'DataSet/validation')

    for emotion in emotion_list:

        time.sleep(1)

        if not os.path.exists(f'DataSet/train/{emotion}'):
            os.mkdir(f'DataSet/train/{emotion}')
        if not os.path.exists(f'DataSet/validation/{emotion}'):
            os.mkdir(f'DataSet/validation/{emotion}')

        path_dir = f'DataSet/{emotion}'

        fileList = [f for f in os.listdir(path_dir) if os.path.isfile(os.path.join(path_dir, f))]

        lenFileList = len(fileList)
        newLenFileList = int(lenFileList * n) + 1

        for i in tqdm(range(0, lenFileList), desc=f'{emotion}'):
            if i < newLenFileList:
                file = fileList[i]
                src = f'DataSet/{emotion}/{file}'
                des = f'DataSet/train/{emotion}/{file}'
                shutil.move(src, des)
            else:

                file = fileList[i]
                src = f'DataSet/{emotion}/{file}'
                des = f'DataSet/validation/{emotion}/{file}'
                shutil.move(src, des)


def main():

    input_shape = (48, 48)

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
    )

    model = keras.saving.load_model('LiteEmotionAI_7.keras')
    model.summary()
    model.evaluate(val_ds)


if __name__ == '__main__':

    with cProfile.Profile() as pr:
        try:
            main()
        except (KeyboardInterrupt, OverflowError):
            traceback.print_exc()

    stats = pstats.Stats(pr)
    stats.sort_stats(pstats.SortKey.TIME)
    stats.dump_stats(filename='STATS.prof')
