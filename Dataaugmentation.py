import numpy as np
import random
import cv2
import os


def rotate90(img):
    return cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)


def rotate180(img):
    return cv2.rotate(img, cv2.ROTATE_180)


def rotate270(img):
    return cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)


def flipY(img):
    return cv2.flip(img, 1)


def rotate90FlipX(img):
    return cv2.flip(rotate90(img), 1)


def rotate180FlipY(img):
    return cv2.flip(rotate180(img), 1)


def rotate270FlipX(img):
    return cv2.flip(rotate270(img), 1)


def noise_S_and_P(image, prob=0.015):
    output = np.zeros(image.shape, np.uint8)
    thres = 1 - prob
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            rdn = random.random()
            if rdn < prob:
                output[i][j] = 64
            elif rdn > thres:
                output[i][j] = 191
            else:
                output[i][j] = image[i][j]
    return output


def noise_Gauss(img):
    mean = 0
    var = 9
    sigma = var ** 1
    x, y, z = img.shape
    gaussian = np.random.normal(mean, sigma, (x, y))  # np.zeros((224, 224), np.float32)

    noisy_image = np.zeros(img.shape, np.float32)

    if len(img.shape) == 2:
        noisy_image = img + gaussian
    else:
        noisy_image[:, :, 0] = img[:, :, 0] + gaussian
        noisy_image[:, :, 1] = img[:, :, 1] + gaussian
        noisy_image[:, :, 2] = img[:, :, 2] + gaussian

    cv2.normalize(noisy_image, noisy_image, 0, 255, cv2.NORM_MINMAX, dtype=-1)
    noisy_image = noisy_image.astype(np.uint8)
    return noisy_image


def random_noise(path_dir, file_name):

    augmentationDict = {'_r90': rotate90, '_r180': rotate180, '_r270': rotate270, '_fY': flipY, '_r90_fX': rotate90FlipX,
                        '_r180_fY': rotate180FlipY, '_r270_fX': rotate270FlipX, '_G': noise_Gauss, '_SaP': noise_S_and_P}

    augmentationFileFlagList = ['_r90', '_r180', '_r270', '_fY', '_r90_fX', '_r180_fY', '_r270_fX', '_G', '_SaP']
    img = cv2.imread(f'{path_dir}/{file_name}')

    idx = random.randint(0, 8)
    new_file_name = f'{file_name[:-4]}{augmentationFileFlagList[idx]}.jpg'

    tmpAugmentedFileFlagList = ['_r90', '_r180', '_r270', '_fY', '_r90_fX', '_r180_fY', '_r270_fX', '_G', '_SaP']

    i = 8
    while os.path.exists(f'{path_dir}/tmp/{new_file_name}'):
        if tmpAugmentedFileFlagList[idx] in ['_G', '_S', '_SaP']:
            j = 1
            while os.path.exists(f'{path_dir}/tmp/{new_file_name}'):
                new_file_name = f'{file_name[:-4]}{tmpAugmentedFileFlagList[idx]}_{j}.jpg'
                j += 1
            else:
                cv2.imwrite(f'{path_dir}/tmp/{new_file_name}', (augmentationDict[tmpAugmentedFileFlagList[idx]])(img))
                return True

        tmpAugmentedFileFlagList.pop(idx)
        i -= 1
        idx = random.randint(0, i)
        new_file_name = f'{file_name[:-4]}{tmpAugmentedFileFlagList[idx]}.jpg'
    else:
        cv2.imwrite(f'{path_dir}/tmp/{new_file_name}', (augmentationDict[tmpAugmentedFileFlagList[idx]])(img))
        return True
