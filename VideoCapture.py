import keras.src.models.functional
import numpy as np
import traceback
import cProfile
import pstats
import keras
import time
import cv2

# snakeviz ./STATS.prof
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')


def main():

    model = keras.saving.load_model('LiteEmotionAI_7.keras')
    vid = cv2.VideoCapture(0)

    while True:

        start_t = time.perf_counter()

        ret, img = vid.read()

        if ret:

            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            faceList = []
            faceListCord = []
            faces = face_cascade.detectMultiScale(gray, 1.3, minNeighbors=5)

            for i, (x, y, w, h) in enumerate(faces):

                mulyiplyer = 1.2

                x = int(x/mulyiplyer)
                y = int(y/mulyiplyer)
                w = int(w*mulyiplyer)
                h = int(h*mulyiplyer)

                cv2.rectangle(img, (x, y), (x + w, y + h), (255, 255, 0), 2)
                face = img[y:y + h, x:x + w]
                face = cv2.resize(face, (48, 48))

                faceList.append(face)
                faceListCord.append((x, y, w, h))

            if faceList:
                font = cv2.FONT_HERSHEY_SIMPLEX
                fontScale = 1
                thickness = 2

                x = np.asarray(faceList)
                predictions = model.predict(x)
                for i, prediction in enumerate(predictions):
                    org = (faceListCord[i][0], faceListCord[i][1])
                    score = float(keras.ops.sigmoid(prediction))
                    happy = f'{100 * (1 - score):.2f}'
                    not_happy = f'{100 * score:.2f}'
                    if happy >= not_happy:
                        img = cv2.putText(img, f'Happy: {happy}%', org, font, fontScale,
                                          (0, 255, 0), thickness, cv2.LINE_AA, False)
                    else:
                        img = cv2.putText(img, f'Not Happy: {not_happy}%', org, font, fontScale,
                                          (0, 0, 255), thickness, cv2.LINE_AA, False)

            end_t = time.perf_counter()
            dif_t = end_t-start_t
            FPS = int(1/dif_t)
            img = cv2.putText(img, f'FPS: {FPS}', (0, 465), cv2.FONT_HERSHEY_SIMPLEX,  1, (0, 255, 0),
                              1, cv2.LINE_AA, False)
            cv2.imshow('webcam', img)
            print(f'FPS: {FPS}')

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    vid.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':

    with cProfile.Profile() as pr:
        try:
            main()
        except (KeyboardInterrupt, OverflowError):
            traceback.print_exc()

    stats = pstats.Stats(pr)
    stats.sort_stats(pstats.SortKey.TIME)
    stats.dump_stats(filename='STATS.prof')
