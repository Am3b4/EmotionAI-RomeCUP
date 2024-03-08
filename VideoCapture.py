import keras.src.models.functional
import numpy as np
import traceback
import cProfile
import pstats
import keras
import cv2

# snakeviz ./STATS.prof
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')


def main():

    model = keras.saving.load_model('NEW_save_at_8.keras')
    vid = cv2.VideoCapture(0)

    while True:

        ret, img = vid.read()

        if ret:

            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            faceList = []
            faceListCord = []
            faces = face_cascade.detectMultiScale(gray, 1.3, minNeighbors=5)
            try:
                for i, (x, y, w, h) in enumerate(faces):

                    padding = 1.2

                    nw = w*padding
                    nh = h*padding

                    difw = nw-w
                    difh = nh-h

                    x = int(x-difw/2)
                    y = int(y-difh/2)
                    w = int(nw)
                    h = int(nh)

                    face = img[y:y + h, x:x + w]
                    face = cv2.resize(face, (48, 48))
                    cv2.rectangle(img, (x, y), (x + w, y + h), (255, 255, 0), 2)

                    faceList.append(face)
                    faceListCord.append((x, y, w, h))

            except cv2.error:
                pass

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

            cv2.imshow('webcam', img)

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
