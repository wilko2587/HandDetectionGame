import cv2
import mediapipe as mp
import time
import numpy as np
import normalisation_tools as nt

class handDetector():
    def __init__(self, nHands=1, detectionCon=0.5, trackCon=0.5, smoothness=0):
        self.detectionCon = detectionCon
        self.trackCon = trackCon
        self.landmarks = []
        self.nHands = nHands

        self.mpdetector = mp.solutions.hands.Hands(False, nHands,
                                                   self.detectionCon, self.trackCon)

        self.finger_landmarks = {0: 4, 1: 8, 2: 12, 3: 16, 4: 20}  # index in mp landmarks of fingers
        self.img = None
        self.smoothness = smoothness
        self.landmark_history = None # keep history of the prior landmarks in order to smooth out the landmarking

    def render(self, img):
        self.img = img
        #img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        #R, G, B = cv2.split(img)

        #output1_R = cv2.equalizeHist(R)
        #output1_G = cv2.equalizeHist(G)
        #output1_B = cv2.equalizeHist(B)

        #newimg = cv2.merge((output1_B, output1_G, output1_R))
        #effect = 0.5
        #img = effect*newimg.astype(np.float) + (1.-effect)*img.astype(np.float)
        #img = img.astype('uint8')
        #self.img = img

        self.detection = self.mpdetector.process(img)

        if self.detection.multi_hand_landmarks:
            for hand in self.detection.multi_hand_landmarks:
                mp.solutions.drawing_utils.draw_landmarks(self.img, hand,
                                                          mp.solutions.hands.HAND_CONNECTIONS)
        return img

    def genFingerPos(self, draw=True):
        # need to call render first
        scale = None
        self.landmarks = []
        if self.detection.multi_hand_landmarks:
            for handno in range(len(self.detection.multi_hand_landmarks)):
                xs, ys = [], []
                self.landmarks.append({})
                myHand = self.detection.multi_hand_landmarks[handno]
                for id, landmark in enumerate(myHand.landmark):
                    h, w, c = self.img.shape
                    x, y = int(landmark.x * w), int(landmark.y * h)  # true x/y positions of landmark scaled for imagesize

                    if self.landmark_history and self.smoothness:
                        try:
                            priorx = self.landmark_history[handno][id][0]
                            priory = self.landmark_history[handno][id][1]
                            x = int(priorx + (x - priorx) / self.smoothness)
                            y = int(priory + (y - priory) / self.smoothness)
                        except:
                            pass # easily fails if a landmark moves off screen. try statement is appropriate

                    xs.append(x)
                    ys.append(y)

                    self.landmarks[-1][id] = [x, y]

                if draw:
                    scale = (max(xs) - min(xs)) + (max(ys) - min(ys))
                    radius = int(scale / 100)
                    for i in range(len(xs)):
                        cv2.circle(self.img, (xs[i], ys[i]), radius, (255, 0, 255), cv2.FILLED)

        if len(self.landmarks) > 0:
            self.landmark_history = self.landmarks

        return self.landmarks, scale

    def getClick(self, thresh=30, handno=0):

        # returns True if thumb + index finger touch (distance between them is within thresh
        click = False
        if self.detection.multi_hand_landmarks:
            if len(self.detection.multi_hand_landmarks) > handno:
                thumb_pos, index_pos = self.landmarks[handno][4], self.landmarks[handno][8]
                thumb_pos = np.array(thumb_pos)
                index_pos = np.array(index_pos)
                dist = np.sqrt(np.sum(thumb_pos - index_pos) ** 2)
                if dist < thresh:
                    click = True
        return click

    def getIndexCoords(self, handno=0, draw=False):
        # returns fractional height of index finger on screen
        self.genFingerPos(draw=draw)
        if self.detection.multi_hand_landmarks:
            indexpos = self.landmarks[handno][8]
            return self.img, indexpos
        else:
            return self.img, None


class FingerTracker:
    def __init__(self, smoothing=0, invertx=False):
        self.detector = handDetector(nHands=1, smoothness=smoothing)
        self.cap = cv2.VideoCapture(0)
        self.coord_history = []
        self.invertx=invertx

    def getClick(self, thresh=30):
        success, img = self.cap.read()
        if success:
            self.detector.render(img)
            self.detector.genFingerPos(draw=False)
            return self.detector.getClick(thresh=thresh, handno=0)
        else:
            return False

    def getfingerpos(self, draw=False):
        success, img = self.cap.read()

        # normalisation
        #img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        #R, G, B = cv2.split(img)

        #output1_R = cv2.equalizeHist(R)
        #output1_G = cv2.equalizeHist(G)
        #output1_B = cv2.equalizeHist(B)

        #newimg = cv2.merge((output1_B, output1_G, output1_R))
        #effect = 0. # turn hist norm off
        #img = effect*newimg.astype(np.float) + (1.-effect)*img.astype(np.float)
        #img = img.astype('uint8')
        #self.img = img

        self.detector.render(img)
        img, indexcoords = self.detector.getIndexCoords(draw=draw)

        if draw:
            if indexcoords:
                cv2.putText(img, "index finger {}".format(indexcoords), (10, 70), cv2.FONT_HERSHEY_PLAIN, 1,
                            (255, 0, 255), 3)
            cv2.imshow("FingerTracker", img)

        if indexcoords:
            indexpos = [indexcoords[0]/img.shape[1], indexcoords[1]/img.shape[0]]
            if self.invertx:
                indexpos[0] = 1 - indexpos[0]
            self.coord_history.append(indexpos)
            return indexpos

        elif len(self.coord_history) > 0:
            return self.coord_history[-1]

        else:
            return 0, 0


def main():
    t0 = time.time()
    cap = cv2.VideoCapture(0)
    detector = handDetector(nHands=1, smoothness=5)
    while True:
        success, img = cap.read()

        # img = img[:, ::-1, :] # mirror the image

        img = detector.render(img)
        fingerpos, scale = detector.genFingerPos(draw=True)
        click = detector.getClick(handno=0) # click with left hand

        t1 = time.time()
        fps = 1 / (t1 - t0)
        t0 = t1

        cv2.putText(img, "FPS {}".format(int(fps)), (10, 70), cv2.FONT_HERSHEY_PLAIN, 3,
                    (255, 0, 255), 3)

        cv2.putText(img, "Click {}".format(click), (10, 150), cv2.FONT_HERSHEY_PLAIN, 3,
                    (255, 0, 255), 3)

        # if scale is not None:
        #    cv2.putText(img, "Hand Dist {0:.3g}".format(1e3/scale), (10, 150), cv2.FONT_HERSHEY_PLAIN, 3,
        #                (255, 0, 255), 3)

        cv2.imshow("Image", img)
        cv2.waitKey(1)


if __name__ == "__main__":
    main()
