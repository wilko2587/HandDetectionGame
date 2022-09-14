import cv2
import time
import numpy as np
import handdetector as hd
import os
import random
import pandas as pd
import gestureDetector as gd

def insidebox(x, y, box): # returns true if x, y inside the box
    return x > box[0] and y > box[1] and x < box[2] and y < box[3]

class Guesturegame:
    def __init__(self, cursorfunc, configfile='./guesturegameconfig/.', openingmode='startscreen', interval=1):

        self._getcursorpos = cursorfunc
        self.gestureDetector = gd.GestureDetector()
        self.configfile = configfile
        self.background = cv2.imread(os.path.join(configfile, "background.png"))
        self.startscreen = cv2.imread(os.path.join(configfile, "startscreen.png"))
        self.gameoverscreen = cv2.imread(os.path.join(configfile, "background.png"))
        self.instructionsscreen= cv2.imread(os.path.join(configfile, "instructions.png"))
        self.gestures = [cv2.imread(os.path.join(configfile, "{}.png".format(i))) for i in range(6)]
        self.playwindow_width = self.background.shape[1]
        self.playwindow_height = self.background.shape[0]
        self.guesturesize = (50, 50)
        self.start = False # turn True to start the game running
        self.birdcoords = (0,0,0,0) # ymin, ymax, xmin, xmax
        self.score = 0
        self.highscore = pd.read_csv(os.path.join(configfile, "highscore.csv"), index_col=0).values[0][0]
        self.mode = openingmode
        self.interval = interval

    def _showcursor(self, img, x, y, alpha=0.1, counter=0):
        radius = 20
        selectionConfirmed = False # turns True if counter = 15
        overlay = np.zeros_like(img)
        cv2.circle(overlay, (int(x), int(y)), radius, (255, 255, 255), cv2.FILLED)
        img = cv2.addWeighted(img, 1, overlay, alpha, 0)
        if counter > 0:
            overlay2 = np.zeros_like(img)
            cv2.circle(overlay2, (int(x), int(y)), min(radius, int(counter/2)), (255, 255, 255), cv2.FILLED)
            img = cv2.addWeighted(img, 1, overlay2, 1, 0) # add with alpha 1
            if int(counter/2) == radius:
                selectionConfirmed = True
        return img, selectionConfirmed

    def run(self):
        if self.mode == 'startscreen':
            self._runStartScreen()
        if self.mode == "showinstructions":
            self._runInstructionsScreen()
        if self.mode == "playgame":
            self._rungame()
        if self.mode == 'gameover':
            self._gameOverScreen()


    def _runStartScreen(self):

        playy, playx, pdy, pdx = self.startscreen.shape[0]*0.57, self.startscreen.shape[1]*0.27, 120, 80
        playbox = (int(playy), int(playx), int(playy+pdy), int(playx+pdx))
        insy, insx, idy, idx = self.startscreen.shape[0]*0.52, self.startscreen.shape[1]*0.42, 200, 80
        insbox = (int(insy), int(insx), int(insy+idy), int(insx+idx))
        playcounter = 0 # counts how many loops the cursor has been inside the "play" bounding box
        inscounter = 0
        while True:
            cursorx, cursory = self._getcursorpos()
            x, y = cursorx * self.playwindow_width, cursory * self.playwindow_height
            img, s = self._showcursor(self.startscreen.copy(), x, y, alpha=0.1, counter=max(playcounter, inscounter))
            img = cv2.rectangle(img, playbox[0:2], playbox[2:], (255, 0, 0), 1)
            img = cv2.rectangle(img, insbox[0:2], insbox[2:], (0, 0, 255), 1)
            cv2.imshow("Image", img)
            cv2.waitKey(1)
            if insidebox(x, y, playbox):
                playcounter += 1
            else:
                playcounter = 0

            if insidebox(x, y, insbox):
                inscounter += 1
            else:
                inscounter = 0

            if playcounter > 0 and s:
                self.mode = 'playgame'
                break

            if inscounter > 0 and s:
                self.mode = 'showinstructions'
                break

        return

    def _runInstructionsScreen(self):

        backy, backx, backdy, backdx = self.instructionsscreen.shape[0]*0.55, self.startscreen.shape[1]*0.47, 120, 80
        backbox = (int(backy), int(backx), int(backy+backdy), int(backx+backdx))
        backcounter = 0 # counts how many loops the cursor has been inside the "play" bounding box
        while True:
            cursorx, cursory = self._getcursorpos()
            x, y = cursorx * self.playwindow_width, cursory * self.playwindow_height
            img, s = self._showcursor(self.instructionsscreen.copy(), x, y, alpha=0.1, counter=backcounter)
            img = cv2.rectangle(img, backbox[0:2], backbox[2:], (255, 0, 255), 1)
            cv2.imshow("Image", img)
            cv2.waitKey(1)
            if insidebox(x, y, backbox):
                backcounter += 1
            else:
                backcounter = 0

            if backcounter > 0 and s:
                self.mode = 'startscreen'
                break
        return


    def _runCountdownScreen(self): # runs the screen saying "READY... GO!"
        img = self.background.copy()
        font = cv2.FONT_HERSHEY_PLAIN
        fontscale = 6
        thickness = 2
        color = (255, 0, 255)
        (width, height), baseline = cv2.getTextSize("READY", font, fontscale, thickness)
        img = cv2.putText(img, "READY", (int(img.shape[1] / 2 - width / 2), int(img.shape[0] / 2 - height / 2)),
                          font, fontscale, color, thickness)
        cv2.imshow("Image", img)
        cv2.waitKey(1)
        time.sleep(1.5)
        img = self.background.copy()
        (width, height), baseline = cv2.getTextSize("GO!", font, fontscale, thickness)
        img = cv2.putText(img, "GO!", (int(img.shape[1] / 2 - width / 2), int(img.shape[0] / 2 - height / 2)),
                          font, fontscale, color, thickness)
        cv2.imshow("Image", img)
        cv2.waitKey(1)
        time.sleep(0.5)
        return

    def _rungame(self):
        self.__init__(self._getcursorpos, configfile=self.configfile, openingmode='playgame')
        self._runCountdownScreen()
        start_time = time.time()
        success = True
        while time.time() - start_time <= 10:
            success = self._tickgame(success, 10 + start_time - time.time())
        self.mode = 'gameover'
        return

    def _tickgame(self, nextFrame, timeremaining):

        if nextFrame:
            self.gesture = np.random.randint(4)

        gestureImg = self.gestures[self.gesture]

        img = self.background.copy()

        overlay = np.zeros_like(img)
        overlay[0:gestureImg.shape[0], 0:gestureImg.shape[1], :] += gestureImg
        img[overlay != 0] = 0
        img += overlay

        packed = self.gestureDetector.predict(show=False, niterations=1)
        flat_landmarks, pred = packed[0][0], packed[1][0]

        success = False
        if flat_landmarks is not None and pred is not None:
            if pred == self.gesture:
                color = (0, 255, 0)
                self.score += 1
                success = True
            else:
                color = (0, 0, 255)

            for i in range(0, len(flat_landmarks), 2):
                x = 500 + int(flat_landmarks[i] * img.shape[1] / 4)
                y = 200 + int(flat_landmarks[i + 1] * img.shape[0] / 4)
                cv2.circle(img, (x, y), 5, color, cv2.FILLED)

        textx, texty = 20, 20
        scoretext = "score: {}".format(self.score)
        cv2.putText(img, scoretext, (textx, texty), cv2.FONT_HERSHEY_PLAIN, 2,
                    (0, 0, 255), 1)
        (width, height1), baseline = cv2.getTextSize(scoretext, cv2.FONT_HERSHEY_PLAIN, 2, 1)
        cv2.putText(img, "highscore: {}".format(self.highscore), (textx, texty+height1+10),
                    cv2.FONT_HERSHEY_PLAIN, 2, (0, 0, 0), 1)
        (width, height2), baseline = cv2.getTextSize(scoretext, cv2.FONT_HERSHEY_PLAIN, 2, 1)
        cv2.putText(img, "Remaining Time: {}".format(int(timeremaining)), (textx, texty+height1+10 + height2 + 10),
                    cv2.FONT_HERSHEY_PLAIN, 2, (0, 0, 0), 1)

        cv2.imshow("Image", img)
        cv2.waitKey(1)
        return success

    def _gameOverScreen(self):
        img = self.gameoverscreen
        font = cv2.FONT_HERSHEY_PLAIN
        fontscale = 2
        thickness = 2
        color = (255, 0, 0)
        y = 300
        margin = 30
        if self.score > self.highscore:
            newhighscore = pd.DataFrame(index=['highscore'], columns=['value'], data=self.score)
            newhighscore.to_csv(os.path.join(self.configfile, 'highscore.csv'))
            self.highscore = self.score
            (width, height), baseline = cv2.getTextSize("NEW HIGH SCORE!".format(self.score), font, fontscale, thickness)
            img = cv2.putText(img, "NEW HIGH SCORE!", (int(img.shape[1] / 2 - width / 2), int(y - height / 2)),
                              font, fontscale, color, thickness)
            y += height/2 + margin

        (width, height), baseline = cv2.getTextSize("score {}".format(self.score), font, fontscale, thickness)
        img = cv2.putText(img, "score {}".format(self.score), (int(img.shape[1] / 2 - width / 2), int(y - height / 2)),
                          font, fontscale, color, thickness)
        y += height / 2 + margin

        (width, height), baseline = cv2.getTextSize("highscore {}".format(self.highscore), font, fontscale, thickness)
        img = cv2.putText(img, "highscore {}".format(self.highscore), (int(img.shape[1] / 2 - width / 2), int(y - height / 2)),
                          font, fontscale, color, thickness)
        y += height / 2 + margin

        rety, retx, retdy, retdx = self.startscreen.shape[0]*0.55, self.startscreen.shape[1]*0.38, 150, 60
        retbox = (int(rety), int(retx), int(rety+retdy), int(retx+retdx))
        retrycounter = 0 # counts how many loops the cursor has been inside the "play" bounding box
        imgcopy = cv2.rectangle(img, retbox[0:2], retbox[2:], (255, 0, 255), 1)
        while True:
            cursorx, cursory = self._getcursorpos()
            x, y = cursorx * self.playwindow_width, cursory * self.playwindow_height
            img, s = self._showcursor(imgcopy, x, y, alpha=0.1, counter=retrycounter)
            cv2.imshow("Image", img)
            cv2.waitKey(1)
            if insidebox(x, y, retbox):
                retrycounter += 1
            else:
                retrycounter = 0

            if retrycounter > 0 and s:
                self.mode = 'playgame'
                break
        return

def main():
    detector = hd.FingerTracker(smoothing=5, invertx=True)
    gg = Guesturegame(detector.getfingerpos)
    while True:
        gg.run()

if __name__ == '__main__':
    main()