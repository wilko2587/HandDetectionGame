import cv2
import time
import numpy as np
import handdetector as hd
import os
import random
import pandas as pd
import gestureDetector as gd

class pipe:
    def __init__(self, x, y, width, yend=0, type='up'):
        self.x = int(x)
        self.y = int(y)
        self.width = int(width)
        self.startpoint = (self.x, self.y)
        self.endpoint = (self.x + width, yend)
        self.yend = yend
        self.type = type

    def move(self, dx):
        self.x += - dx
        self.startpoint = (self.x, self.y)
        self.endpoint = (self.x + self.width, self.yend)


class pipes:
    def __init__(self, x, y, width, gap=100, ymin=0, ymax=100):
        self.pipeup = pipe(x, y + int(gap / 2), width, yend=ymax, type='up')
        self.pipedown = pipe(x, y - int(gap / 2), width, yend=ymin, type='down')

    def move(self, dx):
        self.pipeup.move(dx)
        self.pipedown.move(dx)

    def __getitem__(self, index):
        return (self.pipeup, self.pipedown)[index]


def insidebox(x, y, box):  # returns true if x, y inside the box
    return x > box[0] and y > box[1] and x < box[2] and y < box[3]


def showcursor(img, x, y, alpha=0.1, counter=0):
    radius = 20
    selectionConfirmed = False  # turns True if counter = 15
    overlay = np.zeros_like(img)
    cv2.circle(overlay, (int(x), int(y)), radius, (255, 255, 255), cv2.FILLED)
    img = cv2.addWeighted(img, 1, overlay, alpha, 0)
    if counter > 0:
        overlay2 = np.zeros_like(img)
        cv2.circle(overlay2, (int(x), int(y)), min(radius, int(counter / 2)), (255, 255, 255), cv2.FILLED)
        img = cv2.addWeighted(img, 1, overlay2, 1, 0)  # add with alpha 1
        if int(counter / 2) == radius:
            selectionConfirmed = True
    return img, selectionConfirmed


class GameSuper:
    def __init__(self, configfile='./generalConfig/'):
        self.detector = hd.FingerTracker(smoothing=5, invertx=True)
        self._getcursorpos = self.detector.getfingerpos
        self.startscreen = cv2.imread(os.path.join(configfile, "background.png"))
        self.playwindow_width = self.startscreen.shape[1]
        self.playwindow_height = self.startscreen.shape[0]
        self.mode = 'startscreen'
        self.draw = False

    def begin(self):
        while True:
            if self.mode == 'startscreen':
                self._runStartScreen()
            if self.mode == 'flappyfinger':
                self._beginFlappyFinger()
            if self.mode == 'gesturegame':
                self._beginGestureGame()

            self.mode = 'startscreen'

    def _runStartScreen(self):

        playy, playx, pdy, pdx = self.startscreen.shape[0] * 0.19, self.startscreen.shape[1] * 0.32, 165, 120
        playbox = (int(playy), int(playx), int(playy + pdy), int(playx + pdx))
        insy, insx, idy, idx = self.startscreen.shape[0] * 0.82, self.startscreen.shape[1] * 0.32, 275, 120
        insbox = (int(insy), int(insx), int(insy + idy), int(insx + idx))
        playcounter = 0  # counts how many loops the cursor has been inside the "play" bounding box
        inscounter = 0
        while True:
            cursorx, cursory = self._getcursorpos(draw=self.draw)
            x, y = cursorx * self.playwindow_width, cursory * self.playwindow_height
            img, s = showcursor(self.startscreen.copy(), x, y, alpha=0.1, counter=max(playcounter, inscounter))
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
                self.mode = 'flappyfinger'
                break

            if inscounter > 0 and s:
                self.mode = 'gesturegame'
                break
        return

    def _beginFlappyFinger(self):
        fb = FlappyBird(self._getcursorpos)
        fb.run()
        del(fb)
        return

    def _beginGestureGame(self):
        return_signal = False

        gg = Gesturegame(self._getcursorpos)
        while not return_signal:
            return_signal = gg.run()
        return


class MiniGameSuper:
    def __init__(self, cursorFunc, configfile, openingmode, draw=False):
        self.configfile = configfile
        self._getcursorpos = cursorFunc
        self.background = cv2.imread(os.path.join(configfile, "background.png"))
        self.startscreen = cv2.imread(os.path.join(configfile, "startscreen.png"))
        self.gameoverscreen = cv2.imread(os.path.join(configfile, "gameover.png"))
        self.instructionsscreen = cv2.imread(os.path.join(configfile, "instructions.png"))
        self.playwindow_width = self.background.shape[1]
        self.playwindow_height = self.background.shape[0]
        self.mode = openingmode
        self.draw = draw

    def _runStartScreen(self):

        playy, playx, pdy, pdx = self.startscreen.shape[0] * 0.57, self.startscreen.shape[1] * 0.27, 120, 80
        playbox = (int(playy), int(playx), int(playy + pdy), int(playx + pdx))
        insy, insx, idy, idx = self.startscreen.shape[0] * 0.52, self.startscreen.shape[1] * 0.42, 200, 80
        insbox = (int(insy), int(insx), int(insy + idy), int(insx + idx))
        playcounter = 0  # counts how many loops the cursor has been inside the "play" bounding box
        inscounter = 0
        while True:
            cursorx, cursory = self._getcursorpos(draw=self.draw)
            x, y = cursorx * self.playwindow_width, cursory * self.playwindow_height
            img, s = showcursor(self.startscreen.copy(), x, y, alpha=0.1, counter=max(playcounter, inscounter))
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

        backy, backx, backdy, backdx = self.instructionsscreen.shape[0] * 0.55, self.startscreen.shape[
            1] * 0.47, 120, 80
        backbox = (int(backy), int(backx), int(backy + backdy), int(backx + backdx))
        backcounter = 0  # counts how many loops the cursor has been inside the "play" bounding box
        while True:
            cursorx, cursory = self._getcursorpos(draw=self.draw)
            x, y = cursorx * self.playwindow_width, cursory * self.playwindow_height
            img, s = showcursor(self.instructionsscreen.copy(), x, y, alpha=0.1, counter=backcounter)
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

    def _runPostGameScreen(self, img, textcolor, text):  # runs the screen saying "READY... GO!"

        font = cv2.FONT_HERSHEY_PLAIN
        fontscale = 6
        thickness = 3
        (width, height), baseline = cv2.getTextSize(text, font, fontscale, thickness)
        img = cv2.putText(img, text, (int(img.shape[1] / 2 - width / 2), int(img.shape[0] / 2 - height / 2)),
                          font, fontscale, textcolor, thickness)
        cv2.imshow("Image", img)
        cv2.waitKey(1)
        time.sleep(1.5)
        return

    def _rungame(self):
        pass

    def _runGameOverScreen(self):
        pass

    def run(self):
        if self.mode == 'startscreen':
            self._runStartScreen()
        if self.mode == "showinstructions":
            self._runInstructionsScreen()
        if self.mode == "playgame":
            self._rungame()
        if self.mode == 'gameover':
            replay = self._runGameOverScreen()
            if replay:
                self.mode = 'playgame'
                return False
        return True


class FlappyBird(MiniGameSuper):
    def __init__(self, cursorFunc, configfile='./flappybirdconfig/.', pipespacing=0.35, openingmode='startscreen'):

        # cursorfunc = function which when called with cursorfunc() returns an x,y position of a cursor to be used
        # throughout the game

        super().__init__(cursorFunc, configfile, openingmode)

        self.pipes = []  # pipes will be stored in this list. Pipes have 2 attributes: x position and height
        self.pipespacing = pipespacing
        self.pipexmax = self.playwindow_width
        self.pipewidth = 50
        self.birdsize = (50, 50)
        self.birdimage = cv2.resize(cv2.imread(os.path.join(configfile, "bird.png")), self.birdsize)
        self.start = False  # turn True to start the game running
        self.birdcoords = (0, 0, 0, 0)  # ymin, ymax, xmin, xmax
        self.score = 0
        self.highscore = pd.read_csv(os.path.join(configfile, "highscore.csv"), index_col=0).values[0][0]

    def _genpipes(self):
        spacing = self.pipespacing * self.playwindow_width
        while self.pipexmax + spacing < self.playwindow_width * 1.3:
            self.pipexmax += spacing
            newpipe = pipes(self.pipexmax, random.randint(100, self.playwindow_height - 100), self.pipewidth,
                            ymin=0, ymax=self.playwindow_height)
            self.pipes.append(newpipe)

    def _drawpipes(self, img):
        # cv2.rectangle(image, start_point, end_point, color, thickness)
        for pipepair in self.pipes:
            for pipe in pipepair:
                if pipe.x >= 0 and pipe.x + pipe.width <= self.playwindow_width:
                    img = cv2.rectangle(img, pipe.startpoint, pipe.endpoint, (255, 0, 0), cv2.FILLED)
                    # M = cv2.getAffineTransform(pts2, pts1)
                    # dst = cv2.warpAffine(img2, M, (cols, rows))
                    # final = cv2.addWeighted(dst, 0.5, img1, 0.5, 1)
        return img

    def _updateBirdPos(self, img, x, y):

        ymin = int(y - self.birdsize[0] / 2)
        xmin = int(x - self.birdsize[1] / 2)

        # must make sure the bird falls inside the range of the image
        if ymin < 0:
            ymin = 0
        if xmin < 0:
            xmin = 0
        if xmin + self.birdsize[1] > self.playwindow_width:
            xmin = self.playwindow_width - self.birdsize[1]
        if ymin + self.birdsize[0] > self.playwindow_height:
            ymin = self.playwindow_height - self.birdsize[0]

        self.birdcoords = [ymin, ymin + self.birdsize[1], xmin, xmin + self.birdsize[1]]
        img[ymin:ymin + self.birdsize[0], xmin:xmin + self.birdsize[1]] = self.birdimage
        return img

    def _detectcollision(self):
        for pipepair in self.pipes:
            for pipe in pipepair:
                xcondition = self.birdcoords[3] > pipe.x and self.birdcoords[2] < pipe.x + self.pipewidth
                ycondition_up = self.birdcoords[1] > pipe.y and pipe.type == 'up'
                ycondition_down = self.birdcoords[0] < pipe.y and pipe.type == 'down'
                if xcondition and (ycondition_up or ycondition_down):
                    return True
        return False

    def _runCountdownScreen(self):  # runs the screen saying "READY... GO!"
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
        self.__init__(self._getcursorpos, configfile=self.configfile, pipespacing=self.pipespacing,
                      openingmode='playgame')
        collision = False
        img = None
        self._runCountdownScreen()
        while not collision:
            collision, img = self._tickgame()

        self._runPostGameScreen(img, (0, 0, 0), 'You Crashed!')
        self.mode = 'gameover'
        return

    def _tickgame(self):
        cursorx, cursory = self._getcursorpos(draw=self.draw)
        x, y = cursorx * self.playwindow_width, cursory * self.playwindow_height
        dx = 10
        for pipepair in self.pipes:
            for pipe in pipepair:
                pipe.move(dx)
            if pipepair[0].x < 0:
                self.pipes.remove(pipepair)
                del pipepair

        img = self.background.copy()
        self._genpipes()
        img = self._drawpipes(img)
        img = self._updateBirdPos(img, x, y)
        self.pipexmax += -dx
        collision = self._detectcollision()
        self.score += dx

        textx, texty = 20, 20
        scoretext = "score: {}".format(self.score)
        cv2.putText(img, scoretext, (textx, texty), cv2.FONT_HERSHEY_PLAIN, 2,
                    (0, 0, 255), 1)
        (width, height), baseline = cv2.getTextSize(scoretext, cv2.FONT_HERSHEY_PLAIN, 2, 1)
        cv2.putText(img, "highscore: {}".format(self.highscore), (textx, texty + height + 10),
                    cv2.FONT_HERSHEY_PLAIN, 2, (0, 0, 0), 1)

        cv2.imshow("Image", img)
        cv2.waitKey(1)
        return collision, img

    def _runGameOverScreen(self):
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
            (width, height), baseline = cv2.getTextSize("NEW HIGH SCORE!".format(self.score), font, fontscale,
                                                        thickness)
            img = cv2.putText(img, "NEW HIGH SCORE!", (int(img.shape[1] / 2 - width / 2), int(y - height / 2)),
                              font, fontscale, color, thickness)
            y += height / 2 + margin

        (width, height), baseline = cv2.getTextSize("score {}".format(self.score), font, fontscale, thickness)
        img = cv2.putText(img, "score {}".format(self.score), (int(img.shape[1] / 2 - width / 2), int(y - height / 2)),
                          font, fontscale, color, thickness)
        y += height / 2 + margin

        (width, height), baseline = cv2.getTextSize("highscore {}".format(self.highscore), font, fontscale, thickness)
        img = cv2.putText(img, "highscore {}".format(self.highscore),
                          (int(img.shape[1] / 2 - width / 2), int(y - height / 2)),
                          font, fontscale, color, thickness)
        y += height / 2 + margin

        rety, retx, retdy, retdx = self.startscreen.shape[0] * 0.55, self.startscreen.shape[1] * 0.38, 160, 60
        retbox = (int(rety), int(retx), int(rety + retdy), int(retx + retdx))
        endy, endx, enddy, enddx = self.startscreen.shape[0] * 0.52, self.startscreen.shape[1] * 0.50, 210, 60
        endbox = (int(endy), int(endx), int(endy + enddy), int(endx + enddx))
        retrycounter = 0  # counts how many loops the cursor has been inside the "play" bounding box
        endcounter = 0
        imgcopy = cv2.rectangle(img, retbox[0:2], retbox[2:], (255, 0, 255), 1)
        imgcopy = cv2.rectangle(imgcopy, endbox[0:2], endbox[2:], (255, 0, 255), 1)
        replay = False

        while True:
            cursorx, cursory = self._getcursorpos()
            x, y = cursorx * self.playwindow_width, cursory * self.playwindow_height
            img, s = showcursor(imgcopy, x, y, alpha=0.1, counter=retrycounter)
            img, s2 = showcursor(img, x, y, alpha=0.1, counter=endcounter)
            cv2.imshow("Image", img)
            cv2.waitKey(1)
            if insidebox(x, y, retbox):
                retrycounter += 1
            else:
                retrycounter = 0

            if insidebox(x, y, endbox):
                endcounter += 1
            else:
                endcounter = 0

            if retrycounter > 0 and s:
                replay = True
                self.mode = 'startscreen'
                break

            if endcounter > 0 and s2:
                break

        return replay


class Gesturegame(MiniGameSuper):
    def __init__(self, cursorFunc, configfile='./guesturegameconfig/.', interval=1, openingmode='startscreen'):
        super().__init__(cursorFunc, configfile, openingmode)
        self.gestureDetector = gd.GestureDetector()

        self.gestures = [cv2.imread(os.path.join(configfile, "{}.png".format(i))) for i in range(6)]
        self.numGestures = len(self.gestures)
        self.guesturesize = (50, 50)
        self.start = False  # turn True to start the game running
        self.score = 0
        self.highscore = pd.read_csv(os.path.join(configfile, "highscore.csv"), index_col=0).values[0][0]
        self.mode = openingmode
        self.interval = interval
        self.past_gestures = [-1]*5
        self.gesture = None

    def _runStartScreen(self):

        playy, playx, pdy, pdx = self.startscreen.shape[0] * 0.57, self.startscreen.shape[1] * 0.27, 120, 80
        playbox = (int(playy), int(playx), int(playy + pdy), int(playx + pdx))
        insy, insx, idy, idx = self.startscreen.shape[0] * 0.52, self.startscreen.shape[1] * 0.42, 200, 80
        insbox = (int(insy), int(insx), int(insy + idy), int(insx + idx))
        playcounter = 0  # counts how many loops the cursor has been inside the "play" bounding box
        inscounter = 0
        while True:
            cursorx, cursory = self._getcursorpos()
            x, y = cursorx * self.playwindow_width, cursory * self.playwindow_height
            img, s = showcursor(self.startscreen.copy(), x, y, alpha=0.1, counter=max(playcounter, inscounter))
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

    def _runCountdownScreen(self):  # runs the screen saying "READY... GO!"
        img = self.background.copy()
        font = cv2.FONT_HERSHEY_PLAIN
        fontscale = 6
        thickness = 2
        color = (255, 255, 255)
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
        self.priorgesture = []
        return

    def _rungame(self):
        self.__init__(self._getcursorpos, configfile=self.configfile, openingmode='playgame')
        self._runCountdownScreen()
        start_time = time.time()
        nextFrame = True
        img = None
        self._successcounter = 0
        self._pastSuccess = False
        while time.time() - start_time <= 10:
            success, img = self._tickgame(nextFrame, 10 + start_time - time.time())
            nextFrame = False
            if self._pastSuccess:
                self._successcounter += 4
            if self._successcounter >= 20:
                nextFrame = True
                self._pastSuccess = False
                self._successcounter = 0
        self.mode = 'gameover'
        self._runPostGameScreen(img, (255, 255, 255), "Time Up!")
        return

    def _tickgame(self, nextFrame, timeremaining):

        if nextFrame:
            newGesture = None
            while newGesture == self.gesture or newGesture is None:
                newGesture = np.random.randint(self.numGestures)
            self.gesture = newGesture

        gestureImg = self.gestures[self.gesture]

        img = self.background.copy()

        overlay = np.zeros_like(img)
        overlay[200:gestureImg.shape[0]+200, 100:gestureImg.shape[1]+100, :] += gestureImg
        img[overlay != 0] = 0
        img += overlay

        packed = self.gestureDetector.predict(show=False, niterations=1)
        flat_landmarks, pred = packed[0][0], packed[1][0]
        self.priorgesture.append(pred)

        success = False
        if flat_landmarks is not None and pred is not None:
            if self.priorgesture[-5:].count(self.gesture) >= 3 and self._pastSuccess == False: # if mode of last 5 observations matches
                color = (0, 255, 0) # green color for "success!"
                self.score += 1
                success = True
                self._pastSuccess = True
            else:
                color = (100, 100, 255)

            radius = self._successcounter + 5
            alpha = min(1., 1. / (self._successcounter + 1)**(-1./3)) # fade the green success circles as they grow
            for i in range(0, len(flat_landmarks), 2):
                x = 300 + int(flat_landmarks[i] * img.shape[1] / 2)
                y = 200 + int(flat_landmarks[i + 1] * img.shape[0] / 2)
                cv2.circle(img, (x, y), 10, color, cv2.FILLED)
                if self._pastSuccess:
                    success_overlay = np.zeros_like(img)
                    success_overlay = cv2.circle(success_overlay, (x, y), radius, (0, 255, 0), cv2.FILLED)
                    img = cv2.addWeighted(img, 1, success_overlay, alpha, 0)

        textx, texty = 20, 20
        scoretext = "score: {}".format(self.score)
        cv2.putText(img, scoretext, (textx, texty), cv2.FONT_HERSHEY_PLAIN, 2,
                    (255, 255, 255), 1)
        (width, height1), baseline = cv2.getTextSize(scoretext, cv2.FONT_HERSHEY_PLAIN, 2, 1)
        cv2.putText(img, "highscore: {}".format(self.highscore), (textx, texty + height1 + 10),
                    cv2.FONT_HERSHEY_PLAIN, 2, (255, 255, 255), 1)
        (width, height2), baseline = cv2.getTextSize(scoretext, cv2.FONT_HERSHEY_PLAIN, 2, 1)
        cv2.putText(img, "Remaining Time: {}".format(int(timeremaining)), (int(img.shape[0]/2 - 200), int(img.shape[1] / 2 + 200)),
                    cv2.FONT_HERSHEY_PLAIN, 4, (255, 255, 255), 2)

        cv2.imshow("Image", img)
        cv2.waitKey(1)
        return success, img

    def _runGameOverScreen(self):
        img = self.gameoverscreen
        font = cv2.FONT_HERSHEY_PLAIN
        fontscale = 2
        thickness = 2
        color = (255, 255, 255)
        y = 300
        margin = 30
        if self.score > self.highscore:
            newhighscore = pd.DataFrame(index=['highscore'], columns=['value'], data=self.score)
            newhighscore.to_csv(os.path.join(self.configfile, 'highscore.csv'))
            self.highscore = self.score
            (width, height), baseline = cv2.getTextSize("NEW HIGH SCORE!".format(self.score), font, fontscale,
                                                        thickness)
            img = cv2.putText(img, "NEW HIGH SCORE!", (int(img.shape[1] / 2 - width / 2), int(y - height / 2) - 50),
                              font, fontscale, color, thickness)

            y += height / 2 + margin

        (width, height), baseline = cv2.getTextSize("score {}".format(self.score), font, fontscale, thickness)
        img = cv2.putText(img, "score {}".format(self.score),
                          (int(img.shape[1] / 2 - width / 2), int(y - height / 2) - 50),
                          font, fontscale, color, thickness)
        y += height / 2 + margin

        (width, height), baseline = cv2.getTextSize("highscore {}".format(self.highscore), font, fontscale, thickness)
        img = cv2.putText(img, "highscore {}".format(self.highscore),
                          (int(img.shape[1] / 2 - width / 2), int(y - height / 2) - 50),
                          font, fontscale, color, thickness)
        y += height / 2 + margin

        rety, retx, retdy, retdx = self.startscreen.shape[0] * 0.55, self.startscreen.shape[1] * 0.33, 210, 90
        retbox = (int(rety), int(retx), int(rety + retdy), int(retx + retdx))
        endy, endx, enddy, enddx = self.startscreen.shape[0] * 0.50, self.startscreen.shape[1] * 0.45, 270, 90
        endbox = (int(endy), int(endx), int(endy + enddy), int(endx + enddx))
        retrycounter = 0  # counts how many loops the cursor has been inside the "play" bounding box
        endcounter = 0
        imgcopy = cv2.rectangle(img, retbox[0:2], retbox[2:], (255, 0, 255), 1)
        imgcopy = cv2.rectangle(imgcopy, endbox[0:2], endbox[2:], (255, 0, 255), 1)
        replay = False

        while True:
            cursorx, cursory = self._getcursorpos()
            x, y = cursorx * self.playwindow_width, cursory * self.playwindow_height
            img, s = showcursor(imgcopy, x, y, alpha=0.1, counter=retrycounter)
            img, s2 = showcursor(img, x, y, alpha=0.1, counter=endcounter)
            cv2.imshow("Image", img)
            cv2.waitKey(1)
            if insidebox(x, y, retbox):
                retrycounter += 1
            else:
                retrycounter = 0

            if insidebox(x, y, endbox):
                endcounter += 1
            else:
                endcounter = 0

            if retrycounter > 0 and s:
                replay = True
                self.mode = 'startscreen'
                break

            if endcounter > 0 and s2:
                break

        return replay


def main():
    game = GameSuper()
    game.begin()


if __name__ == '__main__':
    main()
