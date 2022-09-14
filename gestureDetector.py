# script to use handdetector in tandem with training a neural network to identify hand guestureConfig

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import handdetector as hd
import time
import cv2
import os
import pandas as pd
from sklearn.model_selection import train_test_split
from normalisation_tools import histogram_equalisation
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import recall_score, precision_score, f1_score


def rotate_landmarks(landmarks, flattened=False):
    '''

    rotates landmarks so landmark 9 and landmark 0 are vertically oriented

    :param landmarks: list of landmarks
    :return: rotated list of landmarks
    '''
    if flattened:
        landmarks = landmarks.reshape([21, 2])
    angle = np.arctan((landmarks[0][0] - landmarks[9][0])/max([1e-5, (landmarks[0][1] - landmarks[9][1])])) # clockwise angle of the hand
    img_ctr = landmarks[0]
    landmarks = [[(img_ctr[0] + np.cos(angle) * (landmarks[i][0] - img_ctr[0]) - np.sin(angle) * (landmarks[i][1] - img_ctr[1])),
                 (img_ctr[1] + np.sin(angle) * (landmarks[i][0] - img_ctr[0]) + np.cos(angle) * (landmarks[i][1] - img_ctr[1]))] for i in range(len(landmarks))]
    if flattened:
        landmarks = np.array(landmarks).flatten()
    return landmarks


class MyDataSet(Dataset):
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def __len__(self):
        return len(self.x)

    def __getitem__(self, index):
        return self.x[index], self.y[index]


class FFnet(nn.Module):

    def __init__(self, lr, input_dims, fc1_dims, fc2_dims, fc3_dims, out_dims, activation = F.relu,
                 weight_decay=1e-3):
        super(FFnet, self).__init__()
        self.fc1 = nn.Linear(input_dims, fc1_dims)
        self.fc2 = nn.Linear(fc1_dims, fc2_dims)
        self.fc3 = nn.Linear(fc2_dims, fc3_dims)
        self.fc4 = nn.Linear(fc3_dims, out_dims)
        self.optimizer = optim.Adam(self.parameters(), lr=lr, weight_decay=weight_decay)
        self.criterion = nn.CrossEntropyLoss()
        self.activation = activation
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.to(self.device)

    def forward(self, x):
        if isinstance(x, np.ndarray):
            x = torch.tensor(x).type(torch.float)
        x = self.activation(self.fc1(x))
        x = self.activation(self.fc2(x))
        x = self.activation(self.fc3(x))
        x = self.activation(self.fc4(x))
        return x

    def predict(self, x):
        logits = self(x)
        if len(x.shape) == 2:
            return np.argmax(logits.clone().detach().tolist(), axis=1)
        else:
            return np.argmax(logits.clone().detach().tolist())

    def fit(self, x, y, max_epochs=10000, _rotate_landmarks=False, plot=False):
        train_losses = []
        valid_losses = []
        x_train, x_valid, y_train, y_valid = train_test_split(x, y, test_size = 0.33, random_state = 42)
        if _rotate_landmarks:
            x_train = np.array([rotate_landmarks(x, flattened=True) for x in x_train])
            x_valid = np.array([rotate_landmarks(x, flattened=True) for x in x_valid])
        x_train = torch.tensor(x_train).type(torch.float)
        x_valid = torch.tensor(x_valid).type(torch.float)
        train_data = MyDataSet(x_train, y_train)
        trainingloader = DataLoader(dataset=train_data, batch_size=32, shuffle=True)
        valid_data = MyDataSet(x_valid, y_valid)
        validloader = DataLoader(dataset=valid_data, batch_size=32, shuffle=True)

        for epoch in range(max_epochs):
            for x_batch, y_batch in trainingloader:
                self.optimizer.zero_grad()  # zero the gradient buffers
                train_output = self(x_batch)
                train_loss = self.criterion(train_output, torch.tensor(y_batch).type(torch.long))
                train_loss.backward()
                self.optimizer.step()  # Does the update
                train_losses.append(train_loss)

            with torch.set_grad_enabled(False):
                for x_batch, y_batch in validloader:
                    valid_output = self(x_batch)
                    valid_loss = self.criterion(valid_output, torch.tensor(y_batch).type(torch.long))
                    valid_losses.append(valid_loss.item())

            print("epoch {} | train loss {} | valid loss {}".format(epoch, train_loss.item(), valid_loss.item()))

        if plot:
            plt.figure()
            plt.suptitle('learning curve')
            plt.set_ylabel('CE loss')
            plt.set_xlabel('training epoch (batch wise)')
            plt.plot(train_losses, label='training loss')
            plt.plot(valid_losses, label='validation loss')
        return


class Recorder:
    def __init__(self, datapath = './guestureConfig/'):

        self.datapath = datapath
        self.datafile = os.path.join(datapath, "trainingdata.csv")
        self.detector = hd.handDetector(nHands=1, smoothness=2)

    def record(self, label, delay = 5, maxT = 30):

        if os.path.exists(self.datafile):
            results = pd.read_csv(self.datafile, index_col=0)
        else:
            results = pd.DataFrame(index=[], columns=range(43), data=None) # one label and 21 landmarks, each with x,y values

        tstart = time.time()
        t0 = time.time()
        cap = cv2.VideoCapture(0)
        while time.time() - tstart < maxT:
            success, img = cap.read()
            #img = histogram_equalisation(img)
            img = self.detector.render(img)
            fingerpos, scale = self.detector.genFingerPos(draw=True)

            if time.time() - tstart > delay and len(fingerpos) > 0:
                landmarks = list(fingerpos[0].values())
                flat_landmarks = [item for sublist in landmarks for item in sublist]
                results.loc[len(results+1)] = [label] + flat_landmarks

            cv2.imshow("Image", img)
            cv2.waitKey(1)

            t1 = time.time()
            fps = 1 / (t1 - t0)
            t0 = t1

            cv2.putText(img, "FPS {}".format(int(fps)), (10, 70), cv2.FONT_HERSHEY_PLAIN, 3,
                        (255, 0, 255), 3)

        results.to_csv(self.datafile)
        return results


class GestureDetector:
    def __init__(self, traindata='./guestureConfig/trainingdata.csv',
                 load_pretrained=True,
                 modelname='guestureDetector3.pt',
                 max_iterations=300):

        all_data = pd.read_csv(traindata, index_col=0)
        data, valid_data = train_test_split(all_data, test_size=0.1, train_size=0.9)

        labels = data.iloc[:, 0]
        inputs = data.iloc[:, 1:]
        valid_labels = valid_data.iloc[:, 0]
        valid_inputs = valid_data.iloc[:, 1:]

        # normalise data from the 0th landmark
        inputs.iloc[:, 0::2] = inputs.iloc[:, 0::2].subtract(inputs.iloc[:, 0], axis=0)
        inputs.iloc[:, 1::2] = inputs.iloc[:, 1::2].subtract(inputs.iloc[:, 1], axis=0)
        valid_inputs.iloc[:, 0::2] = valid_inputs.iloc[:, 0::2].subtract(valid_inputs.iloc[:, 0], axis=0)
        valid_inputs.iloc[:, 1::2] = valid_inputs.iloc[:, 1::2].subtract(valid_inputs.iloc[:, 1], axis=0)

        # min-max norm
        xmins, xmaxs, ymins, ymaxs = inputs.iloc[:, 0::2].min(axis=1), inputs.iloc[:, 0::2].max(axis=1), inputs.iloc[:, 1::2].min(axis=1), inputs.iloc[:, 1::2].max(axis=1)
        inputs.iloc[:, 0::2] = inputs.iloc[:, 0::2].subtract(xmins, axis=0).divide(xmaxs-xmins, axis=0)
        inputs.iloc[:, 1::2] = inputs.iloc[:, 1::2].subtract(ymins, axis=0).divide(ymaxs-ymins, axis=0)
        # do same for validation data
        xmins, xmaxs, ymins, ymaxs = valid_inputs.iloc[:, 0::2].min(axis=1), valid_inputs.iloc[:, 0::2].max(axis=1), valid_inputs.iloc[:, 1::2].min(axis=1), valid_inputs.iloc[:, 1::2].max(axis=1)
        valid_inputs.iloc[:, 0::2] = valid_inputs.iloc[:, 0::2].subtract(xmins, axis=0).divide(xmaxs-xmins, axis=0)
        valid_inputs.iloc[:, 1::2] = valid_inputs.iloc[:, 1::2].subtract(ymins, axis=0).divide(ymaxs-ymins, axis=0)

        nlabels = len(data.iloc[:, 0].unique())
        self.model = FFnet(1e-3, 42, 100, 100, 100, nlabels, weight_decay=1e-5)

        if load_pretrained == False:
            self.model.fit(inputs.to_numpy(), labels.to_numpy(), max_epochs=max_iterations, _rotate_landmarks=True)
        else:
            self.load_model(modelname)

        self.detector = hd.handDetector(smoothness=1)
        self.cap = cv2.VideoCapture(0)

        print("Model Stats:")
        p = pd.DataFrame(columns=['gesture {}'.format(i) for i in range(6)] + ['Macro'], index=['precision', 'recall', 'f1'])
        metrics = [precision_score, recall_score, f1_score]
        preds = self.model.predict(torch.tensor(valid_inputs.values).to(torch.float))
        for i in range(3):
            for _class in range(6):
                y_true = (valid_labels == _class).values
                y_pred = (preds==_class)
                metric = metrics[i](y_true, y_pred)
                p.iloc[i].loc['gesture {}'.format(_class)] = metric
            p.iloc[i].loc['Macro'] = np.sum(p.iloc[i, :-1]/len(p.iloc[i, :-1]))
        print(p)

    def predict(self, show=True, niterations = 10000):
        t0 = time.time()
        prior_bbox = None
        imgs = []
        preds = []
        for iteration in range(niterations):
            success, img = self.cap.read()
            img = self.detector.render(img)
            fingerpos, scale = self.detector.genFingerPos(draw=True)
            imgsize = img.shape

            img[0:int(img.shape[0]/7), 0:int(img.shape[1]/7)] = 0 # black out corner
            prediction = None
            flat_landmarks = None
            if len(fingerpos) > 0:
                landmarks = list(fingerpos[0].values())
                landmarks = rotate_landmarks(landmarks)

                flat_landmarks = np.array([item for sublist in landmarks for item in sublist])
                # normalisation
                flat_landmarks[0::2] += - flat_landmarks[0]
                flat_landmarks[1::2] += - flat_landmarks[1]

                flat_landmarks = (flat_landmarks - min(flat_landmarks))/(max(flat_landmarks)-min(flat_landmarks))
                for i in range(0, len(flat_landmarks), 2):
                    x = int(flat_landmarks[i] * imgsize[1]/8 - imgsize[1]/60)
                    y = int(flat_landmarks[i+1] * imgsize[0]/8 + imgsize[0]/60)

                    cv2.circle(img, (x, y), 5, (0, 255, 0), cv2.FILLED)
                prediction = self.model.predict(flat_landmarks)
                cv2.putText(img, "Guesture #{}".format(prediction), (10, 150), cv2.FONT_HERSHEY_PLAIN, 3,
                            (255, 0, 255), 3)

            if show:
                cv2.imshow("Image", img)
                cv2.waitKey(1)

            imgs.append(flat_landmarks)
            preds.append(prediction)

            t1 = time.time()
            fps = 1 / (t1 - t0)
            t0 = t1

            cv2.putText(img, "FPS {}".format(int(fps)), (10, 70), cv2.FONT_HERSHEY_PLAIN, 3,
                        (255, 0, 255), 3)
        return imgs, preds


    def save_model(self, modelname):
        if modelname[-3:] != '.pt':
            raise NameError("Model Name needs to end in .pt. Currently: {}".format(modelname))
        torch.save(self.model.state_dict(), os.path.join('./guestureConfig/', modelname))


    def load_model(self, modelname):
        x = torch.load(os.path.join('./guestureConfig/', modelname))
        self.model.load_state_dict(x)
        self.model.eval()


if __name__ == "__main__":
    #r = Recorder()
    #r.record(4, maxT=60)
    g = GestureDetector(load_pretrained=True)
    #g.save_model("guestureDetector3.pt")
    g.predict()