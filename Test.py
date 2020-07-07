import gym
import numpy as np
from numpy import *
from matplotlib import pyplot as plt
import matplotlib.animation as anim
env = gym.make('CartPole-v1')
env.reset()

import numpy as np

class Network():
   
    def __init__(self, inp_nodes, hidden_nodes_1, hidden_nodes_2, out_nodes):
        self.i_n = inp_nodes
        self.h_1 = hidden_nodes_1
        self.h_2 = hidden_nodes_2
        self.o_n = out_nodes
        self.func = [False, False, False, False]

    def SetActFunc(self, function):
        if function=='sigmoid':
            self.func = [True, False, False, False]
        elif function == 'relu':
            self.func = [False, True, False, False]
        elif function == 'lrelu':
            self.func == [False, False, True, False]
        elif function == 'softmax': 
            self.func = [False, False, False, True]
        else:
            print('Error: Mentioned',function, 'is not present in function list. Kindly choose among \'sigmoid\' for Sigmoid output, \'relu\' for ReLU outptu, \'lrelu\' for Leaky ReLU or \'softmax\' for Softmax function output')
       
    def Fit(self, inp):
        self.i_l = np.asarray(inp).reshape(-1, 1)
        weight_ = self.i_w
        out = np.dot(weight_, inp)
        self.h_l_1 = self.Act(out).reshape(-1, 1)
        weight_ = self.h_1
        out = np.dot(weight_, self.h_l_1)
        self.h_l_2 = self.Act(out).reshape(-1, 1)

        weight_ = self.h_2

        out = np.dot(weight_, self.h_l_2)
        
        self.o_l = self.Act(out)[0][0]
        
        if self.o_l > 0.5:
            self.o_l = 1
        else:
            self.o_l = 0
        return self.o_l

    def Act(self, _):
        if self.func[0]:
            #print('Yay !\n')
            return 1/(1 + np.exp(-_))
        elif self.func[1]:
            return np.maximum(0, _)
        elif self.func[2]:
            return  _*self.alpha
        else:
            return np.exp(_)/np.sum(np.exp(_))

    def SetAlpha(self, alpha):
        try:
            if self.func == [False, False, True, False]:
                self.alpha = alpha
        except AttributeError:
            print('Activation function not defined. Set Alpha value after defining activation function')

    def Alpha(self):
        try:
            print(self.alpha)
        except AttributeError:
            print('Alpha Value not found! Only LeakyReLU activation can have alpha value. Change the activation function and then set value of Alpha')

    def SetWeights(self, i_w, h_1, h_2):
        self.i_w = i_w
        self.h_1 = h_1 
        self.h_2 = h_2

   
model = np.load('Chromosomes_3.npy',allow_pickle=True).reshape(1, -1)




print(model)

i_w, h_1, h_2 = model[0, 0:32], model[0, 32:64], model[0, 64:68]
i_w = i_w.reshape(-1, 4)
h_1 = h_1.reshape(-1, 8)
h_2 = h_2.reshape(-1, 4)
nn = Network(4, 4, 2, 1)
nn.SetActFunc('relu')
nn.SetWeights(i_w, h_1, h_2)



observation = env.reset()

def update(alpha, data):
    alpha.set_ydata(np.append(alpha.get_ydata(),data))
    alpha.set_xdata(np.append(alpha.get_ydata(),data))
    plt.draw()
prev_angle = 0
award = 0
angle_error = -99

import time
timer = []
for _ in range(3000):
    start_time = time.time()
    env.render()
    action = nn.Fit(observation)
    nn.SetWeights(i_w, h_1, h_2)
    observation, reward, done, info = env.step(int(action))
    a = time.time()-start_time
    timer.append(a*1000)
    angle_error = max(angle_error, abs(observation[2]))
    award += reward
    if _%300==0:
        print('Fitness Score', award, 'Angle Error', angle_error)
        award = 0
        angle_error = 0
t = linspace(0, 3000, 3000)
plt.plot(t, timer, 'r', label='Time Taken for Execution')
plt.title('Scaled View of Render Time')
plt.xlabel('Iterations')
plt.ylabel('Time(microseconds * 1000) Smaller is Better')
plt.legend(loc='lower right')
plt.show()
env.close()
