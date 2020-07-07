
import gym
import random
import numpy as np
from numpy import *
import math
from time import sleep
from matplotlib import pyplot as plt
pendulum = gym.make('CartPole-v1')
pendulum.reset()

def ConvertWeights(_):
    i_w, h_1, h_2 = _[0, 0:32], _[0, 32:64], _[0, 64:68]
    i_w = i_w.reshape(-1, 4)
    i_w = (i_w - mean(i_w))/std(i_w)
    h_1 = h_1.reshape(-1, 8)
    h_1 = (h_1 - mean(h_1))/std(h_1)
    h_2 = h_2.reshape(-1, 4)
    h_2 = (h_2 - mean(h_2))/std(h_2)
    return i_w, h_1, h_2
        
    
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
        #print(self.i_l)
        weight_ = self.i_w
        #print(weight_)
        out = np.dot(weight_, inp)
        #print(out)
        #out = (out - np.mean(out, 0))/np.std(out)
        self.h_l_1 = self.Act(out).reshape(-1, 1)
        #print(self.h_l)
        weight_ = self.h_1
        #print(weight_)
        out = np.dot(weight_, self.h_l_1)
        #print(out)
        
        self.h_l_2 = self.Act(out).reshape(-1, 1)

        weight_ = self.h_2

        out = np.dot(weight_, self.h_l_2)
        self.func == [False, False, False, True]
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

   
class GA():

    def __init__(self, i_n, h_n_1, h_n_2, o_n):
        self.i_n = i_n
        self.h_n_1 = h_n_1
        self.h_n_2 = h_n_2
        self.o_n = o_n
        
    def GetWeights(self):
       
        #He Weight Initialization
        i_w = np.random.randn(self.h_n_1, self.i_n)*np.sqrt(2/self.i_n)
        h_1 = np.random.randn(self.h_n_2, self.h_n_1)*np.sqrt(2/self.h_n_1)
        h_2 = np.random.randn(self.o_n, self.h_n_2)*np.sqrt(2/self.h_n_2)

        return i_w, h_1, h_2


    def mutate(self, child):
        self.child = child
        i = random.randint(0, child.shape[1]-1)
        j = random.randint(0, child.shape[1]-1)
        k = random.randint(0, child.shape[1]-1)
        mutation = random.randint(0, 10)
        des = random.randint(0, 10)
        if des < 4:
            self.child[0][k] += mutation

        return child
    
        
    #def crossover(self, par_1, par_2):
    def crossover(self, parents):
       new_population = []
       parent_1 = parents[0].reshape(1, -1)
       parent_2 = parents[1].reshape(1, -1)
       for _ in range(population):
           index = random.randint(0, parent_1.shape[1])
           child = np.concatenate((parent_1[..., 0:index], parent_2[..., index:parent_1.shape[1]]), 1)
           mutated_child = self.mutate(child)
           new_population.append(mutated_child)
       return new_population


    def natural_selection(self, score, gene):
        loc_1 = score.index(max(score))
        score.remove(max(score))
        parent_1 = gene[loc_1]
        del gene[loc_1]
        loc_2 = score.index(max(score))
        parent_2 = gene[loc_2]
        parents = [parent_1, parent_2]
        return parents


    def natures_first(self, population, iteration_time):
        population_award = []
        population_gene_pool = []
        for _ in range(population):
            i_w, h_1, h_2 = GAmodel.GetWeights()
            observation = pendulum.reset()
            award =  0
            for __ in range(iteration_time):
                #pendulum.render()
                model = Network(4, 4, 2, 1)
                model.SetWeights(i_w, h_1, h_2)
                action = model.Fit(observation)
                observation, reward, done, info = pendulum.step(int(action))
                award += reward
                if done:
                    break
            population_award.append(award)
            chromosome = np.concatenate((i_w.flatten(), h_1.flatten(), h_2.flatten()))
            population_gene_pool.append(chromosome)
        return population_award, population_gene_pool

generations = 50
population = 35
iteration_time = 400

i_n = 4
h_1 = 8
h_2 = 4
o_n = 1
GAmodel = GA(i_n, h_1, h_2, o_n)
model = Network(4, 8, 4, 1)
model.SetActFunc('relu')
       
pop_award, pop_gene = GAmodel.natures_first(population, iteration_time)

best_awards_gen = []
min_awards_gen = []
med_awards_gen = []
it = []
PID = []
prev = 0
prev_pos = 0

avg = -99
current_award = 0
generation_awards = []
i = 1


for gen in range(generations):
    parents = GAmodel.natural_selection(pop_award, pop_gene)

    new_population = GAmodel.crossover(parents)
    pop_award = []
    pop_gene = []
    j = 0
    for _ in new_population:
        observation = pendulum.reset()
        input_weight, hidden_1, hidden_2 = ConvertWeights(_)
        model.SetWeights(input_weight, hidden_1, hidden_2)

        award = 0
        for x in range(iteration_time):
            #pendulum.render()
            action = model.Fit(observation)
            observation, reward, done, info = pendulum.step(int(action))
            award += reward
            if done:
                break
        PID.append(award)
        pop_award.append(award)
       
        chromosome = np.concatenate((input_weight.flatten(), hidden_1.flatten(), hidden_2.flatten()))
        pop_gene.append(chromosome)

    it = np.append(it, np.average(PID))
    best_awards_gen = np.append(best_awards_gen, np.amax(pop_award)) #Store Maximum of Each Generation
    min_awards_gen = np.append(min_awards_gen, np.amin(PID))
    med_awards_gen = np.append(med_awards_gen, np.median(PID))
    
    if np.median(pop_award) >= current_award:
        current_award = max(current_award, np.median(PID))
        np.save('Check_new_I',pop_gene[pop_award.index(max(pop_award))])

    print('[Generation: %3d] [Max Score:%5d]  [Median Score:%5d]' %(gen, round(current_award, 2), round(np.median(PID),2)))
    i+=1


pendulum.close()
#t = linspace(0, generations, generations)
t = linspace(0, i-1, i-1)
plt.plot(t, best_awards_gen, 'r', label='Best Fitness Scores')
plt.plot(t, it, 'g', label='Average Fitness Scores')
plt.plot(t, med_awards_gen, 'b', label='Median Fitness Scores')

plt.title('Fitness vs Generation Plot')
plt.xlabel('Generations')
plt.ylabel('Fitness Score')
plt.grid()  
print('Average', np.mean(best_awards_gen), 0)
plt.legend(loc='lower right')
plt.show()

exec(open('Test.py').read())


