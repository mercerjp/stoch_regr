# -*- coding: utf-8 -*-
"""
Created on Sun Dec 13 12:09:27 2020

@author: James
"""
import random
random.seed (0) # For marking purposes

import math


class LogisticRegression: #define the class
    def __init__(self, alpha=0.1): 
            self.alpha = alpha #initialise the stepping hyperparameter
            self.b0 = 0.1 #initialise weight of b0
            self.b1 = 0.1 #initialise weight of b1
            self.b2 = 0.1 #initialise weight of b2
            
        #Calculate probability using the function of logistic regression
    def probability(self, data): #sigma function of logistic regression
        t = float(self.b0 + (self.b1 * data[0]) + (self.b2 * data[1])) #set value of t
        sigma = 1/(1+(math.exp(-t))) #implement t in the equation
        return sigma #return sigma
        #Update weights B0, B1, B2
    def update_weights(self, data_Index):      
        datapoint = self.rand_data[data_Index] #set datapoints index from rand_data
        
        sigma = self.sigmax_list[data_Index] #set index value for each sigma
        
        label = self.rand_labels[data_Index] #set index of the individual label
            

        self.b1 =  self.b1 - self.alpha*(sigma-label)*(sigma)*(1-sigma)*datapoint[0] #calculate weights for every datapoint
                    
        self.b2 = self.b2 - self.alpha *(sigma-label)*(sigma)*(1-sigma)*datapoint[1] #calculate weights for every datapoint
                
        self.b0 = self.b0 - self.alpha *(sigma-label)*(sigma)*(1-sigma)*1 #calculate weights for intercept
                

        return self.b0, self.b1, self.b2
    
    #Let's train the logistic model!
    def train(self, dataset, labels, n = 100): #must find weights to minimise error
    
        #we are only training if the dataset has a maximum of two variables, so return FALSE if otherwise
        for i in range(len(dataset)):
            if len(dataset[i]) != 2:
                return False
        
        #Calculate Mean Squared Error
        def MSE(sigma, label): #set the MSE equation (mean squared error)
            summup = 0 #start from 0 to change value as it is summed
            label = len(sigma)*label
            for i in range(len(sigma)): #iterate through the list of sigma(x) values
                diff = (sigma[i] - label[i]) #for each value calculate error
                squard = diff**2
                summup = summup + squard
                MSE_val = summup/len(sigma) #divide by length
            return MSE_val
        

        #MSE_val = 0 #initialise MSE_val (mean square error)
        MSE_list = [] #initialise list of MSE for each epoch
        self.rand_data = dataset
        self.rand_labels = labels            

        self.n = n
        #Train the data
        for i in range(n): #set number of epochs
            datalabs = list(zip(self.rand_data, self.rand_labels))
            random.shuffle(datalabs)
            self.rand_data, self.rand_labels = zip(*datalabs)
            self.sigmax_list = []
            for j in range(len(self.rand_data)): #for length in range of data
                sigma = self.probability(self.rand_data[j])
                self.sigmax_list.append(sigma)
                self.b0, self.b1, self.b2 = self.update_weights(j)
                
                current_MSE = 0 #initialise current MSE value for epoch
                current_MSE = MSE(self.sigmax_list, self.rand_labels) #get MSE value for current epoch
            MSE_list.append(current_MSE) #add to list
    
        
        return(MSE_list) #return list of MSE
        
                  
    #Classifier for the predicted labels
    def classify(self, dataset): #define classify function
    #we are only training if the dataset has a maximum of two variables, so return FALSE if otherwise
        for i in range(len(dataset)):
            if len(dataset[i]) != 2:
                return False
        probabilities = [] #probabilities list initialise
        #calculate the probabilities of each of the data points
        for i in range(len(dataset)):
            sigma = self.probability(dataset[i]) #weights are already optimised so just need to calculate probability
            probabilities.append(sigma)
        
        label_list = [] #initialise the predicted label list
        #predict the labels
        for i in range(len(probabilities)): #for range in the probability list
            if probabilities[i] <= 0.5: #predict label outputs
                label = 0 #label = 0 is sigma of datapoint is less than 0.5
                label_list.append(label) #add to final list
            else:
                label = 1 #label = 1 is sigma of datapoint is more than 0.5
                label_list.append(label) #append to list
        return label_list

    #Acknowledgements: Thanks to Helen Craven and Kieran Molloy for their help explaining shuffle and help getting the correct updated weights.

            
            
#test
"""
dataset = [[0,0],[1,0],[2,1],[1,2],[3,1],[4,1],[5,2],[3,3],[2,5]]
labels = [0, 0, 0, 0, 0, 1, 1, 1, 1]

lr = LogisticRegression()

MSEList = lr.train(dataset,labels,20)

predictedLabels = lr.classify(dataset)
print(predictedLabels)
printRounded(MSEList)

#output
#[0, 0, 1, 1, 1, 1, 1, 1, 1]
#[0.2433221, 0.2383775, 0.2345934, 0.2300313, 0.2257632, 0.2222386, 0.2179849, 0.2145854, 0.2095505, 0.207312, 0.2048598, 0.2015079, 0.1983592, 0.1952583, 0.1923027, 0.1878779, 0.1864044, 0.184024, 0.1785786, 0.1790818]


dataset = [[0,0],[1,0],[2,1],[1,2],[3,1],[4,1],[5,2],[3,3],[2,5]]
labels = [0, 0, 0, 0, 0, 1, 1, 1, 1]


lr = LogisticRegression()
MSEList = lr.train(dataset, labels, 20)
print(MSEList)

#Should be getting an output similar to this
#[0.2433221, 0.2383775, 0.2345934, 0.2300313, 0.2257632, 0.2222386, 0.2179849, 0.2145854, 0.2095505, 0.207312, 0.2048598, 0.2015079, 0.1983592, 0.1952583, 0.1923027, 0.1878779, 0.1864044, 0.184024, 0.1785786, 0.1790818]
predictedLabels = lr.classify(dataset)
print(predictedLabels)


import matplotlib . pyplot as plt
...
plt. plot ( MSEList )
plt. show ()
"""
def printRounded ( myList ):
    print ("[",end="")
    for i in range (len( myList ) -1):
        print (str( round ( myList [i] ,7)),end=", ")
    print (str( round ( myList [ -1] ,7)),end="]\n")


dataset = [[-3,-3],[5,4],[-1,0.5],[3.7,9.5],[0.8,-3],[4,-5.6],[1.5,3],[0.4,3.5],[2.5,3.4],[2,-1]]
labels = [0,1,0,1,1,1,1,0,0,0]
lr = LogisticRegression(0.015)
MSEList1 = lr.train(dataset,labels)
MSEList2 = lr.train(dataset,labels)
predictedLabels = lr.classify(dataset)
print(predictedLabels)
printRounded(MSEList1 + MSEList2)





