# -*- coding: utf-8 -*-
# We will randomly define initial values for connection weights, and also randomly select
#   which training data that we will use for a given run.
import random
from random import randint

# We want to use the exp function (e to the x); it's part of our transfer function definition
from math import exp

# Biting the bullet and starting to use NumPy for arrays
import numpy as np

# So we can make a separate list from an initial one
import copy

#for saving output
import pandas as pd
from pandas import DataFrame

import matplotlib.pyplot as plt


import sys
if sys.path.count("../MLP") == 0:
    sys.path.append('../MLP')

#sys.path.append('/Users/markh/Documents/Northwestern/P490-DeepLearning/MLP')

import datafile
import mlp
import cnn

import alphabet_datasets_MarkH as alph

#define constants
MAX_ITERATIONS = 600000
MAX_EPOCH = 3
#MAX_ITERATIONS = 2
#MAX_ITERATIONS = 10
MAXINT = 9223372036854775807


#clears the console
def clear():
    sys.stderr.write("\x1b[2J\x1b[H")
    
addNoise = False

# For pretty-printing the arrays
np.set_printoptions(precision=3)
np.set_printoptions(suppress=True) 

####################################################################################################
####################################################################################################
#
# This is a tutorial program, designed for those who are learning Python, and specifically using 
#   Python for neural networks applications
#
#
####################################################################################################
####################################################################################################
#
# Code Map: List of Procedures / Functions
# - welcome
#
# == set of basic functions ==
# - computeTransferFnctn
# - computeTransferFnctnDeriv
# - matrixDotProduct
#
# == identify crucial parameters (these can be changed by the user)
#    obtainNeuralNetworkSizeSpecs
#
#    -- initializeWeight
# - initializeWeightArray
# - initializeBiasWeightArray
#
# == obtain the training data (two possible routes; user selection & random)
# - obtainSelectedAlphabetTrainingValues
# - obtainRandomAlphabetTrainingValues
#
# == the feedforward modules
#   -- ComputeSingleFeedforwardPassFirstStep
#   -- ComputeSingleFeedforwardPassSecondStep
# - ComputeOutputsAcrossAllTrainingData
#
# == the backpropagation training modules
# - backpropagateOutputToHidden
# - backpropagateBiasOutputWeights
# - backpropagateHiddenToInput
# - backpropagateBiasHiddenWeights
# - main




####################################################################################################
####################################################################################################
#
# Procedure to welcome the user and identify the code
#
####################################################################################################
####################################################################################################


def welcome ():


    print
    print '******************************************************************************'
    print
    print 'Welcome to the Multilayer Perceptron Neural Network'
    print '  trained using the backpropagation method.'
    print 'Version 0.4, 03/05/2017, A.J. Maren'
    print 'For comments, questions, or bug-fixes, contact: alianna.maren@northwestern.edu'
    print ' ' 
    print 'This program learns to distinguish between broad classes of capital letters'
    print 'It allows users to examine the hidden weights to identify learned features'
    print
    print '******************************************************************************'
    print
    return()


####################################################################################################
####################################################################################################
#
# Function to obtain the neural network size specifications
#
####################################################################################################
####################################################################################################

def obtainNeuralNetworkSizeSpecs (dataset, filterMasks, nHidden=6, bOutputByClass = True):

# This procedure operates as a function, as it returns a single value (which really is a list of 
#    three values). It is called directly from 'main.'
#        
# This procedure allows the user to specify the size of the input (I), hidden (H), 
#    and output (O) layers.  
# These values will be stored in a list, the arraySizeList. 
# This list will be used to specify the sizes of two different weight arrays:
#   - wWeights; the Input-to-Hidden array, and
#   - vWeights; the Hidden-to-Output array. 
# However, even though we're calling this procedure, we will still hard-code the array sizes for now.   

    #bOutputClass defines the number of outputs - by class letter or letter itself
    dd = pd.DataFrame(data=dataset)
    
    #number of inputs
    #assume all inputs are the same length
    dIn = dataset[0][1]
    numIn = len(dIn)

    #number of greybox outputs are in column 4 (letter classes)
    gbOut = dd[4]
    numGbOut = max(gbOut)+1
    
    #number of cnn outputs are in column 2 (letters)
    cnnOut = dd[2]
    numCnnOut = max(cnnOut)+1
    
    #num masks is used to determine # of inputs to cnn
    numMasks = len(filterMasks.masks)
    
    #count the number of distinct outputs
    #numInputNodes = 333
    #numHiddenNodes = nHidden
    #numOutputNodes = 27 
    
    numInputNodes = numMasks * numIn + numGbOut
    numHiddenNodes = nHidden
    numOutputNodes = numCnnOut
    
    print ' '
    print '  The number of nodes at each level are:'
    print '    Input: 9x9 (square array)'
    print '    Hidden: ', numHiddenNodes
    print '    Output: ', numOutputNodes
            
# We create a list containing the crucial SIZES for the connection weight arrays    

    #we always are outputting by letter now (letter = 2, class = 4)
    arraySizeList = (numInputNodes, numHiddenNodes, numOutputNodes, 2)
    
# We return this list to the calling procedure, 'main'.       
    return (arraySizeList)  


#builds a sorted list of keys in the dict
#keys repeat for every item in dict
def LettersInDict(SSE_Dict):
    letters=[]
    keys = np.sort(SSE_Dict.keys())
    for k in keys:
        dd = SSE_Dict[k]
        numLetters = len(dd.keys())
        for i in range(numLetters):
            letters.append(k)
            
    return letters

#builds a heatmap using the specified key in the dictionary
#also build row labels for heatmap
def BuildMap(SSE_Dict, key):
    
    letters=[]
    heatmap=[]
    
    keys = np.sort(SSE_Dict.keys())
    for k in keys:
        dd = SSE_Dict[k]
        indexes = np.sort(dd.keys())
        for i in indexes:
            letters.append(k)
            
            row = dd[i][key]
            heatmap.append(row)
    
    return(letters,heatmap)

#builds a list of MSE values by letter
def MSE(SSE_Dict):
    
    letters=[]
    mse=[]

    keys = np.sort(SSE_Dict.keys())
    for k in keys:
        dd = SSE_Dict[k]
        indexes = np.sort(dd.keys())
        sse = 0
        for i in indexes:
            sse = sse + dd[i]['sse']
            
        sse = sse/len(indexes)
        mse.append(sse)
        letters.append(k)
    
    return(letters,mse)
    
#can't successfully write letters
def lettersToNum(letters):
    nums = [ord(l)-ord('A') for l in letters]
    return nums
    
def numToLetters(nums):
    letters = [chr(int(n)+ord('A')) for n in nums]
    return letters
    
    

#load a mlp from data set to run a set of test inputs through
def ComputeAll (mm, dataset, filterMasks, outputCol, debug = False, bHidden=False):

    nHidden = mm.nHidden
    nOut = mm.nOutput
    
    #get a count of the number of letters being tested
    #a letter may have multiple variants
    dd = pd.DataFrame(dataset)
    letters = dd[3]
    freq = pd.value_counts(letters)
    llist = list(freq.index)
    
    #initialize dictionary
    #using a dictionary because test set may not have all letters
    SSE_Dict = {}
    for ll in np.sort(llist):
        SSE_Dict[ll]={}
        
    totalSSE = 0
    numTrainingSets = len(dataset)
        
    #loop through each training set
    for iTest in range(numTrainingSets):
        
        #test sample
        trainingDataList = dataset[iTest]
        
        #input values
        trainingDataInputList = trainingDataList[1]      
        inputDataArray = np.array(trainingDataInputList)

        letterNum = trainingDataList[2]
        letterChar = trainingDataList[3]
        letterClass = trainingDataList[4]
        
        outputArrayLength = mm.nOutput
        desiredClass = trainingDataList[outputCol]                 # identify the desired class
        desiredOutputArray = np.zeros(outputArrayLength)    # iniitalize the output array with 0's
        desiredOutputArray[desiredClass] = 1                # set the desired output for that class to 1
        
        #create filtered input data
        filteredInputDataArray = filterMasks.FilterLetter(inputDataArray)
        
        #use mlp class         
        outputArray = mm.forward(inputDataArray, filteredInputDataArray)
        
        #calculate SSE
        errorArray = (desiredOutputArray-outputArray)**2
        newSSE = sum(errorArray)
        totalSSE = totalSSE + newSSE
        
        #store statistics for the value
        SSE_Dict[letterChar][iTest] = {}
        SSE_Dict[letterChar][iTest]['index']=iTest
        SSE_Dict[letterChar][iTest]['hidden'] = list(mm.Hidden())
        SSE_Dict[letterChar][iTest]['out'] = list(mm.Output())
        SSE_Dict[letterChar][iTest]['sse'] = newSSE
        
        
    if(bHidden==True):
        letters,heatmap = BuildMap(SSE_Dict,'hidden')
        f = datafile.DataFile("heatmap_%d.csv" % nHidden)
        f.add(heatmap)
        f.write()

        f = datafile.DataFile("heatmap_%d_letters.csv" % nHidden)
        f.add(letters)
        f.write()

        f = datafile.DataFile("heatmap_%d_letnums.csv" % nHidden)
        f.add(lettersToNum(letters))
        f.write()
        
        letters,heatmap = BuildMap(SSE_Dict,'out')
        f = datafile.DataFile("outmap_%d.csv" % nHidden)
        f.add(heatmap)
        f.write()

        f = datafile.DataFile("outmap_%d_letters.csv" % nHidden)
        f.add(letters)
        f.write()

        f = datafile.DataFile("outmap_%d_letnums.csv" % nHidden)
        f.add(lettersToNum(letters))
        f.write()
    
    #return the errors
    return(SSE_Dict)        


#get list of input letters in the data set
def Classes(dataset):
    dd = pd.DataFrame(dataset)
    iclass = dd[4]
    lclass = dd[5]
    
    ilist = iclass.value_counts()
    rticks = range(len(ilist))
    ticks = [list(dd.loc[dd[4]==x][5])[0] for x in rticks]

    return ticks    
    
def Letters(dataset):
    dd = pd.DataFrame(dataset)
    iletter = dd[2]
    lletter = dd[3]
    
    ilist = iletter.value_counts()
    rticks = range(len(ilist))
    ticks = [list(dd.loc[dd[2]==(x)][3])[0] for x in rticks]
 
    return ticks    
    
#plots SSEs of the test set
def plot_test_error(SSE_Dict, nHidden):
    
    ticks, yvals = MSE(SSE_Dict)
    
    plt.clf()
    plt.title('Test MSE By Letter, %d Hidden Nodes' % nHidden)
    plt.ylabel('SSE')
    plt.xlabel('Letter')
    plt.xticks(range(len(ticks)),ticks)
    
    plt.margins(0.2)
    y_pos = np.arange(len(ticks))
    plt.bar(y_pos,yvals,align='center')
    plt.show()    
    plt.savefig('test_msebyletter_%d.png' % nHidden)
       
def plot_heatmap(filename, title):
    
#    ticks = Letters(dataset)
#    rticks = range(len(ticks))
    
    # Plot heatmap (this works)
    f = datafile.DataFile('%s.csv' % filename)
    h = f.read()
    xlen = h.shape[1]
    
    f = datafile.DataFile('%s_letnums.csv' % filename)
    ticks = numToLetters(f.read())
    rticks = range(len(ticks))

    plt.clf()
    plt.title(title)
    plt.ylabel('Letter Class')
    plt.xlabel('Node')
    plt.yticks(rticks,ticks)
    plt.xticks(range(xlen),range(xlen))
    plt.margins(0.2)
    plt.imshow(h,cmap="bwr")
    plt.show()    
    plt.savefig('%s.png' % filename)
            
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                           
####################################################################################################
####################################################################################################
#
# Procedure to print out a letter, given the number of the letter code
#
####################################################################################################
####################################################################################################

def printLetter (trainingDataList):    

    print ' '
    print ' in procedure printLetter'
    print ' '                         
    print 'The training data set is ', trainingDataList[0]
    print 'The data set is for the letter', trainingDataList[3], ', which is alphabet number ', trainingDataList[2]

    if trainingDataList[0] > 25: print 'This is a variant pattern for letter ', trainingDataList[3] 

    pixelArray = trainingDataList[1]
                
    iterAcrossRow = 0
    iterOverAllRows = 0
    while iterOverAllRows <gridHeight:
        while iterAcrossRow < gridWidth:
#            arrayElement = pixelArray [iterAcrossRow+iterOverAllRows*gridWidth]
#            if arrayElement <0.9: printElement = ' '
#            else: printElement = 'X'
#            print printElement, 
            iterAcrossRow = iterAcrossRow+1
#        print ' '
        iterOverAllRows = iterOverAllRows + 1
        iterAcrossRow = 0 #re-initialize so the row-print can begin again
    
    return         
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                 
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                          
####################################################################################################
#**************************************************************************************************#
####################################################################################################    
        
            
                    
            
####################################################################################################
#**************************************************************************************************#
####################################################################################################
#
# The MAIN module comprising of calls to:
#   (1) Welcome
#   (2) Obtain neural network size specifications for a three-layer network consisting of:
#       - Input layer
#       - Hidden layer
#       - Output layer (all the sizes are currently hard-coded to two nodes per layer right now)
#   (3) Initialize connection weight values
#       - w: Input-to-Hidden nodes
#       - v: Hidden-to-Output nodes
#   (4) Compute a feedforward pass in two steps
#       - Randomly select a single training data set
#       - Input-to-Hidden
#       - Hidden-to-Output
#       - Compute the error array
#       - Compute the new Summed Squared Error (SSE)
#   (5) Perform a single backpropagation training pass
#
####################################################################################################
#**************************************************************************************************#
####################################################################################################


#def main():
def run_test(dataset, filterMasks, gb_base, tTransferH, tTransferO, eta, seed, nHidden=6, bOutputByClass = True, bHidden=False, bWeights=False, pDropout=0):

# Define the global variables        
    global inputArrayLength
    global hiddenArrayLength
    global outputArrayLength
    global gridWidth
    global gridHeight
    global eGH # expandedGridHeight, defined in function expandLetterBoundaries 
    global eGW # expandedGridWidth defined in function expandLetterBoundaries 
    global mask1
    
    #noise factor
    gamma = 0.0

####################################################################################################
# Obtain unit array size in terms of array_length (M) and layers (N)
####################################################################################################                

# This calls the procedure 'welcome,' which just prints out a welcoming message. 
# All procedures need an argument list. 
# This procedure has a list, but it is an empty list; welcome().

    welcome()

    
# Right now, for simplicity, we're going to hard-code the numbers of layers that we have in our 
#   multilayer Perceptron (MLP) neural network. 
# We will have an input layer (I), an output layer (O), and a single hidden layer (H). 

# Define the variable arraySizeList, which is a list. It is initially an empty list. 
# Its purpose is to store the size of the array.

    arraySizeList = list() # empty list

# Obtain the actual sizes for each layer of the network       
    arraySizeList = obtainNeuralNetworkSizeSpecs (dataset, filterMasks, nHidden, bOutputByClass)
    
# Unpack the list; ascribe the various elements of the list to the sizes of different network layers
# Note: A word on Python encoding ... the actually length of the array, in each of these three cases, 
#       will be xArrayLength. For example, the inputArrayLength for the 9x9 pixel array is 81. 
#       These values are passed to various procedures. They start filling in actual array values,
#       where the array values start their count at element 0. However, when filling them in using a
#       "for node in range[limit]" statement, the "for" loop fills from 0 up to limit-1. Thus, the
#       original xArrayLength size is preserved.   
    inputArrayLength = arraySizeList [0] 
    hiddenArrayLength = arraySizeList [1] 
    outputArrayLength = arraySizeList [2]
    outputClassColumn = arraySizeList[3]
    
    print ' '
    print ' inputArrayLength = ', inputArrayLength
    print ' hiddenArrayLength = ', hiddenArrayLength
    print ' outputArrayLength = ', outputArrayLength        
    print ' outputClassColumn = ', outputClassColumn        

    #load greybox mlp from previous weeks
    #todo: number of hidden nodes for both gb and cnn
    #gb = mlp.mlp.read('from_week4/weights_%d' % nHidden)
    gb = mlp.mlp.read(gb_base)
    
    #init new cnn
    mm = cnn.cnn.init(gb,inputArrayLength, hiddenArrayLength, tTransferH, outputArrayLength, tTransferO, eta)
    
# Parameter definitions for backpropagation, to be replaced with user inputs
    #alpha = 1.0
    #eta = 0.5    
    maxNumIterations = MAX_ITERATIONS    # temporarily set to 10 for testing
    epsilon = 0.01
    iteration = 0
    SSE = 0.0
    numTrainingDataSets = len(dataset)
    dd = pd.DataFrame(dataset)
    
####################################################################################################
# Initialize the weight arrays for two sets of weights; w: input-to-hidden, and v: hidden-to-output
####################################################################################################                

    #randomize initial weights
    mm.randomize(seed)
        
          
####################################################################################################
# Before we start training, get a baseline set of outputs, errors, and SSE 
####################################################################################################                
                            
    print ' '
    print '  Before training:'
    
    mm.SetDropout(0)
    SSE_Dict = ComputeAll(mm, dataset, filterMasks, outputClassColumn, bHidden=False)                           
                                             
          
####################################################################################################
# Next step - Obtain a single set of randomly-selected training values for alpha-classification 
####################################################################################################                
  
  
    while iteration < maxNumIterations:           

# Increment the iteration count
        iteration = iteration +1

####################################################################################################
# While training - STEP 1: Obtain a set of training data; inputs and desired outputs
####################################################################################################     
            
# Randomly select one of 26 traing sets. number returned is 0 <= n <= numTrainingDataSets-1
        index = random.randint(0, numTrainingDataSets-1)
        
# We return the list from the function, with values placed inside the list.           
#        print ' in main, about to call obtainSelected'
        trainingDataList = dataset[index]
          
# Optional print/debug
#        printLetter(trainingDataList)        
          
                                                                                                                                                    
####################################################################################################
# While training - STEP 2: Create an input array based on the input training data list
####################################################################################################     

        
# The trainning inputs are drawn from the first element (starting count at 0) in the training data list

        thisTrainingDataList = trainingDataList[1]
        inputDataArray = np.array(thisTrainingDataList)
        
        #add random noise to input data
        inputDataLen = len(inputDataArray)
        
        #need some un-noisy samples?
        noise = np.zeros(inputDataLen)
        #if random.random() < .5:
        #    noise = np.random.random(inputDataLen)<gamma
        #else:
        #    noise = np.zeros(inputDataLen)
        noisyInputDataArray = abs(inputDataArray-noise)
            
        
# The desired outputs are drawn from the fourth element (starting count at 0) in the training data list
#   This represents the "big shape class" which we are training towards in GB1

        desiredOutputArray = np.zeros(outputArrayLength)    # iniitalize the output array with 0's
        desiredClass = trainingDataList[outputClassColumn]  # identify the desired class
        desiredOutputArray[desiredClass] = 1                # set the desired output for that class to 1

          
####################################################################################################
# Compute a single feed-forward pass and obtain the Actual Outputs
####################################################################################################                

        #run input through filter
        filteredInputDataArray = filterMasks.FilterLetter(noisyInputDataArray)
                                        
        ## use mlp class
        outputArray = mm.forward(noisyInputDataArray, filteredInputDataArray)
    
####################################################################################################
# Perform backpropagation
####################################################################################################                
                
        ### use mlp class
        mm.backprop(filteredInputDataArray,desiredOutputArray)
    
        # Compute a forward pass, test the new SSE                                                                                
        # outputArray = mm.forward(noisyInputDataArray, filteredInputDataArray)
        
        #not keeping track of all outputs any more
        #keeping average SSE across all letter variants
        #so do a ComputeAll every 100 iterations
    
        # Determine the error between actual and desired outputs
        maxSSE = epsilon+1
        if iteration % 1000 == 0:
            
            #things are really slow - make sure it is still working
            print 'iteration = %d' % iteration
            
            mm.SetDropout(0)
            SSE_Dict = ComputeAll(mm, dataset, filterMasks, outputClassColumn, bHidden=False)                           
            letters,mse = MSE(SSE_Dict)
            maxSSE = max(mse)
        
        #break out when worst SSE is less than threshold
        if maxSSE < epsilon:
            break
            
    print 'Out of while loop at iteration ', iteration 
    
####################################################################################################
# After training, get a new comparative set of outputs, errors, and SSE 
####################################################################################################                           

    print ' '
    print '  After training:'   
    
                                                      
    mm.SetDropout(0)
    SSE_Dict = ComputeAll (mm, dataset, filterMasks, outputClassColumn, debug = True, bHidden=True)
    letters,mse = MSE(SSE_Dict)
    totalMSE = sum(mse)/len(mse)                     

    #save weight array
    if( bWeights == True ):
        
        mm.write_weights('weights_%d' % nHidden)

    #MSE and iterations by epoch    
    if( bHidden == True):
        f = datafile.DataFile("sse_%d.csv" % nHidden )
        f.add(mse)
        f.write()

        f = datafile.DataFile("sse_%d_letters.csv" % nHidden )
        f.add(lettersToNum(letters))
        f.write()
    
    print( 'iterations = %d' % iteration )
    print( 'MSE = %.6f' % totalMSE )

    return(iteration,totalMSE)

                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                  
                                              
####################################################################################################
# Conclude specification of the MAIN procedure
####################################################################################################                

##############################################
# scans through combinations of alpha and eta
###############################################
    
#note: this just scans through eta now

def search_parms(dataset, filterMasks, gb_base, tTransferH, tTransferO, nHidden):

    best_sse = 1e6
    best_iterations = 1e6
    alpha = 1.7
	
    for epoch in range(5):
        #random seed for each run - not sure this is valid, but it takes forever otherwise
        seed = random.randint(1,MAXINT)
        
        print 'epoch: %d' % epoch
   	
   	#just 1 alpha loop
        for alpha in range(1,2):
            for eta in range(20):
 		
                e = 0.1 + eta/50.0
                a = alpha *2.0
                print 'alpha = %f' % a
                print 'eta = %f' % e
                rc = run_test(dataset, filterMasks, gb_base, tTransferH, tTransferO, eta=e, seed=seed, nHidden=nHidden, bOutputByClass = True, bHidden=True, bWeights=True)
                iterations = rc[0]
                sse = rc[1]
                if iterations < best_iterations:
                    best_seed = seed
                    best_sse = sse
                    best_alpha = a
                    best_eta = e
                    best_iterations = iterations
				
    print '*******'
    print( 'best seed = %d' % best_seed )
    print( 'best alpha = %.4f' % best_alpha )
    print( 'best eta = %.4f' % best_eta )
    print( 'best MSE = %.4f' % best_sse )
    print( 'best_iterations = %d' % best_iterations )
    print '*******'
        
### run a test using the specified number of hidden nodes
def test_hidden(dataset, filterMasks, gb_base, tTransferH, tTransferO, eta, nHidden, bOutputByClass=True, bPlot = True, pDropout = 0):

    #just keep running tests with random seeds
    #keep track of which had the lowest iteration count
    epoch = 0  
    best_iterations = 1e6
    best_seed = 1e6
    best_sse = 1e6
    best_epoch = 0
    
    #start a data file
    #this is overall mse and iterations
    f = datafile.DataFile("training_%d.csv" % nHidden)
        
    while epoch < MAX_EPOCH:
        seed = random.randint(1,MAXINT)
        print '###################'
        print 'epoch = %d' % epoch
        print 'seed = %d' % seed
        print '###################'
        
        #use previously found best alpha and eta
        rc = run_test(dataset, filterMasks, gb_base, tTransferH, tTransferO, eta,seed=seed,nHidden=nHidden, bOutputByClass=bOutputByClass, bHidden=True, bWeights=True, pDropout=pDropout)
        iterations = rc[0]
        sse = rc[1]
#        if iterations < best_iterations:
        if sse < best_sse:
            best_iterations = iterations
            best_sse = sse
            best_seed = seed
            best_epoch = epoch
            
        #print after every loop
        print '*******'
        print( 'best iterations = %d' % best_iterations )
        print( 'seed = %d' % best_seed )
        print( 'MSE = %.4f' % best_sse )
        print '*******'
        
        row = np.array([iterations, sse])
        f.add(row)
        
        #keep running average of sse
        fsseavg = datafile.DataFile("sse_avg_%d.csv" % nHidden)
        fsse = datafile.DataFile("sse_%d.csv" % nHidden)
        fsse.read()
        if epoch==0:
            #don't include total mse
            fsseavg.add(fsse.array())
        else:
            fsseavg.read()
            a1 = fsseavg.array()
            a2 = fsse.array()
            aa = a1+a2
            #dont include total mse
            fsseavg.clear()
            fsseavg.add(aa)
            
        fsseavg.write()
            
        #next loop
        epoch = epoch + 1
        
    #save the training SSE results 
    f.write()

    #averages
    fsseavg = datafile.DataFile("sse_avg_%d.csv" % nHidden)
    fsseavg.read()
    a_sse = fsseavg.array()
    a_sse = [x/epoch for x in a_sse]
    fsseavg.clear()
    fsseavg.add(a_sse)
    fsseavg.write()
    
    #create plots
    
    if( bPlot == True):
        #iterations by runs
        df = DataFrame(f.array())
        plt.figure(1)
        plt.subplot(211)
        plt.plot(df[0],'b-')
        plt.plot(best_epoch, best_iterations, color='black', marker='.', markersize=10)
        plt.xlabel('epoch')
        plt.ylabel('Iterations')
        plt.title('Number of Iterations')
        
        #mse by runs
        plt.subplot(212)
        plt.plot(df[1],'r-')
        plt.plot(best_epoch, best_sse, color='black', marker='.', markersize=10)
        plt.xlabel('epoch')
        plt.ylabel('MSE')
        plt.title('MSE')
        
        plt.subplots_adjust(hspace=.5, wspace=0.35)
        plt.show()
        plt.savefig('training_%d.png' % nHidden)
        
        #average sse by letter
        plt.clf()
        plt.title('Average SSE By Letter, %d Hidden Nodes, %d Runs' % (nHidden,epoch))
        plt.ylabel('SSE')
        plt.xlabel('Letter')
        #ticks = [x+ord('A') for x in range(26)]
        #ticks = [chr(x) for x in ticks]
        #plt.xticks(range(26),ticks)
        ticks = Letters(dataset)
        rticks = range(len(ticks))
        plt.xticks(rticks,ticks)
        plt.margins(0.2)
        #plt.plot(sse)
        y_pos = np.arange(26)
        #last item is mse, don't plot it
        plt.bar(y_pos,a_sse,align='center')
        plt.show()    
        plt.savefig('sseavg_%d.png' % nHidden)
    
#idea for plot - number of iterations to converge? maybe they all will?
#find quickest convergence

    #rerun using best_seed, save the hidden outputs
    rc = run_test(dataset, filterMasks, gb_base, tTransferH, tTransferO, eta,seed=best_seed,nHidden=nHidden, bOutputByClass=bOutputByClass, bHidden=True, bWeights=True, pDropout=pDropout)
    #rc = run_test(dataset, tTransferH, alpha, eta=1.5,seed=seed,nHidden=nHidden, bOutputByClass = True, bHidden=True, bWeights=True)
    iterations = rc[0]
    sse = rc[1]
    print '*******'
    print( 'best iterations = %d' % iterations )
    print( 'seed = %d' % best_seed )
    print( 'mse = %.4f' % best_sse )
    print '*******'
    
    if bPlot == True:
        # Plot heatmap (this works)
        filename = "heatmap_%d" % nHidden
        plot_heatmap(filename, "Hidden Node Activations" );
        filename = "outmap_%d" % nHidden
        plot_heatmap(filename, "Output Node Activations" );
        
        #plot SSE
        fsse = datafile.DataFile("sse_%d.csv" % nHidden)
        sse = fsse.read()
        f = datafile.DataFile("sse_%d_letters.csv" % nHidden)
        ticks = numToLetters(f.read())
        rticks = range(len(ticks))
        
        plt.clf()
        plt.title('MSE By Letter, %d Hidden Nodes' % nHidden)
        plt.ylabel('MSE')
        plt.xlabel('Letter')
    
        plt.xticks(rticks,ticks)
        #ticks = [chr(x+ord('A')) for x in range(26)]
        #plt.xticks(range(26),ticks)
        
        plt.margins(0.2)
        #plt.plot(sse)
        y_pos = np.arange(len(ticks))
        #last item is mse, don't plot it
        plt.bar(y_pos,sse,align='center')
        plt.show()    
        plt.savefig('ssebyletter_%d.png' % nHidden)
        
    #return iterations and mse
    return rc
        
    
######################################################

#does a forward pass of all of the values in dataset
#directory is the subdirectoy containing the saved weights of the MLP
#nHidden is used as part of the base filename of the saved weights
def test_nn(dataset, filterMasks, gb_base, cnn_base,nHidden):
    
    gb = mlp.mlp.read(gb_base)
    mm = cnn.cnn.read(gb,cnn_base)
    SSE_Dict = ComputeAll(mm, dataset, filterMasks, 2, bHidden=True, debug=True)   
    plot_test_error(SSE_Dict, nHidden)       
    plot_heatmap("heatmap_%d" % nHidden, "Hidden Node Activations")
    plot_heatmap("outmap_%d" % nHidden, "Output Node Activations")
                        
#######################################                      

    
### use cross validation to find # of hidden nodes ###

def cv(dataset, filterMasks, gb_base, tHidden, tOut, eta, rHidden):

    NUM_FOLDS=10
    #NUM_FOLDS=2
    
    N = len(dataset)
    indexes = np.array(range(N))
    folds = np.array([random.randint(0,NUM_FOLDS-1) for x in range(N)])
    
    f = datafile.DataFile("cross.csv")

    #loop through each hidden value size
    for nHidden in rHidden:
        
        #loop through folds (10-fold cross validation)
        mse = np.zeros(NUM_FOLDS)
        for i in range(NUM_FOLDS):
            
            print ('*** Hidden = %d' % nHidden)
            print ('*** fold = %d' % i)
            
            iTrain = np.array(indexes[folds!=i])
            iTest = np.array(indexes[folds==i])
            
            #build training and test data sets
            train = [dataset[ii] for ii in iTrain]
            test = [dataset[ii] for ii in iTest]
            
            #train a network
            test_hidden(train, filterMasks, gb_base, tHidden, tOut, eta, nHidden, bPlot = False, bOutputByClass=False, pDropout=0)
            
            #now test it
            fbase = 'weights_%d' % nHidden
            
            gb = mlp.mlp.read(gb_base)
            mm = cnn.cnn.read(gb,fbase)
            SSE_Dict = ComputeAll(mm, dataset, filterMasks, 2, bHidden=True, debug=False)   
            letters,sseArray = MSE(SSE_Dict)
            thisMse = sum(sseArray)/len(sseArray)
            mse[i] = thisMse
            
        
        #get the best mse of the 10 and save it
        foldMse = sum(mse)/len(mse)
        row = np.array([nHidden, foldMse])
        f.add(row)
        
    f.write()
    mseArray = f.array()

    bestMse, bestIdx = min((val[1],idx) for (idx, val) in enumerate(mseArray))
    bestHidden = mseArray[bestIdx][0]
    
    #plot it
    plt.clf()
    plt.title('Hidden Node Cross Validation')
    plt.ylabel('MSE')
    plt.xlabel('Hidden Nodes')

    plt.xticks(rHidden,rHidden)
    plt.margins(0.2)
        
    plt.plot(mseArray[:,0], mseArray[:,1],'r-')
    #plt.plot(best_epoch, best_sse, color='black', marker='.', markersize=10)
    plt.show()
    plt.savefig('cross.png')
    
    print('************')
    print('Best MSE = %f' % bestMse)
    print('Best Hidden nodes = %d' % bestHidden)
    print('************')
    
    return(bestHidden,bestMse)

###################################################        
                  
def main():

    #masks
    mask4 = [    
        (0,1,0, 0,1,0, 0,1,0),
        (0,0,0, 1,1,1, 0,0,0),       
        (1,0,0, 0,1,0, 0,0,1),
        (0,0,1 ,0,1,0, 1,0,0) 
        ]        

    mask8 = [    
        (0,1,0, 0,1,0, 0,1,0),
        (0,0,0, 1,1,1, 0,0,0),       
        (1,0,0, 0,1,0, 0,0,1),
        (0,0,1 ,0,1,0, 1,0,0), 

        #add corners        
        (1,1,1, 1,0,0, 1,0,0),
        (1,1,1, 0,0,1, 0,0,1),
        (1,0,0, 1,0,0, 1,1,1),
        (0,0,1, 0,0,1, 1,1,1),
        
        #add points
        (0,0,0, 0,1,0, 1,0,1),
        (1,0,1, 0,1,0, 0,0,0)
        ]        

    #for applying filters to input letters
    filterMasks4 = cnn.FilterMask(mask4)
    filterMasks8 = cnn.FilterMask(mask8)

    ############################
    #
    # CNN using original 4 masks:
    #
    #keep this - best network with 10 hidden nodes, original masks
    #*******
    #best iterations = 400000
    #seed = 7122691150465666943
    #mse = 0.0021
    
    #run number 2:
    #best iterations = 360000
    #seed = 5803821598874378114
    #mse = 0.0014
    
    #******* 20 nodes, new gb
    #best iterations = 271000
    #seed = 6526982303104022899
    #mse = 0.0010
    #*******
    
    #*******
    alpha = 1.5
    eta = .2
    # (from week 5. Copy to week6 project directory)
    gb_base = 'gb_training_8/weights_8' 
    #tHidden = mlp.TransferReLU(0)
    tHidden = mlp.TransferSigmoid(alpha=2.0)
    tOut = mlp.TransferSoftmax(0)
    #tOut = mlp.TransferSoftmax(0)
    #####################################
    rc = test_hidden(alph.Noise_Letters, filterMasks4, gb_base, tHidden, tOut, eta, nHidden=20)
    
    #just train the best network directly by specifying the seed
    #rc = run_test(alph.Standard_Letters_9, filterMasks4, gb_base, tHidden, alpha, eta,seed=7685137965096759918,nHidden=7, bOutputByClass = False, bHidden=True, bWeights=True)
    
    ############################
    #keep this - best network with 20 hidden nodes, 10 masks
    #*******
    #best iterations = 427000
    #seed = 195153308359688240
    #mse = 0.0018    
    #*******
    alpha = 2.0
    eta = .2
    # (from week 5. Copy to week6 project directory)
    tHidden = mlp.TransferSigmoid(alpha)
    tOut = mlp.TransferSoftmax(0)
    gb_base = 'gb_training_8/weights_8' 
    
    ############################
    rc = test_hidden(alph.Noise_Letters, filterMasks8, gb_base, tHidden, tOut, eta, nHidden=20)

    
    #verify against test sets    
    cn4_base = 'cn_training_4/weights_20'
    cn8_base = 'cn_training_10/weights_20'
    
    rc = test_nn(alph.All_Letters, filterMasks4, gb_base, cn4_base, nHidden=20)
    rc = test_nn(alph.Noise5, filterMasks4, gb_base, cn4_base,nHidden=20)
    rc = test_nn(alph.Noise10, filterMasks4, gb_base, cn4_base,nHidden=20)
    
    rc = test_nn(alph.All_Letters, filterMasks8, gb_base, cn8_base, nHidden=20)
    rc = test_nn(alph.Noise5, filterMasks8, gb_base, cn8_base,nHidden=20)
    rc = test_nn(alph.Noise10, filterMasks8, gb_base, cn8_base,nHidden=20)
    
    

def dontrun():
    x=1
                    
if __name__ == "__main__": dontrun()

####################################################################################################
# End program
#################################################################################################### 

