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
MAX_ITERATIONS = 400000
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



#flattens a dictionary into an array sorted by key
def FlattenDict(d):
    keys = np.sort(d.keys())
    aa = np.zeros(len(keys))
    
    i = 0;
    for k in keys:
        v = d[k]
        aa[i] = v
        i = i+1
    
    return aa
    
    
    

#load a mlp from data set to run a set of test inputs through
def ComputeAll (mm, dataset, filterMasks, outputCol, debug = False, bHidden=False):

    
    #get a count of the number of letters being tested
    #a letter may have multiple variants
    dd = pd.DataFrame(dataset)
    letters = dd[3]
    freq = pd.value_counts(letters)
    llist = list(freq.index)
    
    heatmap = np.zeros((mm.nOutput,mm.nHidden))
    
    #initialize dictionary
    #using a dictionary because test set may not have all letters
    SSE_Dict = {}
    for ll in llist:
        SSE_Dict[ll]=0
        
    #loop through each training set
    for iTest in range(len(dataset)):
        
        #test sample
        trainingDataList = dataset[iTest]
        
        #input values
        trainingDataInputList = trainingDataList[1]      
        inputDataArray = np.array(trainingDataInputList)

        letterNum = trainingDataList[2]
        letterChar = trainingDataList[3]
        letterClass = trainingDataList[4]
        outputArrayLength = mm.nOutput
        
        if debug == True:
            print ' '
            print '  Data Set Number', iTest, ' for letter ', letterChar, ' with letter number ', letterNum+1 
        
          
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
        
        if debug == True:
            print ' '
            print ' The hidden node activations are:'
            print mm.Hidden()
            
            print ' '
            print ' The output node activations are:'
            print outputArray   
            
            print ' '
            print ' The desired output array values are: '
            print desiredOutputArray  
            
            print ' '
            print ' The error values are:'
            print (desiredOutputArray-outputArray)   
            
            print 'New SSE = %.6f' % newSSE 
        

        #save error in dictionary
        e = SSE_Dict[letterChar]
        e = e + newSSE
        SSE_Dict[letterChar] = e
        
        #build heatmap
        heatmap[desiredClass] = mm.Hidden()
        
        
    mseTotal = 0
    nTotal = 0
    
    #calculate average error of each SSE value
    for char in SSE_Dict:
        mseTotal = mseTotal + SSE_Dict[char]
        nTotal = nTotal + freq[char]
        
        e = SSE_Dict[char]/freq[char]
        SSE_Dict[char] = e
    
    mseTotal = mseTotal/nTotal
    
    if(bHidden==True):
        fheatmap = datafile.DataFile("heatmap_%d.csv" % mm.nHidden)
        fheatmap.add(heatmap)
        fheatmap.write()

    if debug == True:
        print SSE_Dict
        print 'Total MSE = ',mseTotal
    
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
    
    ticks = np.sort(SSE_Dict.keys())
    yvals = [SSE_Dict[y] for y in ticks] 

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
       
def plot_heatmap(dataset, filename):
    
    ticks = Letters(dataset)
    rticks = range(len(ticks))
    
    # Plot heatmap (this works)
    fheatmap = datafile.DataFile(filename)
    h = fheatmap.read()
    nHidden = h.shape[1]
    
    plt.clf()
    plt.title('Hidden Layer Activations')
    plt.ylabel('Letter Class')
    plt.xlabel('Hidden Node')
    plt.yticks(rticks,ticks)
    plt.xticks(range(nHidden),range(nHidden))
    plt.margins(0.2)
    plt.imshow(h,cmap="bwr")
    plt.show()    
    plt.savefig('heatmap_%d.png' % nHidden)
            
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                           
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
def run_test(dataset, filterMasks, gb_base, tTransferH, alpha, eta, seed, nHidden=6, bOutputByClass = True, bHidden=False, bWeights=False):

# Define the global variables        
    global inputArrayLength
    global hiddenArrayLength
    global outputArrayLength
    global gridWidth
    global gridHeight
    global eGH # expandedGridHeight, defined in function expandLetterBoundaries 
    global eGW # expandedGridWidth defined in function expandLetterBoundaries 
    
    #noise factor
    #(no noise)
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
    outputClassColumn =arraySizeList[3]
    
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
    mm = cnn.cnn.init(gb,inputArrayLength, hiddenArrayLength, tTransferH, outputArrayLength, mlp.TransferSigmoid(alpha), eta)
    
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
#        index = random.randint(0, numTrainingDataSets-1)
        index = random.randint(1, numTrainingDataSets)
        index = index - 1
        
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
        if addNoise == True:
            if random.random() < .5:
                noise = np.random.random(inputDataLen)<gamma
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
        outputArray = mm.forward(noisyInputDataArray, filteredInputDataArray)
        
        #not keeping track of all outputs any more
        #keeping average SSE across all letter variants
        #so do a ComputeAll every 100 iterations
    
        # Determine the error between actual and desired outputs
        maxSSE = epsilon+1
        if iteration % 100 == 0:
            
            maxSSE = 0
            SSE_Dict = ComputeAll(mm, dataset, filterMasks, outputClassColumn, bHidden=False)                           
            numvals = len(SSE_Dict.keys())
                        
            #add up all the SSEs
            for k in SSE_Dict:
                sse = SSE_Dict[k]
                if( sse > maxSSE ):
                    maxSSE = sse
        
        #break out when worst SSE is less than threshold
        if maxSSE < epsilon:
            break
            
    print 'Out of while loop at iteration ', iteration 
    
####################################################################################################
# After training, get a new comparative set of outputs, errors, and SSE 
####################################################################################################                           

    print ' '
    print '  After training:'   
    
                                                      
#    SSE_Array = ComputeOutputsAcrossAllTrainingData (mm, dataset, outputClassColumn, bHidden=True)                           
    SSE_Dict = ComputeAll (mm, dataset, filterMasks, outputClassColumn, debug = True, bHidden=True)
    SSE_Array = FlattenDict(SSE_Dict)      
    MSE = sum(SSE_Array)/len(SSE_Array)                     

    #save weight array
    if( bWeights == True ):
        
        mm.write_weights('weights_%d' % nHidden)

    #MSE and iterations by epoch    
    if( bHidden == True):
        f = datafile.DataFile("sse_%d.csv" % nHidden )
        f.add(SSE_Array)
        f.write()

    
    print( 'iterations = %d' % iteration )
    print( 'MSE = %.6f' % MSE )

    return(iteration,MSE)

                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                  
                                              
####################################################################################################
# Conclude specification of the MAIN procedure
####################################################################################################                

##############################################
# scans through combinations of alpha and eta
###############################################
    
def search_parms(dataset, gb_base, tTransferH, nHidden):

### -- below is for searching for best alpha and eta. not sure I need to do this --- ###
    #for epoch in range(100):
#        seed = random.randint(1,MAXINT)

			
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
                rc = run_test(dataset, gb_base, tTransferH, alpha=a, eta=e, seed=seed, nHidden=nHidden, bOutputByClass = True, bHidden=True, bWeights=True)
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
def test_hidden(dataset, filterMasks, gb_base, tTransferH, alpha, eta, nHidden):

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
        
    while epoch < 25:
        seed = random.randint(1,MAXINT)
        print '###################'
        print 'epoch = %d' % epoch
        print 'seed = %d' % seed
        print '###################'
        
        #use previously found best alpha and eta
        rc = run_test(dataset, filterMasks, gb_base, tTransferH, alpha, eta,seed=seed,nHidden=nHidden, bOutputByClass = True, bHidden=True, bWeights=True)
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
    rc = run_test(dataset, filterMasks, gb_base, tTransferH, alpha, eta,seed=best_seed,nHidden=nHidden, bOutputByClass = True, bHidden=True, bWeights=True)
    #rc = run_test(dataset, tTransferH, alpha, eta=1.5,seed=seed,nHidden=nHidden, bOutputByClass = True, bHidden=True, bWeights=True)
    iterations = rc[0]
    sse = rc[1]
    print '*******'
    print( 'best iterations = %d' % iterations )
    print( 'seed = %d' % best_seed )
    print( 'mse = %.4f' % best_sse )
    print '*******'
    
    # Plot heatmap (this works)
    filename = "heatmap_%d.csv" % nHidden
    plot_heatmap(dataset, filename );
    
    #plot SSE
    fsse = datafile.DataFile("sse_%d.csv" % nHidden)
    sse = fsse.read()
    ticks = Letters(dataset)
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
    
    
######################################################

def test_nn(dataset, filterMasks, gb_base, cnn_base,nHidden):
    
    gb = mlp.mlp.read(gb_base)
    mm = cnn.cnn.read(gb,cnn_base)
    SSE_Dict = ComputeAll(mm, dataset, filterMasks, 2, bHidden=True, debug=True)   
    plot_test_error(SSE_Dict, nHidden)       
                        
#######################################                      

                  
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
        ]        

    #for applying filters to input letters
    filterMasks4 = cnn.FilterMask(mask4)
    filterMasks8 = cnn.FilterMask(mask8)

    #search for best alpha/eta        

    ############################
    #
    # CNN using original 4 masks:
    #
    #keep this - best network with 7 hidden nodes, original masks
    #*******
    # best iterations = 93200
    # seed = 7685137965096759918
    # mse = 0.0033
    #*******
    alpha = 1.5
    eta = .4
    # (from week 5. Copy to week6 project directory)
    gb_base = 'All_Letters_9/weights_6' 
    tHidden = mlp.TransferSigmoid(alpha)
    #####################################
#    search_parms(alph.Standard_Letters_9, gb_base, tHidden, nHidden=7)
    rc = test_hidden(alph.Standard_Letters_9, filterMasks4, gb_base, tHidden, alpha, eta, nHidden=7)
    
    #just train the best network directly by specifying the seed
    #rc = run_test(alph.Standard_Letters_9, filterMasks4, gb_base, tHidden, alpha, eta,seed=7685137965096759918,nHidden=7, bOutputByClass = False, bHidden=True, bWeights=True)
    
    ############################
    #keep this - best network with 7 hidden nodes, 8 masks
    #*******
    alpha = 1.5
    eta = .2
    # (from week 5. Copy to week6 project directory)
    gb_base = 'All_Letters_9/weights_6'
    tHidden = mlp.TransferSigmoid(alpha)
    
    #*******
    #best iterations = 119700
    #seed = 3437327685492954738
    #mse = 0.0051
    #*******    
    ############################
    
    ###
    #another run
    #*******
    #best iterations = 237800
    #seed = 7556860273200878410
    #mse = 0.0027
    #*******    
    ###
    rc = test_hidden(alph.Standard_Letters_9, filterMasks8, gb_base, tHidden, alpha, eta, nHidden=7)
    #rc = run_test(alph.Standard_Letters_9, filterMasks8, gb_base, filterMasks8, tHidden, alpha, eta,seed=3437327685492954738,nHidden=7, bOutputByClass = False, bHidden=True, bWeights=True)

    #test set vs mask 4
    #rc = test_nn(alph.Standard_Letters_9, filterMasks4, 'Standard_Letters_9/weights_6', 'best7_mask4/weights_7',nHidden=7)
    
    #test mse = 1.1744
    #rc = test_nn(alph.Variant_Letters_9, filterMasks4, 'Standard_Letters_9/weights_6', 'best7_mask4/weights_7',nHidden=7)

    rc = test_nn(alph.Variant_Letters_9, filterMasks4, 'All_Letters_9/weights_6', 'best7_mask4/weights_7',nHidden=7)
    
    #MSE = 1.1313
    rc = test_nn(alph.Variant_Letters_9, filterMasks4, 'All_Letters_9/weights_6', 'best7_mask4_2/weights_7',nHidden=7)

    #test set vs mask 8
    #rc = test_nn(alph.Standard_Letters_9, filterMasks8, 'Standard_Letters_9/weights_6', 'best7_mask8/weights_7',nHidden=7)
    
    #mse = 1.1187
    rc = test_nn(alph.Variant_Letters_9, filterMasks8, 'All_Letters_9/weights_6', 'best7_mask8/weights_7',nHidden=7)
    
    #mse = 1.106
    rc = test_nn(alph.Variant_Letters_9, filterMasks8, 'All_Letters_9/weights_6', 'best7_mask8_2/weights_7',nHidden=7)
    
    
#so I can run this file without training a network (which takes a long time)    
def xyz():
    x = 1
    
#if __name__ == "__main__": main()
if __name__ == "__main__": xyz()

####################################################################################################
# End program
#################################################################################################### 

