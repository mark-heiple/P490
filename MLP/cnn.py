#implement a convolution NN, which contains an mlp.
#the output from the mlp is part of the input to the CNN

import math
import numpy as np
import mlp

class cnn(mlp.mlp): 
    
    # create a new MLP. you must call randomize() on the new object
    # to randomize the weights
    #
    # mm = mlp.init( ...parameters...)
    # mm.randomize(seed = value)
    #
    # Note: the gb must already have been initialized or read
    @classmethod
    def init(cls, gb, nInput, nHidden, tHidden, nOutput, tOutput, eta):
        
        mm = cnn()
        mm.nInput = nInput
        mm.nHidden = nHidden
        mm.nOutput = nOutput
        mm.tHidden = tHidden
        mm.tOutput = tOutput
        mm.eta = eta
        
        #initialize input to hidden weights
        mm.ww = np.zeros((nHidden,nInput))
        mm.bh = np.zeros(nHidden)
        
        #initialize hidden to outpout
        mm.wv = np.zeros((nOutput,nHidden))
        mm.bo = np.zeros(nOutput)
    
        #initialize outputs
        mm.dOut = np.zeros(nOutput)
        mm.dHidden = np.zeros(nHidden)
        mm.gb = gb
        mm.pDropout = 0
        
        return mm
        
    # restore a saved MLP from files created with write_weights()
    # use the same base filename that was used in write_weights()
    #
    # note: the gb has already been initialized or read
    @classmethod
    def read(cls, gb, basename):
        #mm = super(cnn,cls).read(basename)
        mm = cnn()
        mm.read_weights(basename)
        mm.gb = gb
        mm.pDropout = 0
        
        return mm
        
    def get_basename(self,basename):
        basename_cnn = '%s_cnn' % basename
        return basename_cnn
    
    def read_weights(self, basename):
        basename_cnn = self.get_basename(basename)
        super(cnn,self).read_weights(basename_cnn)
        
    def write_weights(self, basename):
        basename_cnn = self.get_basename(basename)
        super(cnn,self).write_weights(basename_cnn)
    
    
    #forward takes 2 sets of inputs:
    #set 1 goes into greybox, which generates outputs
    #set 2 is combined with the output from greybox and fed into cnn
    def forward(self, dgbIn, dIn):
        
        gbOut = self.gb.forward(dgbIn)
        
        #build input array
        #sample code has dIn first, then gbOut
        
        #copy
        #dAllInputs = dIn[:]
        #dAllInputs.extend(gbOut) 
        dAllInputs = np.hstack((dIn,gbOut))
        
        out = super(cnn,self).forward(dAllInputs)
        return out
        
    #note: backprop is the same - the greybox is not involved
    #however, we need the outputs from the gb
    def backprop(self, dIn, dTarget):
        
        #copy
        #dAllInputs = dIn[:]
        #dAllInputs.extend(self.gb.Output) 
        dAllInputs = np.hstack((dIn,self.gb.Output()))

        super(cnn,self).backprop(dAllInputs,dTarget)
        
        
class FilterMask(object):
    
    def __init__(self, masks):
        
        #save array of masks
        self.masks = masks
        
    #dIn is a 81 character array that represents a 9x9 array
    #original code allows for values in the range [0-1] instead of 
    #just 0 and 1, and converts values that are > .9 to 1
    #is this for adding noise later?
    def expandLetter(self, dIn):
        
        #assume a square matrix
        w = h = int(math.sqrt(len(dIn))+.1)
        
        expandedLetterArray = np.zeros((w+2,h+2))
        
        #account for noise? convert values > .9 to 1
        pixels = (np.array(dIn)>.9)*1
    
        #use slices to build 2-D Array
        #allow for empty row at top and bottom
        
        #start and end of row
        for row in range(1,h+1):
            rowStart = (row-1)*w
            rowEnd = rowStart + w
            expandedLetterArray[row][1:h+1] = pixels[rowStart:rowEnd]
            
        return expandedLetterArray
        
        
    #dIn is a 81 character array that represents a 9x9 array
    def applyFilter(self, mask, expandedLetterArray, gridHeight, gridWidth):
        
        #init masks to all zeros        
        maskLetterArray = np.zeros(shape=(gridHeight,gridWidth))
    
        rowVal = 1
        colVal = 1
        
        #scale by number of pixels so result is between [0,1]
        scale = sum(mask)
        
        while rowVal <gridHeight+1: 
            arrayRow = rowVal - 1 
            while colVal <gridWidth+1:           
                e0 =  expandedLetterArray[rowVal-1, colVal-1]
                e1 =  expandedLetterArray[rowVal-1, colVal]
                e2 =  expandedLetterArray[rowVal-1, colVal+1]   
                e3 =  expandedLetterArray[rowVal, colVal-1]
                e4 =  expandedLetterArray[rowVal, colVal]
                e5 =  expandedLetterArray[rowVal, colVal+1]   
                e6 =  expandedLetterArray[rowVal+1, colVal-1]
                e7 =  expandedLetterArray[rowVal+1, colVal]
                e8 =  expandedLetterArray[rowVal+1, colVal+1]               
              
                maskArrayVal    =  (e0*mask[0] + e1*mask[1] + e2*mask[2] + 
                                    e3*mask[3] + e4*mask[4] + e5*mask[5] + 
                                    e6*mask[6] + e7*mask[7] + e8*mask[8] ) / scale                        
                         
                arrayCol = colVal - 1

                maskLetterArray[arrayRow,arrayCol] = maskArrayVal 
                
                #next column
                colVal = colVal + 1

            #next row
            rowVal = rowVal + 1
            colVal = 1
            
        return maskLetterArray 
    
    def FilterLetter(self, dIn):
        
        lenMasks = len(self.masks)
        lenIn = len(dIn)
        lenFiltered = lenMasks * lenIn
        filtered = np.zeros(lenFiltered)

        #assume a square matrix
        gridHeight = gridWidth = int(math.sqrt(lenIn)+.1)
        
        #expand letter for an extra row and column on both sides
        #this is needed for the filters
        expandedLetterArray = self.expandLetter(dIn)

        #loop through masks
        for i in range(lenMasks):
            
            #filter input with 1 mask
            fArray = self.applyFilter(self.masks[i], expandedLetterArray, gridHeight, gridWidth)
            
            #save into array of filtered values
            #figure out start and end points in final array, based on the
            #mask number and input length
            fStart = i*lenIn
            fEnd = fStart + lenIn
            filtered[fStart:fEnd] = fArray.flatten('C')
            
        return filtered
        

#demo code that demonstrates the use of the mlp class
def test():
    Letters = [
    #trainingDataListA0 =  
    (1,[0,0,0,0,1,0,0,0,0, 0,0,0,1,0,1,0,0,0, 0,0,1,0,0,0,1,0,0, 0,1,0,0,0,0,0,1,0, 1,1,1,1,1,1,1,1,1, 1,0,0,0,0,0,0,0,1, 1,0,0,0,0,0,0,0,1, 1,0,0,0,0,0,0,0,1, 1,0,0,0,0,0,0,0,1],1,'A',0,'A'), # training data list 1 selected for the letter 'A'
    #trainingDataListB0 =  
    (2,[1,1,1,1,1,1,1,1,0, 1,0,0,0,0,0,0,0,1, 1,0,0,0,0,0,0,0,1, 1,0,0,0,0,0,0,1,0, 1,1,1,1,1,1,1,0,0, 1,0,0,0,0,0,0,1,0, 1,0,0,0,0,0,0,0,1, 1,0,0,0,0,0,0,0,1, 1,1,1,1,1,1,1,1,0],2,'B',1,'B'), # training data list 2, letter 'E', courtesy AJM
    #trainingDataListC0 =  
    (3,[1,1,1,1,1,1,1,1,1, 1,0,0,0,0,0,0,0,0, 1,0,0,0,0,0,0,0,0, 1,0,0,0,0,0,0,0,0, 1,0,0,0,0,0,0,0,0, 1,0,0,0,0,0,0,0,0, 1,0,0,0,0,0,0,0,0, 1,0,0,0,0,0,0,0,0, 1,1,1,1,1,1,1,1,1],3,'C',2,'C'), # training data list 3, letter 'C', courtesy PKVR
    #trainingDataListD0 =  
    (4,[1,1,1,1,1,1,1,1,0, 1,0,0,0,0,0,0,1,1, 1,0,0,0,0,0,0,0,1, 1,0,0,0,0,0,0,0,1, 1,0,0,0,0,0,0,0,1, 1,0,0,0,0,0,0,0,1, 1,0,0,0,0,0,0,0,1, 1,0,0,0,0,0,0,1,1, 1,1,1,1,1,1,1,1,0],4,'D',3,'O') # training data list 4, letter 'D', courtesy TD
    ]    

    #create objects that represent the transfer functions to use for each layer
    tHidden = mlp.TransferSigmoid(alpha=1.0)
    tOutput = mlp.TransferSigmoid(alpha=1.0)
    
    #number of input, hidden, and output nodes
    nIn = 81
    nHidden = 6
    nOutput = 9
    
    #training rate
    eta = 0.5
    
    #create the grebox network object
    #use init() to start a new random network
    gb = mlp.mlp.init(nIn, nHidden, tHidden, nOutput, tOutput, eta)
    
    #randomize the weights
    gb.randomize(seed=1)
    
    #train gb network: do 1 training pass through the network using all input values
    for i in range(len(Letters)):
        
        #input values
        testIn = Letters[i]
        inputArray = np.array(testIn[1])

        #desired class is array[4]        
        output = testIn[4]
        #desired output array
        desiredOutput = np.zeros(nOutput)
        desiredOutput[output] = 1
        
        #forward calculation to get output
        #only need outputArray if we are interested in error (we aren't)
        outputArray = gb.forward(inputArray)
        
        #do backprop
        gb.backprop(inputArray, desiredOutput)
        

    #create convolution network using the greybox network    
    
    #convolution mask for CNN
    #masks
    masks = [    
        (0,1,0, 0,1,0, 0,1,0),
        (0,0,0, 1,1,1, 0,0,0),       
        (1,0,0, 0,1,0, 0,0,1),
        (0,0,1 ,0,1,0, 1,0,0) 
        ]        

    #for applying filters to input letters
    filterMasks = FilterMask(masks)
    
    #number of inputs is # of greybox outputs + input size * numMasks
    nInCnn = nOutput + nIn * len(masks)
    
    #this can have a different number of hidden layers than the greybox
    nHiddenCnn = 8
    
    #one output per letter
    nOutCnn = 26
    
    #init new cnn
    mm = cnn.init(gb, nInCnn, nHiddenCnn, tHidden, nOutCnn, tOutput, eta)
    mm.randomize(seed=1)
    
    #do 1 training pass through the CNN
    for i in range(len(Letters)):
        
        #input values
        testIn = Letters[i]
        inputArray = np.array(testIn[1])
        
        #apply convolution filter
        filteredInputArray = filterMasks.FilterLetter(inputArray)

        #desired class is array[2]        
        output = testIn[2]
        #desired output array
        desiredOutput = np.zeros(nOutCnn)
        desiredOutput[output] = 1
        
        #forward calculation to get output
        #only need outputArray if we are interested in error (we aren't)
        #inputArray is the input to the greybox, 
        #filteredInputArray is the input to the CNN (greybox output is combined to CNN input)
        outputArray = mm.forward(inputArray, filteredInputArray)
        
        #do backprop
        #the greybox is not included in the backprop
        mm.backprop(filteredInputArray, desiredOutput)
        
    
    #run an input through and keep it for comparison later
    inputArray = np.array(Letters[0][1])
    filteredInputArray = filterMasks.FilterLetter(inputArray)
    out1 = mm.forward(inputArray, filteredInputArray)
    
    #save the network (must save both)
    gb.write_weights("test_")
    mm.write_weights("test_")
    
    #create a new network from the saved weights
    #use mlp.read()
    newgb = mlp.mlp.read("test_")
    newmm = cnn.read(newgb, "test_")
    
    #run the same input through the new NN
    out2 = newmm.forward(inputArray, filteredInputArray)
    
    #they should be the same
    diff = out1 - out2
    
    #differences should be zero
    print diff
    

if __name__ == "__main__": test()
                