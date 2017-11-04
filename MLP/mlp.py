
# Code from Chapter 4 of Machine Learning: An Algorithmic Perspective (2nd Edition)
# by Stephen Marsland (http://stephenmonika.net)

# You are free to use, change, or redistribute the code in any way you wish for
# non-commercial purposes, but please maintain the name of the original author.
# This code comes with no warranty of any kind.

# Stephen Marsland, 2008, 2014

import numpy as np
import datafile as df
import random

# We want to use the exp function (e to the x); it's part of our transfer function definition
from math import exp

# So we can make a separate list from an initial one
import copy

#clears the console
def clear():
    sys.stderr.write("\x1b[2J\x1b[H")
    
## define Transfer class types
TRANSFER_LINEAR = 11
TRANSFER_SIGMOID = 22
TRANSFER_TANH = 33
TRANSFER_SOFTMAX = 44
TRANSFER_RELU = 55

#absract base class from which all transfer classes are derrived
class Transfer:
    
    def Compute(self,x):
        raise NotImplementedError()
        
    def Derivative(self,y):
        raise NotImplementedError()
        
    def type(self):
        raise NotImplementedError()

    def parm(self):
        raise NotImplementedError()
    
#for linear regression
class TransferLinear(Transfer):
    
    def __init__(self, parm):
        return
        
    def type(self):
        return TRANSFER_LINEAR
        
    def parm(self):
        return 0
        
    def Compute(self,x):
        return x
        
    def Derivative(self,y):
        return 1
        
#for logistic
class TransferSigmoid(Transfer):
    
    def __init__(self, alpha):
        self.name="Sigmoid"
        self.alpha = alpha
        
    def type(self):
        return TRANSFER_SIGMOID

    def parm(self):
        return self.alpha

    def Compute(self,x):
        y = 1.0 / (1.0 + np.exp(-self.alpha*x))
        return y
        
    def Derivative(self,y):
        x = self.alpha*y*(1.0 - y)  
        return x
        
#softmax transfer function        
class TransferSoftmax(Transfer):
    
    def __init__(self, ndata):
        self.name="Softmax"
        self.ndata = ndata
        
    def type(self):
        return TRANSFER_SOFTMAX

    def parm(self):
        return self.ndata

    def Compute(self,x):
        #normalisers = np.sum(np.exp(x),axis=1)*np.ones((1,np.shape(x)[0]))
        #y =  np.transpose(np.transpose(np.exp(x))/normalisers)
        e_x = np.exp(x - np.max(x))
        y = e_x / e_x.sum()
        return y
        
    def Derivative(self,y):
        #x=(y*(-y)+y)/self.ndata  
        x = y*(1.0 - y)  
        return x
       
#tanh Transfer function
class TransferTanh(Transfer):
    
    def __init__(self, alpha):
        self.name="Tanh"
        self.alpha = alpha
        
    def type(self):
        return TRANSFER_TANH

    def parm(self):
        return self.alpha

    def Compute(self,x):
        y = np.tanh(self.alpha*x)
        return y
        
    def Derivative(self,y):
        x = self.alpha*(1-y*y)
        return x

#ReLU Transfer function
class TransferReLU(Transfer):
    
    def __init__(self, parm):
        return
        
    def type(self):
        return TRANSFER_RELU
        
    def parm(self):
        return 0
        
    def Compute(self,x):
        z = (np.array(x)>0)*x
        return z
        
    def Derivative(self,y):
        y = np.array(y)
        x = (y<0)*0 + (y==0)*.5 + (y>0)*1
        return x
        
        
#this is used by mlp.read_weights create the appropriate Transfer function object        
def getTransfer(ttype, parm):
    t=Transfer()
    if ttype==TRANSFER_LINEAR:
        t = TransferLinear(parm)
    if ttype==TRANSFER_SIGMOID:
        t = TransferSigmoid(parm)
    if ttype==TRANSFER_TANH:
        t = TransferTanh(parm)
    if ttype==TRANSFER_SOFTMAX:
        t = TransferSoftmax(parm)
    if ttype==TRANSFER_RELU:
        t = TransferReLU(parm)
        
    return t
                

############################################################################
# This class represents an MLP with a single hidden layer
#
# The class can be created using two different methods:
#
#  mlp.init() creates a new mlp. The number of nodes and transfer function
#  in each layer and the training rate paramter. randomize() must also be called
#  to randomize the weights.
#
#  mlp.init(
#    nInput     - The number of input nodes
#    nHidden    - the number of hidden layer nodes
#    tHidden    - the Transfer function for the hidden layer
#    nOutput    - the number of output nodes
#    tOutput    - the Transfer function for the output layer
#    eta        - the training rate
#   )
#
#  mlp.read() creates a new mlp by reading data files created by
#  mlp.write_weights():
#
#  mlp.read(
#   basename    - The base filename that is used to create the filnames
#                 for all of the files
#                 This must be the same base name used in mlp.write_weights()
#   )
#
#  Basic usage:
#
#  forward(    Forward feeds inputs through the network and calculates outputs
#   dIn         - The input data array
#   )
#
#  backprop(    Performs back propogation and updates weights
#   dIn         - The input data array
#   dTarget     - The desired output data array
#   )
#
#  write_weights(   Writes out the internal weights and Transfer function parameters
#                   to data files.
#   basename    - The base name used to write the files. A total of 5 files are
#                 created: 4 weight files and a Transfer function parameter file
#   )
############################################################################

class mlp(object):
    
    """ A Multi-Layer Perceptron (with 1 hidden layer)"""
    
    # create a new MLP. you must call randomize() on the new object
    # to randomize the weights
    #
    # mm = mlp.init( ...parameters...)
    # mm.randomize(seed = value)
    @classmethod
    def init(cls,nInput, nHidden, tHidden, nOutput, tOutput, eta):
        mm = mlp()
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
        
        #init dropout percent to 0 (no dropout)
        mm.pDropout = 0
        
        return mm
        
        
    # restore a saved MLP from files created with write_weights()
    # use the same base filename that was used in write_weights()
    @classmethod
    def read(cls,filename):
        mm = mlp()
        
        #init dropout percent to 0 (no dropout)
        mm.pDropout = 0

        mm.read_weights(filename)
        
        return mm
        
    ### accessors to return layer outputs
    def Output(self):
        return self.dOut
        
    def Hidden(self):
        return self.dHidden
        
    def SetDropout(self, percent):
        self.pDropout = percent
        
    #convert vector into a column matrix
    def toColumnMatrix(self,v):
        
        if len(v.shape)==1:
            m = np.transpose(np.array([v]))
        else:
            m = v
            
        return m
    
    #convert vector into a row matrix
    def toRowMatrix(self,v):
        
        if len(v.shape)==1:
            m = np.array([v])
        else:
            m = v
            
        return m

    #scale a random number [0,1] to [-1,1]
    def scale(self,x):
        y = 1.0 - 2.0 * x
        return y
        
    
    #create a random array using teh same random function as the
    #the original code, so that it duplicates the same random numbers
    #input is a tuple (rows,cols). if len=1 it is a vector
    def randomArray(self, dims):
        
        rows = dims[0]
        cols = dims[1]
            
        if cols == 1:
            #1 dimension list comprehension
            z = np.array([random.random() for x in range(rows)])
            
        else:
            #2 dimensional list comprehension
            z = np.array([[random.random() for y in range(cols)] for x in range(rows)])
            
        #scale it
        z = self.scale(z)
        
        return z
        
        
    #the various random functons are different
    #generate random weights
    def randomize(self, seed):
        
        random.seed(seed)
        self.ww = self.randomArray((self.nHidden,self.nInput))
        self.wv = self.randomArray((self.nOutput, self.nHidden))
        self.bh = self.randomArray((self.nHidden,1))
        self.bo = self.randomArray((self.nOutput,1))

        #I would rather use numpy.randdom(), but
        #it generates a different sequence than random.random(),
        #which is used in the original code
        #self.ww = self.scale(np.random.random((self.nHidden,self.nInput)))
        #self.wv = self.scale(np.random.random((self.nOutput, self.nHidden)))
        #self.bh = self.scale(np.random.random(self.nHidden))
        #self.bo = self.scale(np.random.random(self.nOutput))
        
    #read previously saved weights from files
    #this does not verify that the weights are compatible
    def read_weights(self, filename):
        
        fname = "%s_ww.csv" % filename
        f = df.DataFile(fname)
        self.ww = f.read()
        self.nInput = self.ww.shape[1]
        self.nHidden = self.ww.shape[0]

        fname = "%s_wv.csv" % filename
        f = df.DataFile(fname)
        self.wv = f.read()
        self.nOutput = self.wv.shape[0]

        fname = "%s_bh.csv" % filename
        f = df.DataFile(fname)
        self.bh = f.read()
                
        fname = "%s_bo.csv" % filename
        f = df.DataFile(fname)
        self.bo = f.read()
        
        fname = "%s_parms.csv" % filename
        f = df.DataFile(fname)
        parms = f.read()
        
        #eta is row 0
        self.eta = parms[0][1]
        self.tHidden = getTransfer(parms[1][0],parms[1][1])
        self.tOutput = getTransfer(parms[2][0],parms[2][1])
        
        #initialize outputs
        self.dOut = np.zeros(self.nOutput)
        self.dHidden = np.zeros(self.nHidden)
            
        
    #save weights
    def write_weights(self, filename):
        
        f = df.DataFile("%s_ww.csv" % filename)
        f.add(self.ww)
        f.write()

        f = df.DataFile("%s_wv.csv" % filename)
        f.add(self.wv)
        f.write()

        f = df.DataFile("%s_bh.csv" % filename)
        f.add(self.bh)
        f.write()
                
        f = df.DataFile("%s_bo.csv" % filename)
        f.add(self.bo)
        f.write()
        
        f = df.DataFile("%s_parms.csv" % filename)
        row = np.array([0, self.eta])
        f.add(row)
        row = np.array([self.tHidden.type(), self.tHidden.parm()])
        f.add(row)
        row = np.array([self.tOutput.type(), self.tOutput.parm()])
        f.add(row)
        f.write()
        
      
    #compute forward outputs  
    #Note: this does bias weights separately, even though they
    #could be combined with their respective layers.
    #this involves concatinating of matrices, this is simpler
    def forward(self, dIn):
        
        #input to hidden
        hh = np.dot(self.ww,dIn) + self.bh
        self.dHidden = self.tHidden.Compute(hh)
        
        #do dropout (set it to 0 if network is not training)
        if self.pDropout > 0:
            #array of ones that is num inputs x hidden size
            #TODO: if dIn is a vector, len = 1
            in_dim = 1
            if len(dIn.shape)>1:
                in_dim = len(dIn)
                ones = np.ones((in_dim,self.nHidden))
            else:                
                ones = np.ones(self.nHidden)
            
            #returns a 3 dimensional array, want 2 dimensions
            bb = np.random.binomial([ones],1-self.pDropout)
            bb0 = bb[0]
            self.dHidden = self.dHidden * bb0
            #now scale it
            self.dHidden = self.dHidden * (1/(1-self.pDropout))
        
        #hidden to output
        oo = np.dot(self.wv,self.dHidden) + self.bo
        self.dOut = self.tOutput.Compute(oo)
        
        return self.dOut
       
    #backprop and adjust weights 
    #Note: this does bias weights separately, even though they
    #could be combined with their respective layers.
    #this involves concatinating of matrices, this is simpler
    def backprop(self, dIn, dTarget):
        
        #error is target output - actual output
        errorArray = dTarget - self.dOut
        
        ########################
        ### output to hidden ###
        ########################
        
        #calculate derivative of transfer function on final layer output
        x_Fo = self.tOutput.Derivative(self.dOut)
        
        #This is the Eo*Fo[1-Fo] term. It is used in both the output to hidden
        #and hidden to input backprop equations       
        x_EoFo = errorArray * x_Fo
        
        #need to convert vectors to matrices
        x_a = self.toColumnMatrix(x_EoFo)
        
        #output from hidden layer
        x_b = self.toRowMatrix(self.dHidden)

        #calculate p(SSE)/p(v_h,o) = Eo*Fo[1-Fo] * Hh
        x_pv = np.dot(x_a,x_b)

        #adjust by training rate
        x_dv = self.eta*x_pv
        
        x_new_wv = self.wv + x_dv
        
        ########################
        ###    bias output   ###
        ########################

        #these are the same calculations as the hidden to output weights
        #but Hh = 1, so it drops out.
        
        #p*SSE/p(v_h,o) = Eo*Fo[1-Fo] * 1
        x_pv = x_EoFo
        x_dv = self.eta*x_pv
        x_new_bo = self.bo + x_dv
        
        ########################
        ### hidden to input  ###
        ########################
    
        #calculate derivative of transfer function on hidden layer output
        x_Fh = self.tHidden.Derivative(self.dHidden)
    
        #sum(v_h,o * Eo * Fo[1-Fo]) term (v_h,o does not include bias weight)
        #result is [num hidden nodes x 1]
        
        #Note: A dot B sums across the columns of A. The backprop equation
        #sums across the Outputs, which is represented by rows. This requires
        #the use of numpy.transpose()
        x_dv = np.dot(np.transpose(self.wv),x_EoFo)
    
        #Fh * I * sum
        x_a = self.toColumnMatrix(x_Fh*x_dv)
        x_b = self.toRowMatrix(np.array(dIn))
        x_pw = np.dot(x_a,x_b)
        
        #adjust by training rate
        x_dw = self.eta * x_pw
        x_new_ww = self.ww + x_dw
        
        
        ########################
        ###    hidden bias   ###
        ########################

        #these are the same calculations as the hidden to output weights
        #but input = 1, so it drops out.
        
        #Fh * I * sum, but I = 1
        x_pw = x_Fh * x_dv
        
        #adjust by training rate
        x_dw = self.eta * x_pw
        x_new_bh = self.bh + x_dw
        
        ##########################
        ### update new weights ###
        ##########################
        self.ww = x_new_ww
        self.wv = x_new_wv
        self.bh = x_new_bh
        self.bo = x_new_bo
        
########################
### end of class mlp ###
########################


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
    tHidden = TransferSigmoid(alpha=1.0)
    tOutput = TransferSigmoid(alpha=1.0)
    
    #number of input, hidden, and output nodes
    nIn = 81
    nHidden = 6
    nOutput = 9
    
    #training rate
    eta = 0.5
    
    #create the nn object
    #use init() to start a new random network
    mm = mlp.init(nIn, nHidden, tHidden, nOutput, tOutput, eta)
    
    #randomize the weights
    mm.randomize(seed=1)
    
    #do 1 training pass through the network using all input values
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
        outputArray = mm.forward(inputArray)
        
        #do backprop
        mm.backprop(inputArray, desiredOutput)
        
    #run an input through and keep it for comparison later
    out1 = mm.forward(np.array(Letters[0][1]))
    
    #save the network
    mm.write_weights("test_")
    
    #create a new network from the saved weights
    #use mlp.read()
    newmm = mlp.read("test_")
    
    #run the same input through the new NN
    out2 = newmm.forward(np.array(Letters[0][1]))
    
    #they should be the same
    diff = out1 - out2
    
    #differences should be zero
    print diff
    

if __name__ == "__main__": test()
