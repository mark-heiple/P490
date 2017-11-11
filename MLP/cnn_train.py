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
    
    #each depth slice of the mask must be 1 dimension
    #if it has depth, the 2nd dimension is the depth
    #the input masks must be in the format
    #(numFilters, depth, filtersize)
    def __init__(self, masks):
        
        #debug
        '''
        masks = [    
            (0,1,0, 0,1,0, 0,1,0),
            (0,0,0, 1,1,1, 0,0,0),       
            (1,0,0, 0,1,0, 0,0,1),
            (0,0,1 ,0,1,0, 1,0,0) 
            ]        
        
        #2d
        masks_dd = [    
            [(0,1,0, 0,1,0, 0,1,0),(1,0,1, 1,0,1, 1,0,1)],
            [(0,0,0, 1,1,1, 0,0,0),(1,1,1, 0,0,0, 1,1,1)],       
            [(1,0,0, 0,1,0, 0,0,1),(0,1,1, 1,0,1, 1,1,0)],
            [(0,0,1 ,0,1,0, 1,0,0),(1,1,0, 1,0,1, 0,1,1)]
            ]     
        '''   
        
        #save array of masks
        self.masks = masks
        
        #masks may be either 1D or 2D (if depth > 1)
        #allow for both cases
        mm = np.array(masks)
        mm_shape = mm.shape
        
        #scale the weights to match the old method
        #sum each individual filter depth slice
        mm_scale = np.sum(mm,axis=len(mm_shape)-1)
        
        #need to reshape scale values to original values (add dimension back)
        shape_parms = np.array(mm_shape[0:len(mm_shape)-1])
        shape_parms = np.append(shape_parms,[-1])
        shape_parms = tuple(shape_parms)
        mm_scale = mm_scale.reshape(shape_parms)
        
        #convert to float
        mm_scale = mm_scale*1.0
        #scale it
        mm = mm/mm_scale
        
        #verify the depth slice of the mask is square        
        maskLen = mm_shape[len(mm_shape)-1]
        wf = int(math.sqrt(maskLen)+.1)
        assert maskLen % wf == 0, 'Masks must be square'
        self.WF = self.HF = wf
        
        #expand multi-dimensional masks out
        #the extra dimensions are added as extra columns
        #number of rows is always number of masks
        self.DF = 1
        self.wwf = mm
        if len(mm_shape) > 2:
            self.DF = mm_shape[len(mm_shape)-2]
            self.wwf = mm.reshape(mm_shape[0],-1)
            
        #each filter has its own bias value        
        #l_bias = np.zeros((1, 1, 1, NF))        
        self.wwb = np.zeros(mm_shape[0])

    #the im2col methods have been adapted from the PyConvNet sample.
    #PyConvNet takes multiple inputs with a depth > 1, multiple filters,
    #and creates a single matrix holding it all
    
    #this (for now) is simplified to a single 2 dimensional input and 1 filter
    
    #x_shape: height/width of input (x)
    #HF,WF: height/width of filter
    #pad: pad zero around x_shape
    #stride: amount to move filter
    def im2col_index(self, x_shape, HF, WF, pad=1, stride=1):
        
        # get input size
        #H, W, D, N = x_shape
        H, W, D = x_shape
        
        # get output size
        out_h = 0
        out_w = 0
        
        if type(pad) is int:
            #only allow integer padding
            out_h = (H + 2 * pad - HF) / stride + 1
            out_w = (W + 2 * pad - WF) / stride + 1
        else:
            #custom padding at each side (before x_1, after x_n, before y_1, after y_n)
            out_h = (H + pad[0] + pad[1] - HF) / stride + 1
            out_w = (W + pad[2] + pad[3] - WF) / stride + 1
            
        # for row index, compute the first index of the first HF * WF block
        r0 = np.repeat(np.arange(HF), WF)
        r0 = np.tile(r0, D)
        
        # then compute the bias of each block
        r_bias = stride * np.repeat(np.arange(out_h), out_w)
        
        # then the row index is the r0 + r_bias
        # this creates a 2D matrix
        
        #the value of each column vector item in rr represents the row index
        #of the 2D padded input that is used for the filtered output, one
        #column per filtered output (total of W * H columns)
        rr = r0.reshape(-1, 1) + r_bias.reshape(1, -1)
    
        # the same to the col index
        c0 = np.tile(np.arange(WF), HF * D)
        c_bias = stride * np.tile(np.arange(out_w), out_h)
        
        #the value of each column vector item in cc represents the column index
        #of the 2D padded input that is used for the filtered output, one
        #column per filtered output (total of W * H columns)
        cc = c0.reshape(-1, 1) + c_bias.reshape(1, -1)
        
        #example: 9x9 (81) input, with pad = 1, stride = 1 -> 11x11.
        # Filter is 3x3 (total of 9)
        #
        #rr and cc are both 9 x 81. Each column represents the row and column
        #index of the padded input that is used to compute a single filter
        #output value that will go into a [1 x 81] result vector
        
        #the first 9 columns of rr and cc represent the 1st 9 (one row) of
        #filtered output
        
        #rr[:,0:9]
        # array([[0, 0, 0, 0, 0, 0, 0, 0, 0],
        #        [0, 0, 0, 0, 0, 0, 0, 0, 0],
        #        [0, 0, 0, 0, 0, 0, 0, 0, 0],
        #        [1, 1, 1, 1, 1, 1, 1, 1, 1],
        #        [1, 1, 1, 1, 1, 1, 1, 1, 1],
        #        [1, 1, 1, 1, 1, 1, 1, 1, 1],
        #        [2, 2, 2, 2, 2, 2, 2, 2, 2],
        #        [2, 2, 2, 2, 2, 2, 2, 2, 2],
        #        [2, 2, 2, 2, 2, 2, 2, 2, 2]])
        #
        # if depth of input > 1, then the block of rows
        # is repeated for each filter
        # ex: if 2 filters, rr.shape = [18 x 81] instead of [9 x 81]
        
        
        #cc[:,0:9]
        # array([[ 0,  1,  2,  3,  4,  5,  6,  7,  8],
        #        [ 1,  2,  3,  4,  5,  6,  7,  8,  9],
        #        [ 2,  3,  4,  5,  6,  7,  8,  9, 10],
        #        [ 0,  1,  2,  3,  4,  5,  6,  7,  8],
        #        [ 1,  2,  3,  4,  5,  6,  7,  8,  9],
        #        [ 2,  3,  4,  5,  6,  7,  8,  9, 10],
        #        [ 0,  1,  2,  3,  4,  5,  6,  7,  8],
        #        [ 1,  2,  3,  4,  5,  6,  7,  8,  9],
        #        [ 2,  3,  4,  5,  6,  7,  8,  9, 10]])
        #
        # if depth of input > 1, then the block of rows
        # is repeated for each filter
        # ex: if 2 filters, rr.shape = [18 x 81] instead of [9 x 81]
                                
        
        # filtered[0] = sum(f(padded[rr[:,0],cc[:0]))
        # filtered[1] = sum(f(padded[rr[:,1],cc[:1]))
        #...
        # filtered[80] = sum(f(padded[rr[:,80],cc[:80]))
        
        # where f() is the weighed (filtered) value at padded(rr,cc)
        # the filter values have not been expanded yet
        
        # the number of filters is not used yet either
        # these are the array indexes of the x_padded values
        # that will be mapped to another matrix for the matrix dot product
        
    
        # then the dimension index
        dd = np.repeat(np.arange(D), HF * WF).reshape(-1, 1)
    
        #used double letters because single letters are shared with debugger commands
        return (rr, cc, dd)
    
    def im2col(self, xx, HF, WF, pad, stride):
        
        # padding
        x_padded = None
        
        #only pad the 1st 2 dimensions (width and height, not depth)
        if type(pad) is int:
            #x_padded = np.pad(x, ((pad, pad), (pad, pad), (0, 0), (0, 0)), mode='constant')
            #x_padded = np.pad(xx, ((pad, pad), (pad, pad)), mode='constant')
            x_padded = np.pad(xx, ((pad, pad), (pad, pad), (0, 0)), mode='constant')
        else:
            #x_padded = np.pad(x, ((pad[0], pad[1]), (pad[2], pad[3]), (0, 0), (0, 0)), mode='constant')
            #x_padded = np.pad(xx, ((pad[0], pad[1]), (pad[2], pad[3])), mode='constant')
            x_padded = np.pad(xx, ((pad[0], pad[1]), (pad[2], pad[3]),(0,0)), mode='constant')
            
        #example: padded input, where input is a vector[1,81] arranged by row
        #
        #x_padded
        # array([[ 0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0],
        #        [ 0,  1,  2,  3,  4,  5,  6,  7,  8,  9,  0],
        #        [ 0, 10, 11, 12, 13, 14, 15, 16, 17, 18,  0],
        #        [ 0, 19, 20, 21, 22, 23, 24, 25, 26, 27,  0],
        #        [ 0, 28, 29, 30, 31, 32, 33, 34, 35, 36,  0],
        #        [ 0, 37, 38, 39, 40, 41, 42, 43, 44, 45,  0],
        #        [ 0, 46, 47, 48, 49, 50, 51, 52, 53, 54,  0],
        #        [ 0, 55, 56, 57, 58, 59, 60, 61, 62, 63,  0],
        #        [ 0, 64, 65, 66, 67, 68, 69, 70, 71, 72,  0],
        #        [ 0, 73, 74, 75, 76, 77, 78, 79, 80, 81,  0],
        #        [ 0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0]])                    
        
        #rr, cc, dd = im2col_index(x.shape, HF, WF, pad, stride)
        #cols = x_padded[rr, cc, dd, :]

        #get the x_padded[rr,cc] indexes that are used to calculate each filtered output
        rr, cc, dd = self.im2col_index(xx.shape, HF, WF, pad, stride)
        
        #build a matrix of those values
        #this is the same shape as rr and cc
        cols = x_padded[rr, cc, dd]
        
        #continuing the example of our padded 9x9 input (11x11)
        #the input is a vector [1..81], padded with 0, arranged by row
        #
        # each column is a 3x3 grid of values sliding across the top row of x_padded
        #
        # array([[ 0,  0,  0,  0,  0,  0,  0,  0,  0],
        #        [ 0,  0,  0,  0,  0,  0,  0,  0,  0],
        #        [ 0,  0,  0,  0,  0,  0,  0,  0,  0],
        #        [ 0,  1,  2,  3,  4,  5,  6,  7,  8],
        #        [ 1,  2,  3,  4,  5,  6,  7,  8,  9],
        #        [ 2,  3,  4,  5,  6,  7,  8,  9,  0],
        #        [ 0, 10, 11, 12, 13, 14, 15, 16, 17],
        #        [10, 11, 12, 13, 14, 15, 16, 17, 18],
        #        [11, 12, 13, 14, 15, 16, 17, 18,  0]])        
        
        
        #entire matrix reshaped into a 2 dimensional matrix
        #each layer of the depth dimension is repeated as another
        #set of rows. We only have 1 dimension here
        #cols = cols.reshape(HF * WF * x.shape[2], -1)
        return cols
    
    
    #convert a column back into its image representation
    #this is used for backprop?
    #not sure what it is doing
    def col2im(self, cols, x_shape, HF, WF, pad, stride):
        
        #cols in a 2D matrix, each column is the values
        #that went into calculating a single filtered output
        
        # get input size
        #H, W, D, N = x_shape
        H, W = x_shape
        D = 1
        N = 1
        H_padded = 0
        W_padded = 0
        
        #calculate padded dimensions
        if type(pad) is int:
            H_padded, W_padded = H + 2 * pad, W + 2 * pad
        else:
            H_padded, W_padded = H + pad[0] + pad[1], W + pad[2] + pad[3]
            
        #create a zeroed matrix representning the padded dimensions
        #x_padded = np.zeros((H_padded, W_padded, D, N), dtype=cols.dtype)
        x_padded = np.zeros((H_padded, W_padded), dtype=cols.dtype)
        
        #rr, cc, dd = im2col_index(x_shape, HF, WF, pad, stride)
        #indexes of original padded input that went into cols
        rr, cc = self.im2col_index(x_shape, HF, WF, pad, stride)
        
        #reshape cols into its depth (D==1, so not necessary)
        #cols_reshaped = cols.reshape((HF * WF * D, -1, N))
        
        #summing up somethings...
        #np.add.at(x_padded, (rr, cc, dd, slice(None)), cols_reshaped)
        np.add.at(x_padded, (rr, cc), cols)
        
        #return unpadded result
        if pad == 0:
            x = x_padded
        elif type(pad) is int:
            #x = x_padded[pad:-pad, pad:-pad, :, :]
            x = x_padded[pad:-pad, pad:-pad]
        else:
            x = x_padded[pad[0]:-pad[1], pad[2]:-pad[3]]
            
        return x
            
                
    #old code
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
    
    def FilterLetter_old(self, dIn):
        
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

    def FilterLetter(self, dIn):
        
        #get length of x (must account for both 1 (depth=1) and 2 (depth > 1) dimensions
        xlen = dIn.shape
        xlen = xlen[len(xlen)-1]
        x_w = int(math.sqrt(xlen)+.1)
        assert xlen % x_w == 0, 'Input is not square'
        
        #reshape into 3 dimensions (Width x Height x Depth)
        #if depth = 1, there is still 3 dimensions
        xx = dIn.reshape(1,-1)
        xx = np.transpose(xx.reshape((x_w,x_w,-1), order='F'), (1,0,2))
        
        #get depth
        x_shape = xx.shape
        x_d = x_shape[len(x_shape)-1]
        
        assert x_d == self.DF, 'Input depth does not match filter depth'
        
        #move this to __init__        

        #expand filters to match depth by making wider.
        #we want 1 row per filter
        #the X variables have a block of rows for each filter depth
        
        #NOTE: when we replace this with training filters, we will create
        #each filter depth slice - each input depth slize will have its own filter depth slice
        
        #unroll input
        cols = self.im2col(xx, self.WF, self.HF, pad=1, stride=1)
        
        #filter input with 1 mask
        NF = len(self.wwf) 
            
        #transpose filter into 1 column to so that w can do a dot product
        #(already correct shape)
        #w_col = w.transpose(3, 0, 1, 2).reshape((NF, -1))
        #w_col = ww.reshape(NF,-1)
        w_col = self.wwf
            
        #apply the filter
        #result is [1 x 81]
        output_col = np.dot(w_col,cols)
            
        #add bias (zero for now)
        #must turn wb into a column vector
        output_col = output_col + self.wwb.reshape(-1,1)
            
        #reshape for number of filters and input values
        #don't need to do this
        #not being fed into another filter
        #output_col = w_col.dot(x_col) + b.reshape(-1, 1)
        #output_col = output_col.reshape((NF, HO, WO, N))
        #output_col = output_col.transpose(1, 2, 0, 3)
        out_shape = output_col.shape
        out_len = out_shape[len(out_shape)-1]
        out_w = int(math.sqrt(out_len)+.1)
            
        #output will be NF x out_w x out_w
        output_matrix = output_col.reshape((NF,out_w,out_w))
            
        #reshape to out_w x out_w x NF
        output_matrix = output_matrix.transpose(1,2,0)
            
        #turn into 1 giant 1D array for fully connected layer
        filtered = output_col.reshape(1,-1)
        filtered = filtered[0]

        return filtered
                

def test_filter():
    
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
    
    masks = [    
        (0,1,0, 0,1,0, 0,1,0),
        (0,0,0, 1,1,1, 0,0,0),       
        (1,0,0, 0,1,0, 0,0,1),
        (0,0,1 ,0,1,0, 1,0,0) 
        ]        

    #for applying filters to input letters
    filterMasks = FilterMask(masks)
    
    inputArray = np.array(Letters[0][1])
    
    #old method
    filteredInputArray1 = filterMasks.FilterLetter_old(inputArray)
    
    #unrolled matrix
    filteredInputArray2 = filterMasks.FilterLetter(inputArray)
    
    #should be the same
    diff = filteredInputArray1 - filteredInputArray2
    print diff
    print 'length of filteredInputArray1 ', len(filteredInputArray1)
    print 'length of filteredInputArray2 ', len(filteredInputArray2)

    
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
    

if __name__ == "__main__": test_filter()


#temporary - build x array for testing
x = np.arange(81)+1
#x2 = np.arange(81)+101
#x = np.array([x1,x2])
