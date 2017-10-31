import numpy as np
#for saving output
import pandas as pd
from pandas import DataFrame


#clears the console
def clear():
    sys.stderr.write("\x1b[2J\x1b[H")


## This object implements a data file that can read and write 
## 2 dimensional numpy arrays. Each row must have the same number of
## columns, the number of columns is set by the first row (or rows)
## that are added.
##
## Files are read/written as comma delimited CSV text files. There are no
## column headers or row labels.

class DataFile:
    
    #initialize an empty DataFile object. The filename does not have to exist
    #if a new file is being created.
    def __init__(self, filename):
        self._filename = filename
        self.clear()
        
    #reset the internal array to an empty state
    def clear(self):

        self._t = np.array([])
        
    #return number of rows in internal array
    def numrows(self):

        #assume array is 0 rows
        n = 0
        
        #figure out the number of rows that were added
        if len(self._t.shape) == 1:
            
            #this is a column vector. if empty, then the number of rows = 0
            #otherwise, it is 1
            
            n = self._t.shape[0]

            #n will be zero if the array is empty
            
            if n > 0:
                #non-empty column vector, which is treated as 1 row
                n = 1
        else:
            #this a 2 dimensional array representing multiple rows
            n = len(self._t)
            
        return n
        
    #adds 1 or more rows in a numpy array
    #if adding to an existing array, the new
    #rows must have the same number of columns
    def add(self, row):
        
        #number of rows in internal array
        n = self.numrows()
        
        if n == 0:
            #internal array was empty, start a new one
            self._t = row
        else:
            #add to existing array
            self._t = np.vstack((self._t, row))
        
        #return updated array
        return self._t
        
    #write the internal array as a CSV text file, now column headers or row indexes
    def write(self):
        data = DataFrame(self._t)
        data.to_csv(self._filename, header=False, index=False)
        
    #read the CSV text file into the internal array
    def read(self):
        try:
            self._t = np.genfromtxt(self._filename, delimiter=',')
        except IOError as e:
            z = e
            print z
            
        return self._t
              
    #return internal array
    def array(self):
        return self._t
        
    #return the specified row indexes
    def rows(self,indexes):
        return self._t[indexes]
     
     
                     
# function that demonstrates how to use the DataFile object                                             
def test():     
    
    # test creating a DataFile        
    f = DataFile("test.csv")
    
    #rows to be added
    row1 = np.array([1,2,3])
    row2 = np.array([4,5,6])
    
    #2 rows
    row3 = np.array([[7,8,9],[10,11,12]])
    
    #add the rows
    a = f.add(row1)
    print f.numrows()
    print a
    a = f.add(row2)
    print f.numrows()
    print a
    a = f.add(row3)
    print f.numrows()
    print a
    a = f.array()
    print a
    
    #write file (total of 4 rows)
    f.write()
    
    #test reading a DataFile
    #create a new DataFile
    f = DataFile("test.csv")
    
    #should be empty
    print f.array()
    
    #read data
    a = f.read()
    print a

    #test accessng rows of a DataFile
    
    #a single row (row 1)
    b = f.rows(1)
    print b
    
    #a range of rows (rows 0, 1, 2)
    b = f.rows(range(3))
    print b
    
    #a list of rows (rows 0 and 2)
    b = f.rows([0,2])
    print b

    
    #You can access the array directly and use the built in python slicing
    a = f.array()
    print a
    
    #rows 1,2
    b = a[1:3]
    print b
    
    #rows 0-2
    b = a[:3]
    print b
    
    #rows 2,3
    b = a[2:]
    print b
    
    #test clearing the DataFile
    
    f.clear()
    print f.array()


if __name__ == "__main__": test()
