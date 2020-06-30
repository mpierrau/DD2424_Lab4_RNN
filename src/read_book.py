import numpy as np

def getData(fileName):

    f = open(fileName,'r',encoding='utf-8')
    bookData = f.read()
    f.close()

    bookChars = set(bookData)

    return bookData , bookChars


def makeDicts(charDict):
    """ Creates dictionaries for translating: 
            - ints to one hot encodings (indToVec)
            - ints to characters (indToChar)
            - characters to one hot encodings (charToVec) 
    """
    
    idx = range(len(charDict))

    newVec = np.identity(len(charDict))
    
    indToChar = dict(zip(idx,charDict))
    indToVec = {k: newVec[:,[k]] for k in indToChar.keys()}
    charToVec = {v: indToVec[k] for k,v in indToChar.items()}

    return indToVec , indToChar , charToVec