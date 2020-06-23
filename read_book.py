import numpy as np

def parseBook(fileName):

    f = open(fileName,'r',encoding='utf-8')
    bookData = f.read()
    f.close()

    bookChars = set(bookData)

    return bookData , bookChars , len(bookChars)


def makeDicts(bookData):

    uniqueChars = set(bookData)
    uc_len = len(uniqueChars)
    idx = range(len(uniqueChars))

    newVec = np.identity(uc_len)
    
    indToChar = dict(zip(idx,uniqueChars))
    #indToVecTmp = dict(zip(idx,newVec))
    indToVec = {k: newVec[:,[k]] for k in indToChar.keys()}
    charToInd = {v: k for k, v in indToChar.items()}
    charToVec = {k: indToVec[charToInd[k]] for k in charToInd.keys()}

    return indToVec , indToChar , charToInd , charToVec


