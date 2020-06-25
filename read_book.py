import numpy as np

def getData(fileName):

    f = open(fileName,'r',encoding='utf-8')
    bookData = f.read()
    f.close()

    bookChars = set(bookData)

    return bookData , bookChars