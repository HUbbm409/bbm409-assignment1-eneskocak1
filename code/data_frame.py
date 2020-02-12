# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np

def reader(filepath):
    df = pd.read_csv(filepath,sep=";",encoding="latin-1",error_bad_lines=False,warn_bad_lines=False,low_memory=False)
    
    return df

def filtering(filterDb):
    filtered = filterDb[(filterDb.Location.str.contains("usa")) | (filterDb.Location.str.contains("canada"))]
    return filtered

def renamecolumn(DataFrame,column1,column2,column3):
    DataFrame.columns=[column1,column2,column3]
    return DataFrame


def DfToDict(DataFrame,mode,mode2):
    combinDict = {}
    BookDict={}
    combinlist = DataFrame['UserId'].tolist()
    isbnlist = DataFrame['ISBN'].tolist()
    ratelist = DataFrame['Rating'].tolist()
    totalLen = len(combinlist)
    combinlist = [int(x) for x in combinlist]
    

    for i in range(totalLen):
       # combinDict.update({combinlist[i]:{}})
        if mode2==True:
            control = combinDict.setdefault(combinlist[i],{isbnlist[i]:ratelist[i]})
            if control != {isbnlist[i]:ratelist[i]} :
                control.update({isbnlist[i]:ratelist[i]})
        if mode2==False:
            control = combinDict.setdefault(combinlist[i],{isbnlist[i]:[ratelist[i],None,None]})
            if control != {isbnlist[i]:[ratelist[i],None,None]} :
                control.update({isbnlist[i]:[ratelist[i],None,None]})
        if mode==True:
            control = BookDict.setdefault(isbnlist[i],{combinlist[i]:ratelist[i]})
            if control != {combinlist[i]:ratelist[i]} :
                control.update({combinlist[i]:ratelist[i]})
        
    return combinDict,BookDict

def combine(DataFrame1,DataFrame2,columnname):
    combined = pd.merge(DataFrame1, DataFrame2, on=columnname)
    return combined
    
def filterDrop(data,userid,isbn,rate):
    data = data[[userid,isbn,rate]]
    #data = data.sort_values(by=index)
    return data

