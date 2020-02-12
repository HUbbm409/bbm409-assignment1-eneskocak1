# -*- coding: utf-8 -*-
import numpy as np

def Knn(CosSim_BooksNumpy,number_K,trainBook,predictdict):
    Numpylen =  len(CosSim_BooksNumpy)
    if Numpylen < number_K: # controlling list length 
        number_K = Numpylen 
    
    Keylist=CosSim_BooksNumpy[0][2].keys() # testuser isbn list
    IsbnList=[]# isbns data 
    meanisbn=0
    for k in Keylist: 
        IsbnList.append([k,0,0,0,0,0])# ["isbn","neighbours mean","real rating of testuser","weighted mean","neighbours weight sum","neighbours lengt"]
    for i in range(number_K):
        item = CosSim_BooksNumpy[i][1].keys()
        meanisbn=0 #testuser mean 
        for isbninlist in IsbnList:
            isbninlist[2]= CosSim_BooksNumpy[i][2][isbninlist[0]] # real item rating uploaded
            meanisbn += isbninlist[2] # add item rating for user mean
            if isbninlist[0] in item:
                
                trainmean = sum(CosSim_BooksNumpy[i][1].values())/len(CosSim_BooksNumpy[i][1]) # trainUser mean calculating
    
                isbninlist[1]+=CosSim_BooksNumpy[i][1][isbninlist[0]]-trainmean # neighbour trainusers standart deviation for this item
                
                isbninlist[3]+=((1/(1.1-CosSim_BooksNumpy[i][0]))**2)*(CosSim_BooksNumpy[i][1][isbninlist[0]]-trainmean)#weighted trainusers standart deviation for this item
                
                isbninlist[5] += 1 # how many users rate this book count decreasing
                isbninlist[4] += (1/(1.1-CosSim_BooksNumpy[i][0]))**2# neighbours distance for total weight 
                #print(isbninlist[0],isbninlist[4])
                
                #print(isbninlist[3] ,isbninlist[0])
                    
    MeanAbsolute=0 
    WeightedMeanAbsolute=0
    meanisbn = meanisbn/len(IsbnList) # testuser mean
   
    for isbninlist in IsbnList:
        if isbninlist[5]!=0: # if at least one neighbour rate this book
            isbninlist[1] = round(meanisbn + (isbninlist[1] / isbninlist[5])) # calculate predict value
            
            if isbninlist[1] <0: # predict range will be 0 to 10
                isbninlist[1]=0
            if isbninlist[1]>10: # predict range will be 0 to 10
                isbninlist[1]=10
        
        else: 
            isbninlist[1] = round(meanisbn)# anybody dont rate update predict userMean
            
        if isbninlist[4]!=0: #if at least one neighbour rate this book
            
            isbninlist[3] = round(meanisbn+ (isbninlist[3] / isbninlist[4])) # calculate predict value
            if isbninlist[3] <0: # predict range will be 0 to 10
                isbninlist[3]=0
            if isbninlist[3]>10: # predict range will be 0 to 10
                isbninlist[3]=10
            
        else:
            isbninlist[3] = round(meanisbn) # anybody dont rate update predict userMean
        
        predictdict[isbninlist[0]][1],predictdict[isbninlist[0]][2]=isbninlist[1],isbninlist[3]# update prediction dict
        WeightedMeanAbsolute += abs(isbninlist[3]-isbninlist[2]) # calculating user MAE
        MeanAbsolute += abs(isbninlist[1]-isbninlist[2]) # calculating weighted User MAE
    
    MeanAbsolute=MeanAbsolute/len(IsbnList) # calculating user MAE
    WeightedMeanAbsolute=WeightedMeanAbsolute/len(IsbnList) # calculating weighted User MAE
    IsbnList=np.array(IsbnList)
    """print(IsbnList)
    print("meanisbn",meanisbn)
    print("MEAN:",MeanAbsolute,"weightedMEan:",WeightedMeanAbsolute)
    input("enes")"""
    
    return MeanAbsolute,WeightedMeanAbsolute,len(IsbnList)
