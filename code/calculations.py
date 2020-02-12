 # -*- coding: utf-8 -*-
import numpy as np
from KNN import Knn
import pandas as pd
def cosineSim(TestData,TrainUserData,TrainBookData,TrainSquares):
    
    cossim=list() #smilarity list of testuser
    setuser=set() # intersection trainUser set
    vecttest=0 # sum of squares ratings
    for x,y in TestData.items():
        vecttest += y**2 
        if x in TrainBookData:
            setuser |= set(TrainBookData[x].keys())
            
    vecttest=vecttest**0.5 # root of sum of squares testuser ratings
    for a in setuser:
        dotproduct =0 # sum of common items ratings dots 
        for key,value in TestData.items():
            if key in TrainUserData[a]:
                dotproduct+= value * TrainUserData[a][key] # common item ratings dot
        vecttrain=TrainSquares[a] # root of sum of squares trainuser ratings
        if ( vecttrain !=0) & (vecttest !=0) :
            cossim.append([dotproduct/(vecttrain*vecttest),TrainUserData[a],TestData])#calculate smilarity and append list
        else:
            cossim.append([0,TrainUserData[a],TestData])
    cossim =sorted(cossim,key=lambda l:l[0], reverse=True)
    return cossim
    

def Prediction(trainUser,testUser,trainBook,numberof_k,smilarityname,predictdict):
    MeanAbsolute=0
    predictsum=0
    totalpredict=len(testUser)
    weightedMeanAbsolute=0
    if smilarityname == "cosine":
        TrainSquares=squareforcos(trainUser)
    for testkey in testUser:
        if smilarityname == "cosine":
            cossim = cosineSim(testUser[testkey],trainUser,trainBook,TrainSquares)
        if smilarityname == "correlation":
            cossim = correlation(testUser[testkey],trainUser,trainBook)
        if smilarityname == "adjcosine":
            cossim = adjCosineSim(testUser[testkey],trainUser,trainBook)
        if len(cossim)!=0 :
            #print("\nTEST ID ->",testkey," With K=",numberof_k,"neighbours",len(cossim))
            mean,weightedmean,prediction = Knn(cossim,numberof_k,trainBook,predictdict[testkey])
            MeanAbsolute += mean
            weightedMeanAbsolute += weightedmean
            predictsum += prediction
          
        else: # if test user dont have common item in train data, prediction will be updated users own mean
            lenisbn=len(testUser[testkey])
            predictsum += lenisbn
            testusersum=sum(testUser[testkey].values())/lenisbn
            testMean=0
            testWmean=0
            for key,value in testUser[testkey].items():
                predictdict[testkey][key][1],predictdict[testkey][key][2]=round(testusersum),round(testusersum)
                testMean+=abs(value-testusersum)
                testWmean+=abs(value-testusersum)
            testMean=testMean/lenisbn
            testWmean=testWmean/lenisbn
            
            MeanAbsolute += testMean
            weightedMeanAbsolute += testWmean
            
        
    MeanAbsolute= MeanAbsolute/totalpredict
    weightedMeanAbsolute= weightedMeanAbsolute/totalpredict
    print("\nSmilarity function:",smilarityname,
          "\nNeighbours number ==",numberof_k,
          "\nMEAN ABSOLUTE ERROR:",MeanAbsolute,
          "\nWeighted MEAN ABSOLUTE ERROR:",weightedMeanAbsolute,
          "\nTotal Prediction:",totalpredict)
    
    return MeanAbsolute,weightedMeanAbsolute



def adjCosineSim(TestData,TrainUserData,TrainBookData):
    
    adjcossim=list() #smilarity list of testuser
    setuser=set() #intersection trainUser set
    meantest=sum(TestData.values())/len(TestData) # testuser ratings mean
    
    for x,y in TestData.items():
        if x in TrainBookData:
            setuser |= set(TrainBookData[x].keys())
            
  
    for a in setuser:
        vecttrain=0 # sum of substract common items ratings and trainuser mean
        vecttest=0 # sum of substract common items ratings and testuser mean
        dotproduct =0 # sum of common item rating dots
        meantrain = sum(TrainUserData[a].values())/len(TrainUserData[a])
        for key,value in TestData.items():
            if key in TrainUserData[a]:
               vect1= (value-meantest)
               vect2=(TrainUserData[a][key]-meantrain)
               dotproduct += vect1 *vect2
               vecttest += vect1**2
               vecttrain+= vect2**2
               
        vecttest=vecttest**0.5 # root of sum of squares testuser ratings
        vecttrain=vecttrain**0.5 # root of sum of squares trainuser ratings
        if ( vecttrain !=0) & (vecttest !=0) :
            adjcossim.append([dotproduct/(vecttrain*vecttest),TrainUserData[a],TestData])#calculate smilarity and append list
        else:
            adjcossim.append([0,TrainUserData[a],TestData])
    adjcossim =sorted(adjcossim,key=lambda l:l[0], reverse=True)
    return adjcossim
        
def correlation(TestData,TrainUserData,TrainBookData):
    
    correlation=list() # smilarity list for testuser
    meanitem={} # item means in training data dictionry 
    setuser=set() # intersection trainUser set
    
    for x,y in TestData.items():
        if x in TrainBookData:
            setuser |= set(TrainBookData[x].keys())
            mean=sum(TrainBookData[x].values())/len(TrainBookData[x])
            meanitem.update({x:mean}) # item means updated here
        else:
            meanitem.update({x:0})
            
    for a in setuser:
        vectx =0 #  sum of substract common items ratings and trainusers common itemmean
        vecty=0 # sum of substract common items ratings and trainusers common itemmean
        dotproduct =0
        for key,value in TestData.items():
            if key in TrainUserData[a] :
                vect1= (value-meanitem[key])
                vect2=(TrainUserData[a][key]-meanitem[key])
                dotproduct += vect1 *vect2
                vectx += vect1**2
                vecty+= vect2**2
        vectx=vectx**0.5 # root of sum of squares testuser ratings
        vecty=vecty**0.5 # root of sum of squares trainuser ratings
        if (vectx !=0) & (vecty !=0) :
            correlation.append([dotproduct/(vecty*vectx),TrainUserData[a],TestData])#calculate smilarity and append list
        else:
            correlation.append([0,TrainUserData[a],TestData])
            
    correlation =sorted(correlation,key=lambda l:l[0], reverse=True)
    return correlation


def squareforcos(TrainUsers):
    
    squareddict={}
    for key,value in TrainUsers.items():
        vecttrain = np.array(list(value.values()))
        vecttrain = np.sum(np.square(vecttrain))**0.5
        squareddict[key]=vecttrain
    
    
    return squareddict