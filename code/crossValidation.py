# -*- coding: utf-8 -*-
from calculations import Prediction
from data_frame import DfToDict
import matplotlib.pyplot as plt

def crossValidationSplit(trainData,split_idx):
    datalen=len(trainData)
    datalen = int(datalen/10)
    startidx=(split_idx-1)*datalen
    endidx = split_idx*datalen
    train1 = trainData[0:startidx]
    train2 = trainData[endidx:len(trainData)]
    test = trainData[startidx:endidx]
    train = train1.append(train2)

    return train,test
    




def crossValidation(kvalue,trainData):
    cosx,cosy=[],[]
    wcosx,wcosy=[],[]
    adjcosx,adjcosy=[],[]
    wadjcosx,wadjcosy=[],[]
    corx,cory=[],[]
    wcorx,wcory=[],[]
    for i in range(1,kvalue,2):
        
        cosinemean=0
        cosinewmean=0
        adjcosinemean=0
        adjcosinewmean=0
        correlationmean=0
        correlationwmean=0
        for j in range(1,11):
            #
            print("Test Part:",i,j)
            part1,part2 = crossValidationSplit(trainData,j)
            TrainDict,TrainBookDict=DfToDict(part1,True,True)
            TestDict,TestBookDict=DfToDict(part2,False,True)
            PredictionControlDict,_=DfToDict(part2,False,False)
            
            cosmean,cosweightedmean=Prediction(TrainDict,TestDict,TrainBookDict,i,"cosine",PredictionControlDict)
            adjmean,adjweightedmean=Prediction(TrainDict,TestDict,TrainBookDict,i,"adjcosine",PredictionControlDict)
            cormean,corweightedmean=Prediction(TrainDict,TestDict,TrainBookDict,i,"correlation",PredictionControlDict)
            
            cosinewmean+=cosweightedmean
            adjcosinewmean+=adjweightedmean
            correlationwmean+=corweightedmean
            
            cosinemean+=cosmean
            adjcosinemean+=adjmean
            correlationmean+=cormean
            
        cosx.append(i),wcosx.append(i),adjcosx.append(i),wadjcosx.append(i),corx.append(i),wcorx.append(i)
        cosinewmean=cosinewmean/10
        adjcosinewmean= adjcosinewmean/10
        correlationwmean=correlationwmean/10
        
        cosinemean=cosinemean/10
        adjcosinemean=adjcosinemean/10
        correlationmean=correlationmean/10
        
        cosy.append(cosinemean)
        wcosy.append(cosinewmean)
        adjcosy.append(adjcosinemean)
        wadjcosy.append(adjcosinewmean)
        cory.append(correlationmean)
        wcory.append(correlationwmean)
      
       
       
        """print("KNN =",i)
        print("COSİNE MEAN ABSOLUTE:",cosinemean)
        print("COSİNE WMEAN ABSOLUTE:",cosinewmean)
        
        print("ADJCOSİNE MEAN ABSOLUTE:",adjcosinemean)
        print("ADJCOSİNE WMEAN ABSOLUTE:",adjcosinewmean)
        
        print("CORRELATİON MEAN ABSOLUTE:",correlationmean)
        print("CORRELATİON WMEAN ABSOLUTE:",correlationwmean)"""
        
    plt.title("k cross validation")
    plt.xlabel("K")
    plt.ylabel("mae")
    
    
    plt.plot(cosx, cosy, color="blue", label="cosine",linewidth=1, linestyle="-")
    plt.plot(wcosx, wcosy, color="green"  ,label="cosine weighted",linewidth=1, linestyle="--")
    plt.plot(adjcosx,adjcosy, color="red", label="adj-cosine",linewidth=1, linestyle="-")
    plt.plot(wadjcosx, wadjcosy, color="brown" ,label="adj-cosine weighted",linewidth=1, linestyle="--")
    plt.plot(corx, cory, color="orange", label="correlation",linewidth=1, linestyle="-")
    plt.plot(wcorx, wcory, color="black",  label="correlation weighted",linewidth=1, linestyle="--")
    
    plt.legend(loc='lower left')

    plt.show()