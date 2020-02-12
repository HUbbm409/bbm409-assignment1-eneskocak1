
# -*- coding: utf-8 -*-
from data_frame import reader,filtering,renamecolumn,combine,DfToDict,filterDrop
from calculations import Prediction
from crossValidation  import crossValidation

import matplotlib.pyplot as plt
import time


file = "C:\\Users\\enes\\Desktop\\data\\BX-Users.csv" # UserData
file2 = "C:\\Users\\enes\\Desktop\\data\\BX-Book-Ratings-Train.csv" #TrainData
file3 = "C:\\Users\\enes\\Desktop\\data\\BX-Books.csv"#BookData
file4 = "C:\\Users\\enes\\Desktop\\data\\BXBookRatingsTest.csv" # TestData

UserDataFrame = filtering(renamecolumn(reader(file),'UserId','Location','Age')) # usa and canada filtered dataframe
BooksDataFrame = reader(file3)

TestDataFrame = renamecolumn(reader(file4),'UserId','ISBN','Rating')
RatingsDataFrame = renamecolumn(reader(file2),'UserId','ISBN','Rating')

combined = combine(BooksDataFrame,RatingsDataFrame,'ISBN')
combin = combine(combined,UserDataFrame,'UserId')
combin = filterDrop(combin,"UserId","ISBN","Rating")


combinedtest = combine(BooksDataFrame,TestDataFrame,'ISBN')
combintest = combine(combinedtest,UserDataFrame,'UserId')
combintest = filterDrop(combintest,"UserId","ISBN","Rating")


#crossValidation(50,combin) #this function for cross validaation


TrainDict,TrainBookDict=DfToDict(combin,True,True)
TestDict,_=DfToDict(combintest,False,True)
PredictionControlDict,_=DfToDict(combintest,False,False)


start = time.time()
#Prediction(TrainDict,TestDict,TrainBookDict,45,"cosine",PredictionControlDict)# for cosine
Prediction(TrainDict,TestDict,TrainBookDict,45,"adjcosine",PredictionControlDict)# for adjcosine
#Prediction(TrainDict,TestDict,TrainBookDict,45,"correlation",PredictionControlDict)#for correlation
end=time.time()
result=end-start
print("Run time: ",result,"\n------------")
    


#print("for control",PredictionControlDict) #if you want to control results turn on this row
    

