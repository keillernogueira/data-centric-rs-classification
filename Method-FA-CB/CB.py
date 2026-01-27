# Potsdam keep uniform distribution
import os
import sys
from PIL import Image
from tqdm import tqdm
import pandas as pd
from typing import (
    List,Tuple
)
import numpy as np
import pandas as pd
import csv
import timeit

df = pd.read_csv('potsdam_class_distribution.csv')
indexesInOrder = []
indexesonlyoneclass = []
classDist = df.iloc[:,5:]/65536
numPixels = 65536
classDist = classDist.reset_index().drop(['index'], 1)
classDistOrg = classDist.copy()
refDis = pd.DataFrame(columns = classDist.columns)
unifDis = [1/6]*6

KL = []
N = 28800 # change this to total number of images
i  = 0
start = timeit.default_timer()
while i < N:
    if i != 0:
        temp = (classDist + np.sum(refDis, 0))/(len(refDis)+1)
    else:
        temp = classDist.copy()
    classDist['KL'] = np.sum(temp*np.log(temp/unifDis), 1)
    temp2 = classDist[classDist.KL == classDist.KL.min()]
    KL.append(classDist.KL.min())
    bestIdx = temp2.index[0]
    refDis = refDis.append(temp2.iloc[0, :].drop(['KL'], 0))
    indexesInOrder.append(bestIdx)
    classDist = classDist.drop(index=bestIdx)
    classDist = classDist.drop(columns=['KL'])
    i += 1
stop = timeit.default_timer()
print('Time: ', stop - start) 

df2 = classDistOrg.loc[indexesInOrder, :]

ordered_images = []
for i in range(len(df2)):
    index = df2.iloc[i, :].name
    str_temp=""
    for j in range(5):
        str_temp += str(df.iloc[index, j]) + " "
    name  = str_temp + str(1-(i/len(df2)))
    print(name)
    ordered_images.append([name])
    
    
with open('potsdam_keepclassbalance.csv', 'w') as f:
    # using csv.writer method from CSV package
    write = csv.writer(f)
    write.writerows(ordered_images)
