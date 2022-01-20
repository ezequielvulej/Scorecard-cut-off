# -*- coding: utf-8 -*-
"""
Created on Thu Jan 13 23:05:18 2022

@author: evule
"""
import pandas as pd
import numpy as np


good=[2725,2808,3064,3979,4503,5127,2045,1753,1598,2052,1787,2749]
bad=[312,214,239,263,311,291,94,66,63,84,86,82]
idx=['1-149','150-166','167-179','180-191','192-202','203-214','215-219',
        '220-224','225-229','230-237','238-247','248-HI']
df=pd.DataFrame(zip(good,bad),columns=['good','bad'], index=idx)

df['cut']=df.index.map(lambda x: x[0:x.find('-')])

df['tn']=df['good'].sum()-np.cumsum(df['good']).shift(1)
df['tn']=df['tn'].fillna(df['good'].sum())

df['fn']=df['bad'].sum()-np.cumsum(df['bad']).shift(1)
df['fn']=df['fn'].fillna(df['bad'].sum())

df['fp']=np.cumsum(df['good']).shift(1)
df['fp']=df['fp'].fillna(0)

df['tp']=np.cumsum(df['bad']).shift(1)
df['tp']=df['tp'].fillna(0)

P=df['bad'].sum()
N=df['good'].sum()

df['tpr']=df['tp']/P
df['fpr']=df['fp']/N
df['J']=df['tpr']-df['fpr']

optimalCut=df.cut[np.argmax((df['J']))]

#df.to_excel("Salida2.xls")
#print('The optimal cut-off is {}'.format(optimalCut))


#Relative impact
indeterminate=[245,165,163,201,248,245,66,62,40,53,41,34]
reject=[1537,990,982,1129,1064,952,345,287,240,273,170,210]

df['indeterminate']=indeterminate
df['reject']=reject
df['badRate']=df['bad']/(df['good']+df['bad']+df['indeterminate'])
#print(df)

currentAccepted=df['good'].sum()+df['bad'].sum()+df['indeterminate'].sum()
currentRejected=df['reject'].sum()
currentAcceptanceRate=currentAccepted/(currentAccepted+currentRejected)
currentBadRate=df['bad'].sum()/(df['good'].sum()+df['bad'].sum())

proposedAccepted=df[df['cut']>=optimalCut].filter(['good','bad','indeterminate','reject']).sum().sum()
proposedRejected=df[df['cut']<optimalCut].filter(['good','bad','indeterminate','reject']).sum().sum()
proposedAcceptanceRate=proposedAccepted/(proposedAccepted+proposedRejected)
factor=1.25
proposedBad=df[df['cut']>=optimalCut]['bad'].sum()+np.multiply\
    (df[df['cut']>=optimalCut]['reject'],df[df['cut']>=optimalCut]['badRate']).sum()*factor
proposedBadRate=proposedBad/proposedAccepted

print('The acceptance rate would go from {:.2%} to {:.2%}, which represents a \
relative variation of {:.2%}'.format(currentAcceptanceRate,proposedAcceptanceRate,\
proposedAcceptanceRate/currentAcceptanceRate-1))

print('The bad rate would go from {:.2%} to {:.2%}, which represents a \
relative variation of {:.2%}'.format(currentBadRate,proposedBadRate,\
proposedBadRate/currentBadRate-1))

    
#Discrimination power
df['good+bad']=df['good']+df['bad']
df['currentCDt']=df['good+bad'].cumsum()/df['good+bad'].sum()
df['currentCDd']=df['bad'].cumsum()/df['bad'].sum()
df['perfectd']=np.minimum(df['good+bad'],df['bad'].sum())
df.loc[df['perfectd'].cumsum()>df['bad'].sum(),'perfectd']=0
df['perfectCDd']=df['perfectd'].cumsum()/df['bad'].sum()
df['randomCDd']=df['currentCDt']

B=((df['currentCDt']-df['currentCDt'].shift(1).fillna(0))\
   *(df['currentCDd']+df['currentCDd'].shift(1).fillna(0)))\
    .sum()/2-0.5

AplusB=0.5*(1-currentBadRate)
currentAR=B/AplusB

print('The Accuracy Ratio for the taken up population is {:.8%}'.format(currentAR))

#KS
df['currentCDnd']=df['good'].cumsum()/df['good'].sum()
df['KS']=abs(df['currentCDd']-df['currentCDnd'])
KS=max(df['KS'])
print('The KS for the taken up population is {:.2%}'.format(KS))


