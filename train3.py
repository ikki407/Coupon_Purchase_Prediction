import sys
import pandas as pd
import numpy as np
import os
import hashlib
import argparse, csv, sys, pickle, collections, math
import pandas as pd
import numpy as np
from sklearn.metrics import roc_curve  
from sklearn.metrics import auc  
import numpy as np
import matplotlib.pyplot as plt
from scikits.statsmodels.tools import categorical
import datetime

#set path
sys.path.append('/Users/IkkiTanaka/Documents/kaggle/Coupon_Purchase_Prediction')
sys.path.append('/Users/IkkiTanaka/Documents/kaggle/Coupon_Purchase_Prediction/script')

#set directory
os.chdir('/Users/IkkiTanaka/Documents/kaggle/Coupon_Purchase_Prediction')

#read data
cpatr = pd.read_csv('data/coupon_area_train.csv')#the coupon listing area for the training set coupons（クーポンのエリア）
cpate = pd.read_csv('data/coupon_area_test.csv')#the coupon listing area for the testing set coupons（クーポンのエリア）

cpltr = pd.read_csv('data/coupon_list_train.csv')#the master list of coupons（訓練クーポンのマスター）
cplte = pd.read_csv('data/coupon_list_test.csv')#the master list of coupons.predictions should be sourced only from these 310 coupons（テストクーポンのマスター）

cpvtr = pd.read_csv('data/coupon_visit_train.csv')#the viewing log of users browsing coupons during the training set time period. not provided this table for the test set period（クーポンの閲覧ログ）
cpdtr = pd.read_csv('data/coupon_detail_train.csv')#the purchase log of users buying coupons during the training set time period. not provided this table for the test set period（クーポンの購入ログ）

ulist = pd.read_csv('data/user_list.csv')#the master list of users in the dataset（ユーザーのリスト、予測すべきユーザー）
ss = pd.read_csv('sample_submission.csv')

cpltr.loc[(cpltr[u'CAPSULE_TEXT']=='ビューティ').values,u'CAPSULE_TEXT'] = 'ビューティー'




#trainデータ作成
train = cpdtr.reset_index().merge(cpltr,on=u'COUPON_ID_hash',how='inner').sort('index').drop('index',axis=1)


#時間見ない
train = train[["COUPON_ID_hash","USER_ID_hash","GENRE_NAME","DISCOUNT_PRICE","PRICE_RATE","USABLE_DATE_MON","USABLE_DATE_TUE","USABLE_DATE_WED","USABLE_DATE_THU","USABLE_DATE_FRI","USABLE_DATE_SAT","USABLE_DATE_SUN","USABLE_DATE_HOLIDAY","USABLE_DATE_BEFORE_HOLIDAY","ken_name","small_area_name",u'I_DATE']]

#TestとTrainを結合
cplte['USER_ID_hash'] = 'dummyuser'
cpchar = cplte[["COUPON_ID_hash","USER_ID_hash","GENRE_NAME","DISCOUNT_PRICE","PRICE_RATE","USABLE_DATE_MON","USABLE_DATE_TUE","USABLE_DATE_WED","USABLE_DATE_THU","USABLE_DATE_FRI","USABLE_DATE_SAT","USABLE_DATE_SUN","USABLE_DATE_HOLIDAY","USABLE_DATE_BEFORE_HOLIDAY","ken_name","small_area_name"]]
train = pd.concat([train,cpchar])
train.index = range(0,len(train))

#欠損値補間
train = train.fillna(1)

#Feature Engineering
#train['DISCOUNT_PRICE'] = 1 / np.log10(train['DISCOUNT_PRICE'])
#割引が大きいほど小さい

#train['DISCOUNT_PRICE'] = train['DISCOUNT_PRICE'].apply(lambda x: 1 if x==0 else x)
#train['DISCOUNT_PRICE'] = np.log(train['DISCOUNT_PRICE'])

train['PRICE_RATE'] = (train['PRICE_RATE'] * train['PRICE_RATE']) / (100*100)

#object型のカラムを抽出しダミー変数化
object_col = train.iloc[:,2:].iloc[:,(train.dtypes[2:]==object).values]
for i in object_col.columns.drop(['USER_ID_hash','I_DATE']):
    a = categorical(np.array(train[i]),drop=True)
    colname = i + pd.Series(range(0,len(train[i].value_counts()))).astype(str)
    a = pd.DataFrame(a,columns=colname)
    train = pd.concat([train,a],axis=1)
    print a.shape
    del train[i]



#割引価格を単位（円）でダミー変数化する

digit_dic = {'DP0':pd.Series([2,1],index=['DP0','DP1']),'DP1':pd.Series([1,2,1],index=['DP0','DP1','DP10']),'DP10':pd.Series([1,2,1],index=['DP1','DP10','DP100']),'DP100':pd.Series([1,2,1],index=['DP10','DP100','DP1000']),'DP1000':pd.Series([1,2,1],index=['DP100','DP1000','DP10000']),'DP10000':pd.Series([1,2,1],index=['DP1000','DP10000','DP100000']),'DP100000':pd.Series([1,2],index=['DP10000','DP100000'])}
def discount_price_dummy(num):
    if num == 0:
        return digit_dic['DP0']
    
    digit = int(math.log10(num) + 1)#桁数
    if digit == 1:
        return digit_dic['DP1']
    elif digit == 2:
        return digit_dic['DP10']
    elif digit == 3:
        return digit_dic['DP100']
    elif digit == 4:
        return digit_dic['DP1000']
    elif digit == 5:
        return digit_dic['DP10000']
    elif digit == 6:
        return digit_dic['DP100000']

train[['DP0','DP1','DP10','DP100','DP1000','DP10000','DP100000']] = train['DISCOUNT_PRICE'].apply(discount_price_dummy) / 100.

train['DISCOUNT_PRICE'] = 1 / np.log10(train['DISCOUNT_PRICE'])
#del train['DISCOUNT_PRICE']
train = train.fillna(0)
#testとtrainを分割
test = train.iloc[(train['USER_ID_hash']=='dummyuser').values,:]
test = test[test.columns.drop('USER_ID_hash')]
train = train.iloc[(train['USER_ID_hash']!='dummyuser').values,:]
test.index = range(0,len(test))

f_date = lambda x: datetime.date(int(x[0:4]),int(x[5:7]),int(x[8:10])).toordinal() - 734318
train[u'I_DATE'] = train[u'I_DATE'].apply(f_date)
#20120501より後
train = train.iloc[(train[u'I_DATE']>=(datetime.datetime(2012,05,01).toordinal() - 734318)).values,:]

#20120623より前
train = train.iloc[((datetime.datetime(2012,06,23).toordinal()- 734318)>=train[u'I_DATE']).values,:]
train[u'I_DATE'] = train[u'I_DATE'] - train[u'I_DATE'].min() +1
train[u'I_DATE'] = np.log10(train[u'I_DATE']/10)

#userの特徴
uchar = train.groupby('USER_ID_hash').mean()
#uchar['DISCOUNT_PRICE'] = 1
#uchar['PRICE_RATE'] = 1
uchar['USER_ID_hash'] = uchar.index
uchar.index = range(0,len(uchar))
print len(uchar)
'''
#重み行列 DISCOUNT_PRICE PRICE_RATE USABLE_DATE_ GENRE_NAME ken_name small_area_name DP_
x1 = np.array([1,0,0,3,3,5,1])
x2 = np.repeat(x1, [1,1,9,13,47,55,7], axis=0)

W = np.matrix(np.diag(x2))
'''
#重み行列  DISCOUNT_PRICE u'I_DATE' PRICE_RATE USABLE_DATE_ GENRE_NAME ken_name small_area_name DP_ 
x1 = np.array([1,3,0,0,3,3,4,0.5])
x2 = np.repeat(x1, [1,1,1,9,13,47,55,7], axis=0)

W = np.matrix(np.diag(x2))


#ユーザとクーポンのコサイン類似度計算
score = np.matrix(uchar.iloc[:,:-1]).dot(W).dot(np.matrix(test.iloc[:,1:]).T)

#類似度に従ってクーポンをソートし、上位１０個を取る
def top10(data,score):
    #print data.index[0]
    top10_cp = test['COUPON_ID_hash'].ix[score[data.index[0]].argsort().tolist()[0][::-1][:10]]
    top10_cp = ' '.join(top10_cp)
    return top10_cp

uchar['PURCHASED_COUPONS'] = uchar.groupby('USER_ID_hash').apply(top10,score).values


#保存
ulist = ulist.sort("USER_ID_hash")
submission = ulist.merge(uchar,on=u'USER_ID_hash',how='left')
submission = submission[["USER_ID_hash","PURCHASED_COUPONS"]]
submission["PURCHASED_COUPONS"] = submission["PURCHASED_COUPONS"].fillna('')
submission.to_csv('/Users/IkkiTanaka/Documents/kaggle/Coupon_Purchase_Prediction/prediction/simple_pred1.csv',index=False)


