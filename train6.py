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


#trainデータに存在しないtest user
#ulist.iloc[(ulist['USER_ID_hash'].isin(train.drop_duplicates('USER_ID_hash')['USER_ID_hash'].values)).values,:]


#trainデータ作成
train = cpdtr.reset_index().merge(cpltr,on=u'COUPON_ID_hash',how='inner').sort('index').drop('index',axis=1)


#時間見ない
train = train[["COUPON_ID_hash","USER_ID_hash","GENRE_NAME","DISCOUNT_PRICE","PRICE_RATE","USABLE_DATE_MON","USABLE_DATE_TUE","USABLE_DATE_WED","USABLE_DATE_THU","USABLE_DATE_FRI","USABLE_DATE_SAT","USABLE_DATE_SUN","USABLE_DATE_HOLIDAY","USABLE_DATE_BEFORE_HOLIDAY","ken_name","small_area_name",u'I_DATE']]

#TestとTrainを結合
cplte['USER_ID_hash'] = 'dummyuser'
'''
#cplteの1週間の始め3日間に表示されるクーポンのみを推薦する
'''
#cplte = cplte.iloc[(cplte['DISPFROM'] < '2012-06-27 00:00:00').values,:] 

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


'''
#割引価格を単位（円）でダミー変数化する

digit_dic = {'DP0':pd.Series([1],index=['DP0']),'DP1':pd.Series([1],index=['DP1']),'DP10':pd.Series([1],index=['DP10']),'DP100':pd.Series([1],index=['DP100']),'DP1000':pd.Series([1],index=['DP1000']),'DP10000':pd.Series([1],index=['DP10000']),'DP100000':pd.Series([1],index=['DP100000'])}
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
'''
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
test[u'I_DATE'] = test[u'I_DATE'].apply(f_date)

#train, testのベクトルに忘却率と割引率をそれぞれかける
train[u'forget_factor'] = train[u'I_DATE'].apply(lambda x: (0.99)**(train[u'I_DATE'].max()-x))

#ユーザー間の類似度計算用
uchar2 = train.groupby('USER_ID_hash').mean()
del uchar2['I_DATE']
'''
#20120501より後
'''
#train = train.iloc[(train[u'I_DATE']>=(datetime.datetime(2012,05,01).toordinal() - 734318)).values,:]

#20120623より前
train = train.iloc[((datetime.datetime(2012,06,23).toordinal()- 734318)>=train[u'I_DATE']).values,:]
train[u'I_DATE'] = train[u'I_DATE'] - train[u'I_DATE'].min() +1
train[u'I_DATE'] = np.log10(train[u'I_DATE']/10)

del train['I_DATE'], test['I_DATE']

'''
#logに現れた回数が閾値以上の人のみを抽出
'''
print train['USER_ID_hash'].value_counts()
user_count = train['USER_ID_hash'].value_counts()
user_count = pd.DataFrame(user_count,columns=['user_count'])
user_count['USER_ID_hash'] = user_count.index
user_count.index = range(0,len(user_count))

train = train.merge(user_count,on='USER_ID_hash',how='left')
train['user_count'] = train['user_count'].fillna(0)
train = train.iloc[(train['user_count'] > 10).values,:]#10回以上

del train['user_count']


#userの特徴
uchar = train.groupby('USER_ID_hash').mean()
#uchar['DISCOUNT_PRICE'] = 1
#uchar['PRICE_RATE'] = 1
uchar['USER_ID_hash'] = uchar.index
uchar.index = range(0,len(uchar))
print len(uchar)



#重み行列 DISCOUNT_PRICE PRICE_RATE USABLE_DATE_ GENRE_NAME ken_name small_area_name
x1 = np.array([1,0,0,3,3,4])
x2 = np.repeat(x1, [1,1,9,13,47,55], axis=0)

W = np.matrix(np.diag(x2))
'''
#重み行列  DISCOUNT_PRICE u'I_DATE' PRICE_RATE USABLE_DATE_ GENRE_NAME ken_name small_area_name DP_ 
x1 = np.array([1,3,0,0,3,3,4,0.5])
x2 = np.repeat(x1, [1,1,1,9,13,47,55,7], axis=0)

W = np.matrix(np.diag(x2))
'''


#ユーザとクーポンのコサイン類似度計算
#ユーザとユーザの類似度計算
uchar2['USER_ID_hash'] = uchar2.index
#score_u = np.matrix(uchar2.iloc[:,:-1]).dot(W).dot(np.matrix(uchar2.iloc[:,:-1]).T)
score_u = np.matrix((uchar2.iloc[:,:-1].T/((uchar2.iloc[:,:-1]*uchar2.iloc[:,:-1]).sum(1)**0.5)).T).dot(np.matrix((uchar2.iloc[:,:-1].T/((uchar2.iloc[:,:-1]*uchar2.iloc[:,:-1]).sum(1)**0.5)).T).T)
np.fill_diagonal(score_u, 0)#対角成分0
num_user = 50


uchar2_user = uchar2.index
uchar2.index = range(0,len(uchar2))
def top10_user(data,score=score_u):
    if data.index[0]%1000 == 0:
        print data.index[0]
    top10_cp = uchar2_user[score[data.index[0]].argsort().tolist()[0][::-1][:num_user]]
    #top10_cp = ' '.join(top10_cp)
    #print ','.join(top10_cp.tolist)
    return pd.Series([','.join(top10_cp.tolist())])

uchar2['similar_user'] = uchar2.groupby('USER_ID_hash').apply(top10_user,score_u).values



#ユーザとクーポンのコサイン類似度計算
score = np.matrix((uchar.iloc[:,:-1].T/((uchar.iloc[:,:-1]*uchar.iloc[:,:-1]).sum(1)**0.5)).T).dot(np.matrix((test.iloc[:,1:].T/((test.iloc[:,1:]*test.iloc[:,1:]).sum(1)**0.5)).T).T)


#類似度に従ってクーポンをソートし、上位１０個を取る
def top10(data,score):
    #print data.index[0]
    top10_cp = test['COUPON_ID_hash'].ix[score[data.index[0]].argsort().tolist()[0][::-1][:10]]
    top10_cp = ' '.join(top10_cp)
    #print top10_cp
    return top10_cp

uchar['PURCHASED_COUPONS'] = uchar.groupby('USER_ID_hash').apply(top10,score).values

'''
ulist = ulist.sort("USER_ID_hash")
submission = ulist.merge(uchar,on=u'USER_ID_hash',how='left')
submission = submission[["USER_ID_hash","PURCHASED_COUPONS"]]
submission["PURCHASED_COUPONS"] = submission["PURCHASED_COUPONS"].fillna('')
submission.to_csv('/Users/IkkiTanaka/Documents/kaggle/Coupon_Purchase_Prediction/prediction/simple_pred1.csv',index=False)
'''



submission = uchar2[['USER_ID_hash','similar_user']].merge(uchar[['USER_ID_hash','PURCHASED_COUPONS']],on='USER_ID_hash',how='left')


#似ているユーザのクーポンを推薦する
def sim_coupon(data):
    #print data.Name
    data = data['similar_user'].split(',')
    for i in range(0,num_user):
        #print type(submission.iloc[(submission['USER_ID_hash']==data[i]).values,:]['PURCHASED_COUPONS'].values[0])
        if type(submission.iloc[(submission['USER_ID_hash']==data[i]).values,:]['PURCHASED_COUPONS'].values[0]) == str:
            return pd.Series(submission.iloc[(submission['USER_ID_hash']==data[i]).values,:]['PURCHASED_COUPONS'].values,index=['PURCHASED_COUPONS'])
    return ''


submission.loc[(submission['PURCHASED_COUPONS'].isnull().values),['PURCHASED_COUPONS']] = submission[(submission['PURCHASED_COUPONS'].isnull().values)].apply(sim_coupon,axis=1).values

submission = submission[["USER_ID_hash","PURCHASED_COUPONS"]]
submission = ulist.merge(submission,on=u'USER_ID_hash',how='left')
submission = submission[["USER_ID_hash","PURCHASED_COUPONS"]]
submission = submission.sort("USER_ID_hash")

#trainに存在しないユーザーに人気のクーポンを推薦する
#0a6e889b43a220b23a68f325a6a4cdfc
def popular_cp(data):
    return pd.Series(data.split(' '))

pop_cp = submission["PURCHASED_COUPONS"].dropna().apply(popular_cp).dropna()
submission["PURCHASED_COUPONS"] = submission["PURCHASED_COUPONS"].apply(lambda x: ' '.join(pop_cp[0].value_counts()[:10].index) if x == '' else x)

submission = submission.sort("USER_ID_hash")
submission.to_csv('/Users/IkkiTanaka/Documents/kaggle/Coupon_Purchase_Prediction/prediction/simple_pred1.csv',index=False)






pd.DataFrame(submission[:22873],columns=['USER_ID_hash'])
user_cp = uchar[['USER_ID_hash','PURCHASED_COUPONS']]

#似ているユーザーのクーポンを推薦する
def sim_u_cp(data,score=score_u,us_cp=user_cp):
    #print data['similar_user'].values
    a = data['similar_user'].values[0].split(' ')
    b = []
    for i in a:
        b.append(us_cp.iloc[(us_cp['USER_ID_hash']==i).values,:]['PURCHASED_COUPONS'].values[0].split(' ')[0])
    return ' '.join(b)

uchar['PURCHASED_COUPONS_sim_user'] = uchar.head(5).groupby('USER_ID_hash').apply(sim_u_cp,score_u).values


#保存
ulist = ulist.sort("USER_ID_hash")
submission = ulist.merge(uchar,on=u'USER_ID_hash',how='left')
submission = submission[["USER_ID_hash","PURCHASED_COUPONS"]]
submission["PURCHASED_COUPONS"] = submission["PURCHASED_COUPONS"].fillna('')
submission.to_csv('/Users/IkkiTanaka/Documents/kaggle/Coupon_Purchase_Prediction/prediction/simple_pred1.csv',index=False)




