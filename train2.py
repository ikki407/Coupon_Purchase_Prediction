'''
xgboost
'''
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
coupon_area_train = pd.read_csv('data/coupon_area_train.csv')#the coupon listing area for the training set coupons（クーポンのエリア）
coupon_area_test = pd.read_csv('data/coupon_area_test.csv')#the coupon listing area for the testing set coupons（クーポンのエリア）

coupon_list_train = pd.read_csv('data/coupon_list_train.csv')#the master list of coupons（訓練クーポンのマスター）
coupon_list_test = pd.read_csv('data/coupon_list_test.csv')#the master list of coupons.predictions should be sourced only from these 310 coupons（テストクーポンのマスター）

coupon_visit_train = pd.read_csv('data/coupon_visit_train.csv')#the viewing log of users browsing coupons during the training set time period. not provided this table for the test set period（クーポンの閲覧ログ）
coupon_detail_train = pd.read_csv('data/coupon_detail_train.csv')#the purchase log of users buying coupons during the training set time period. not provided this table for the test set period（クーポンの購入ログ）

user_list = pd.read_csv('data/user_list.csv')#the master list of users in the dataset（ユーザーのリスト、予測すべきユーザー）
sample_submission = pd.read_csv('sample_submission.csv')

coupon_list_train.loc[(coupon_list_train[u'CAPSULE_TEXT']=='ビューティ').values,u'CAPSULE_TEXT'] = 'ビューティー'

#coupon_visit_train[u'ITEM_COUNT'] = coupon_visit_train.reset_index().merge(coupon_detail_train,on=u'PURCHASEID_hash',how='left').sort('index').drop('index',axis=1)[u'ITEM_COUNT'].fillna(0).values


#trainデータを分析可能となるように加工

#trainデータ
label = [u'REG_DATE', u'SEX_ID', u'AGE', u'WITHDRAW_DATE', u'PREF_NAME', u'USER_ID_hash',u'VIEW_COUPON_ID_hash',u'PURCHASE_FLG']#,u'ITEM_COUNT'

train = user_list.merge(coupon_visit_train,on=u'USER_ID_hash',how='outer')[label]
train = train.merge(coupon_list_train,left_on=u'VIEW_COUPON_ID_hash',right_on=u'COUPON_ID_hash',how='left')
train = train.ix[train['COUPON_ID_hash'].dropna().index]
train.index = range(0,len(train))
del train[u'VIEW_COUPON_ID_hash']

train = train.sort('REG_DATE')

f_date = lambda x: datetime.date(int(x[0:4]),int(x[5:7]),int(x[8:10])).toordinal() - 733973 if type(x) == str else np.nan
train[u'REG_DATE'] = train[u'REG_DATE'].apply(f_date)
train[u'WITHDRAW_DATE'] = train[u'WITHDRAW_DATE'].apply(f_date)
train[u'DISPFROM'] = train[u'DISPFROM'].apply(f_date)
train[u'DISPEND'] = train[u'DISPEND'].apply(f_date)
train[u'VALIDFROM'] = train[u'VALIDFROM'].apply(f_date)
train[u'VALIDEND'] = train[u'VALIDEND'].apply(f_date)

label_cat = [ u'SEX_ID', u'PREF_NAME', u'CAPSULE_TEXT', u'GENRE_NAME', u'USABLE_DATE_MON', u'USABLE_DATE_TUE', u'USABLE_DATE_WED', u'USABLE_DATE_THU', u'USABLE_DATE_FRI', u'USABLE_DATE_SAT', u'USABLE_DATE_SUN', u'USABLE_DATE_HOLIDAY', u'USABLE_DATE_BEFORE_HOLIDAY',u'large_area_name', u'ken_name', u'small_area_name']

for i in label_cat:
    print len(train[i].value_counts())


for i in label_cat:
    if True in train[i].isnull().values:
        b = categorical(np.array(train[i]), drop=True)[:,:-1]
    else:
        b = categorical(np.array(train[i]), drop=True)
    b = pd.DataFrame(b)
    train = pd.concat([train,b],axis=1)

#購入したか、しなかったか
train = train.sort([u'USER_ID_hash',u'PURCHASE_FLG'],ascending=False).drop_duplicates([u'USER_ID_hash',u'COUPON_ID_hash'])
train[[u'USER_ID_hash',u'PURCHASE_FLG',u'COUPON_ID_hash']]





train.iloc[(train['REG_DATE'].apply(lambda x: int(x[5:])<698)).values,:][train.columns.drop(u'PURCHASE_FLG')],train.iloc[(train['REG_DATE'].apply(lambda x: int(x[5:])<698)).values,:][u'PURCHASE_FLG'].values

train.iloc[(train['REG_DATE'].apply(lambda x: int(x[5:])>=698)).values,:][train.columns.drop(u'PURCHASE_FLG')],train.iloc[(train['REG_DATE'].apply(lambda x: int(x[5:])>=698)).values,:][u'PURCHASE_FLG'].values
