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

f_date = lambda x: datetime.date(int(x[0:4]),int(x[5:7]),int(x[8:10])).toordinal() - 733973 if type(x) == str else 'nan'
train[u'REG_DATE'] = train[u'REG_DATE'].apply(f_date)
train[u'WITHDRAW_DATE'] = train[u'WITHDRAW_DATE'].apply(f_date)
train[u'DISPFROM'] = train[u'DISPFROM'].apply(f_date)
train[u'DISPEND'] = train[u'DISPEND'].apply(f_date)
train[u'VALIDFROM'] = train[u'VALIDFROM'].apply(f_date)
train[u'VALIDEND'] = train[u'VALIDEND'].apply(f_date)

#購入したか、しなかったか
train = train.sort([u'USER_ID_hash',u'PURCHASE_FLG'],ascending=False).drop_duplicates([u'USER_ID_hash',u'COUPON_ID_hash'])
train[[u'USER_ID_hash',u'PURCHASE_FLG',u'COUPON_ID_hash']]

#LibFM用のデータに変換

#hash関数
NR_BINS = 1000000
def hashstr(input):
    return str(int(hashlib.md5(input.encode('utf8')).hexdigest(), 16)%(NR_BINS-1)+1)

#convert関数

#Categoricalデータ
'''
User         item
0:1:1        1:3:1
0:2:1        1:4:1


'''
label_cat = [ u'SEX_ID', u'PREF_NAME',u'USER_ID_hash', u'CAPSULE_TEXT', u'GENRE_NAME', u'USABLE_DATE_MON', u'USABLE_DATE_TUE', u'USABLE_DATE_WED', u'USABLE_DATE_THU', u'USABLE_DATE_FRI', u'USABLE_DATE_SAT', u'USABLE_DATE_SUN', u'USABLE_DATE_HOLIDAY', u'USABLE_DATE_BEFORE_HOLIDAY',u'large_area_name', u'ken_name', u'small_area_name', u'COUPON_ID_hash']

a = pd.Series(range(0,len(label_cat))).astype(str)+':'
train[label_cat] = a.tolist() + train[label_cat].astype(str).applymap(hashstr) + pd.Series([':1']*len(label_cat)).astype(str).tolist()

#NaNを変換
nan = hashstr('nan')
train[label_cat] = train[label_cat].applymap(lambda x: 'NaN' if nan in x else x)

#Numericalデータ
'''
time         age
2:5:100    3:6:24
2:5:120    3:6:50
'''

label_num = [u'AGE', u'PRICE_RATE', u'CATALOG_PRICE', u'DISCOUNT_PRICE', u'DISPPERIOD',u'VALIDPERIOD',u'REG_DATE',u'WITHDRAW_DATE',u'DISPFROM', u'DISPEND',u'VALIDFROM', u'VALIDEND']
a = pd.Series(range(len(label_cat),(len(label_cat)+len(label_num)))).astype(str)+':'+pd.Series(range(0,len(label_num))).astype(str)+':'
train[label_num] = a.tolist() + train[label_num].astype(str)

#NaNを変換
train[label_num] = train[label_num].applymap(lambda x: 'NaN' if 'nan' in x else x)


train['date'] = train['REG_DATE'].apply(lambda x: int(x[5:]))
train = train.sort('date')
del train['date']
train = train[:-1]

#TrainのLibFMデータ作成（シャッフルせよ）
label = [u'REG_DATE', u'SEX_ID', u'AGE', u'WITHDRAW_DATE', u'PREF_NAME', u'USER_ID_hash', u'CAPSULE_TEXT', u'GENRE_NAME', u'PRICE_RATE', u'CATALOG_PRICE', u'DISCOUNT_PRICE', u'DISPFROM', u'DISPEND', u'DISPPERIOD', u'VALIDFROM', u'VALIDEND', u'VALIDPERIOD', u'USABLE_DATE_MON', u'USABLE_DATE_TUE', u'USABLE_DATE_WED', u'USABLE_DATE_THU', u'USABLE_DATE_FRI', u'USABLE_DATE_SAT', u'USABLE_DATE_SUN', u'USABLE_DATE_HOLIDAY', u'USABLE_DATE_BEFORE_HOLIDAY', u'large_area_name', u'ken_name', u'small_area_name', u'COUPON_ID_hash']
 
train.index = range(0,len(train))


#train = train.iloc[np.random.permutation(len(train))]


def train_convert(train1,parchase, dst_path, is_train):
    with open(dst_path, 'w') as f:
        #for row in csv.DictReader(open(src_path)):
        num = 0
        for row in train1.values:
            #row = train1.iloc[index,:]
            #NaNを削除
            row = filter(lambda x: x != 'NaN', row)
            if is_train == True:
                #print row
                f.write('{0} {1}\n'.format(parchase[num], ' '.join(row)))
            if is_train == False:
                f.write('{0} {1}\n'.format(0, ' '.join(feats)))
            num += 1
            if num % 10000 == 0:
                print num


#trainデータ
train_convert(train.iloc[(train['REG_DATE'].apply(lambda x: int(x[5:])<698)).values,:][train.columns.drop(u'PURCHASE_FLG')],train.iloc[(train['REG_DATE'].apply(lambda x: int(x[5:])<698)).values,:][u'PURCHASE_FLG'].values,'/Users/IkkiTanaka/Documents/kaggle/Coupon_Purchase_Prediction/libFM1_tr.ffm',True)

#validationデータ
train_convert(train.iloc[(train['REG_DATE'].apply(lambda x: int(x[5:])>=698)).values,:][train.columns.drop(u'PURCHASE_FLG')],train.iloc[(train['REG_DATE'].apply(lambda x: int(x[5:])>=698)).values,:][u'PURCHASE_FLG'].values,'/Users/IkkiTanaka/Documents/kaggle/Coupon_Purchase_Prediction/libFM1_val.ffm',True)



#TestデータのLibFMのデータ作成
test = pd.concat([user_list,coupon_list_test],axis=1)

#Categoricalデータ
a = pd.Series(range(0,len(label_cat))).astype(str)+':'
test[label_cat] = a.tolist() + test[label_cat].astype(str).applymap(hashstr) + pd.Series([':1']*len(label_cat)).astype(str).tolist()

#NaNを変換
nan = hashstr('nan')
test[label_cat] = test[label_cat].applymap(lambda x: 'NaN' if nan in x else x)

#Numericalデータ
a = pd.Series(range(len(label_cat),(len(label_cat)+len(label_num)))).astype(str)+':'+pd.Series(range(0,len(label_num))).astype(str)+':'
test[label_num] = a.tolist() + test[label_num].astype(str)

#NaNを変換
test[label_num] = test[label_num].applymap(lambda x: 'NaN' if 'nan' in x else x)

new_user_list = test[user_list.columns][0:len(user_list)]
new_coupon_list_test = test[coupon_list_test.columns][0:len(coupon_list_test)]


def test_convert(user_list1,coupon_list_test1,dst_path,is_train):
    with open(dst_path, 'w') as f:
        #for row in csv.DictReader(open(src_path)):
        num = 0
        for row1 in user_list1.values:#userを抽出
            for row2 in coupon_list_test1.values:#クーポンを抽出
                row = list(row1)
                row.extend(list(row2))
            #row = train1.iloc[index,:]
                #NaNを削除
                row = filter(lambda x: x != 'NaN', row)

                if is_train == True:
                    f.write('{0} {1}\n'.format(parchase[num], ' '.join(row)))
                if is_train == False:
                    f.write('{0} {1}\n'.format(0, ' '.join(row)))
                num += 1
                #print num
                if num % 100000 == 0:
                    print num

#user_ID_hash保存
def test_convert_user_id_hash(user_list1,coupon_list_test1,dst_path):
    with open(dst_path, 'w') as f:
        #for row in csv.DictReader(open(src_path)):
        num = 0
        for row1 in user_list1.values:
            for row2 in coupon_list_test1.values:
                f.write('{0},\n'.format(row1))
                num += 1
                #print num
                if num % 100000 == 0:
                    print num

test_convert(new_user_list,new_coupon_list_test,'/Users/IkkiTanaka/Documents/kaggle/Coupon_Purchase_Prediction/libFM1_test.ffm',False)

test_convert_user_id_hash(user_list[u'USER_ID_hash'],new_coupon_list_test,'/Users/IkkiTanaka/Documents/kaggle/Coupon_Purchase_Prediction/libFM1_test_user_id_hsash.csv')

#!/Users/IkkiTanaka/libfm-master/bin/libFM -dim '1,1,16' -iter 15 -learn_rate 0.05 -method mcmc -init_stdev 2.0 -task c -verbosity 2 -regular '0.01,0,0.01' -train /Users/IkkiTanaka/Documents/kaggle/Coupon_Purchase_Prediction/libFM1.fm -out /Users/IkkiTanaka/Documents/kaggle/Coupon_Purchase_Prediction/prediction.csv -test /Users/IkkiTanaka/Documents/kaggle/Coupon_Purchase_Prediction/libFM1_test.fm

#train(5-fold crossvalidation)
!/Users/IkkiTanaka/libffm-1.11/ffm-train -l 0.0 -k 16 -t 10 -r 0.01 -s 8 -p /Users/IkkiTanaka/Documents/kaggle/Coupon_Purchase_Prediction/libFM1_val.ffm /Users/IkkiTanaka/Documents/kaggle/Coupon_Purchase_Prediction/libFM1_tr.ffm /Users/IkkiTanaka/Documents/kaggle/Coupon_Purchase_Prediction/libFFM_model2.ffm 

!/Users/IkkiTanaka/libffm-1.11/ffm-predict /Users/IkkiTanaka/Documents/kaggle/Coupon_Purchase_Prediction/libFM1_val.ffm /Users/IkkiTanaka/Documents/kaggle/Coupon_Purchase_Prediction/libFFM_model2.ffm /Users/IkkiTanaka/Documents/kaggle/Coupon_Purchase_Prediction/prediction.csv

train.groupby('



#テスト
!/Users/IkkiTanaka/libffm-1.11/ffm-predict /Users/IkkiTanaka/Documents/kaggle/Coupon_Purchase_Prediction/libFM1_test.ffm /Users/IkkiTanaka/Documents/kaggle/Coupon_Purchase_Prediction/libFFM_model1.ffm /Users/IkkiTanaka/Documents/kaggle/Coupon_Purchase_Prediction/prediction.csv


c=pd.read_csv('/Users/IkkiTanaka/Documents/kaggle/Coupon_Purchase_Prediction/prediction.csv',header=None,names=['probability'])

c['USER_ID_hash'] = pd.read_csv('/Users/IkkiTanaka/Documents/kaggle/Coupon_Purchase_Prediction/libFM1_test_user_id_hsash.csv',header=None)[0].values
c['COUPON_ID_hash'] =  list(coupon_list_test['COUPON_ID_hash'].values) * len(user_list)

def top10(data):
    data = data.sort('probability',ascending=False)
    #print data.columns
    top10 = data['COUPON_ID_hash'].values[:10]
    #print top10
    user = data['USER_ID_hash'].values[0]
    #print user
    return pd.Series([user,' '.join(top10)],index=[u'USER_ID_hash',u'PURCHASED_COUPONS'])

pred = c.groupby('USER_ID_hash').apply(top10)
pred = pd.DataFrame(pred.values,columns=[u'USER_ID_hash',u'PURCHASED_COUPONS'])
pred.to_csv('/Users/IkkiTanaka/Documents/kaggle/Coupon_Purchase_Prediction/prediction/FM1.csv',index=False)



index = 0
index2 = 0
for i in user_list['USER_ID_hash'].values:
    c.iloc[(0+index),1] = i
    index += 310
    index2 += 1
    if index2 % 100 == 0:
        print index2

user_list['USER_ID_hash'].values

coupon_list_test['COUPON_ID_hash']


#Train期間に多く見られた順にTestのクーポンを推薦する
sample_submission['PURCHASED_COUPONS'] = [' '.join(coupon_visit_train.iloc[(coupon_visit_train['VIEW_COUPON_ID_hash'].isin(coupon_list_test['COUPON_ID_hash'].values)).values,:]['VIEW_COUPON_ID_hash'].value_counts().index[:10])]*len(sample_submission)
sample_submission.to_csv('/Users/IkkiTanaka/Documents/kaggle/Coupon_Purchase_Prediction/prediction/simple_pred1.csv',index=False)



