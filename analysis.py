'''
train period: 2011-07-01 to 2012-06-23
test period: 2012-06-24 to 2012-06-30

Userベース

Itemベース


'''



import sys
import pandas as pd
import numpy as np
import os

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


#SORT
#PAGE_SERIALの意味を知るために
coupon_visit_train = coupon_visit_train.sort(['USER_ID_hash','PAGE_SERIAL'])

coupon_list_train.loc[(coupon_list_train[u'CAPSULE_TEXT']=='ビューティ').values,u'CAPSULE_TEXT'] = 'ビューティー'


'''
#TRAIN
1.coupon_area_train
u'SMALL_AREA_NAME', u'PREF_NAME', u'COUPON_ID_hash'
市区町村、県名、クーポンID

2.coupon_list_train
u'CAPSULE_TEXT', u'GENRE_NAME', u'PRICE_RATE', u'CATALOG_PRICE', u'DISCOUNT_PRICE', u'DISPFROM', u'DISPEND', u'DISPPERIOD', u'VALIDFROM', u'VALIDEND', u'VALIDPERIOD', u'USABLE_DATE_MON', u'USABLE_DATE_TUE', u'USABLE_DATE_WED', u'USABLE_DATE_THU', u'USABLE_DATE_FRI', u'USABLE_DATE_SAT', u'USABLE_DATE_SUN', u'USABLE_DATE_HOLIDAY', u'USABLE_DATE_BEFORE_HOLIDAY', u'large_area_name', u'ken_name', u'small_area_name', u'COUPON_ID_hash'
簡約テキスト、ジャンル、割引率、元価格、割引後の価格、公開日、公開終了日、公開された日数、有効開始期限、有効終了期限、有効期限の日数、月可、火可、水可、木可、金可、土可、日可、休日可、休前日可、地方、県名、市区町村、クーポンID

3.coupon_visit_train
u'PURCHASE_FLG', u'I_DATE', u'PAGE_SERIAL', u'REFERRER_hash', u'VIEW_COUPON_ID_hash', u'USER_ID_hash', u'SESSION_ID_hash', u'PURCHASEID_hash'
購入FLG、閲覧した時間、ページシリアル、参照ハッシュ、閲覧クーポンID、ユーザID、セッションID、購入ハッシュ

4.coupon_detail_train
u'ITEM_COUNT', u'I_DATE', u'SMALL_AREA_NAME', u'PURCHASEID_hash', u'USER_ID_hash', u'COUPON_ID_hash'
購入数、購入した時間、市区町村、購入ハッシュ、ユーザID、クーポンID

#TEST
5.coupon_area_test
u'SMALL_AREA_NAME', u'PREF_NAME', u'COUPON_ID_hash'
市区町村、県名、クーポンID

6.coupon_list_test
u'CAPSULE_TEXT', u'GENRE_NAME', u'PRICE_RATE', u'CATALOG_PRICE', u'DISCOUNT_PRICE', u'DISPFROM', u'DISPEND', u'DISPPERIOD', u'VALIDFROM', u'VALIDEND', u'VALIDPERIOD', u'USABLE_DATE_MON', u'USABLE_DATE_TUE', u'USABLE_DATE_WED', u'USABLE_DATE_THU', u'USABLE_DATE_FRI', u'USABLE_DATE_SAT', u'USABLE_DATE_SUN', u'USABLE_DATE_HOLIDAY', u'USABLE_DATE_BEFORE_HOLIDAY', u'large_area_name', u'ken_name', u'small_area_name', u'COUPON_ID_hash'
簡約テキスト、ジャンル、割引率、元価格、割引後の価格、公開日、公開終了日、公開された日数、有効開始期限、有効終了期限、有効期限の日数、月可、火可、水可、木可、金可、土可、日可、休日可、休前日可、地方、県名、市区町村、クーポンID

7.user_list
u'REG_DATE', u'SEX_ID', u'AGE', u'WITHDRAW_DATE', u'PREF_NAME', u'USER_ID_hash'
登録日、性別、年齢、退会日、県名、ユーザID
'''


#それぞれのデータの数
print 'それぞれのデータの数'
print 'coupon_area_train: ', len(coupon_area_train)
print 'coupon_list_train: ', len(coupon_list_train)
print 'coupon_visit_train: ', len(coupon_visit_train)
print 'coupon_detail_train: ', len(coupon_detail_train)
print 'coupon_area_test: ', len(coupon_area_test)
print 'coupon_list_test: ', len(coupon_list_test)
print 'user_list: ', len(user_list)

#それぞれのデータの中身の数
for i in ['coupon_area_train','coupon_list_train','coupon_visit_train','coupon_detail_train','coupon_area_test','coupon_list_test','user_list']:
    data = eval(i)
    print i
    print '\n'
    for j in data.columns:
        print j, ': '
        print data[j].value_counts()



#予測に用いられる310個のcoupon_list_testがtrainに入っている数
print coupon_visit_train.iloc[(coupon_visit_train['VIEW_COUPON_ID_hash'].isin(coupon_list_test['COUPON_ID_hash'].values)).values,:]
#coupon_detail_train['COUPON_ID_hash'].isin(coupon_list_test['COUPON_ID_hash'].values)

#train内のuserがtestに現れるか否か
print coupon_detail_train.iloc[(coupon_detail_train['USER_ID_hash'].isin(user_list['USER_ID_hash'].values)).values,:]
#trainの中の全userはuser_listに全て含まれる
#coupon_detail_train['PURCHASEID_hash']
user_list['USER_ID_hash'].isin(coupon_detail_train['USER_ID_hash'].values)
for i in xrange(
print coupon_area_train.ix[0]
print coupon_list_train.iloc[(coupon_list_train['COUPON_ID_hash']==coupon_area_train.ix[0]['COUPON_ID_hash']).values,:]

#user_listにcoupon_list_train, coupon_list_testをmergeしてみる
user_list.merge(coupon_list_train, left_on=u'USER_ID_hash', right_on=u'COUPON_ID_hash', how='outer')

#Trainデータ
train = coupon_visit_train[[u'USER_ID_hash', u'VIEW_COUPON_ID_hash', u'PURCHASE_FLG']]
train = train.reset_index().merge(coupon_list_train, left_on=u'VIEW_COUPON_ID_hash',right_on=u'COUPON_ID_hash',how='outer').sort('index').drop('index',axis=1)

#train[u'VIEW_COUPON_ID_hash'].isin(coupon_list_train[u'COUPON_ID_hash'].values)
#train[u'VIEW_COUPON_ID_hash'].isin(coupon_detail_train[u'COUPON_ID_hash'].values)

#Testデータ
test = np.zeros(len(user_list)*len(coupon_list_test)*(1+len(coupon_list_test.columns))).reshape(len(user_list)*len(coupon_list_test),(1+len(coupon_list_test.columns)))
test = pd.DataFrame(test)
index = 0
for user in user_list[u'USER_ID_hash']:
    test.loc[index:(index+309),0] = user


import matplotlib.pyplot as plt
plt.hist(ulist.AGE)
plt.show()
