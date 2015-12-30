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
import matplotlib.pyplot as plt
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
cpvtr.columns = [u'PURCHASE_FLG', u'I_DATE', u'PAGE_SERIAL', u'REFERRER_hash', u'COUPON_ID_hash', u'USER_ID_hash', u'SESSION_ID_hash', u'PURCHASEID_hash']

cpdtr = pd.read_csv('data/coupon_detail_train.csv')#the purchase log of users buying coupons during the training set time period. not provided this table for the test set period（クーポンの購入ログ）

ulist = pd.read_csv('data/user_list.csv')#the master list of users in the dataset（ユーザーのリスト、予測すべきユーザー）
ss = pd.read_csv('sample_submission.csv')

cpltr.loc[(cpltr[u'CAPSULE_TEXT']=='ビューティ').values,u'CAPSULE_TEXT'] = 'ビューティー'

'''
plt.hist(ulist.AGE)
plt.show()
'''
#'AGE', u'CATALOG_PRICE', u'DISCOUNT_PRICE', u'DISPPERIOD', u'PRICE_RATE'をあるbinsで区切りダミー変数化する
#AGE
age_dic = {'AG10':pd.Series([1],index=['AG10']),'AG20':pd.Series([1],index=['AG20']),'AG30':pd.Series([1],index=['AG30']),'AG40':pd.Series([1],index=['AG40']),'AG50':pd.Series([1],index=['AG50']),'AG60':pd.Series([1],index=['AG60']),'AG70+':pd.Series([1],index=['AG70+'])}
def age_dummy(num):
    if num < 20:
        return age_dic['AG10']
    elif 20 <= num < 30:
        return age_dic['AG20']
    elif 30 <= num < 40:
        return age_dic['AG30']
    elif 40 <= num < 50:
        return age_dic['AG40']
    elif 50 <= num < 60:
        return age_dic['AG50']
    elif 60 <= num < 70:
        return age_dic['AG60']
    elif 70 <= num:
        return age_dic['AG70+']


#CATALOG_PRICE
catalog_dic = {'CP1':pd.Series([1],index=['CP1']),'CP10':pd.Series([1],index=['CP10']),'CP100':pd.Series([1],index=['CP100']),'CP1000':pd.Series([1],index=['CP1000']),'CP10000':pd.Series([1],index=['CP10000']),'CP100000':pd.Series([1],index=['CP100000'])}
def catalog_price_dummy(num):
    digit = int(math.log10(num) + 1)#桁数
    if digit == 1:
        return catalog_dic['CP1']
    elif digit == 2:
        return catalog_dic['CP10']
    elif digit == 3:
        return catalog_dic['CP100']
    elif digit == 4:
        return catalog_dic['CP1000']
    elif digit == 5:
        return catalog_dic['CP10000']
    elif digit == 6:
        return catalog_dic['CP100000']

#DISCOUNT_PRICE
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

#DISPPERIOD
DI_dic = {'DI0-5':pd.Series([1],index=['DI0-5']),'DI6-10':pd.Series([1],index=['DI6-10']),'DI11-15':pd.Series([1],index=['DI11-15']),'DI16-20':pd.Series([1],index=['DI16-20']),'DI20+':pd.Series([1],index=['DI20+'])}
def dispperiod_dummy(num):
    if num <= 5:
        return DI_dic['DI0-5']
    elif 5 < num <= 10:
        return DI_dic['DI6-10']
    elif 10 < num <= 15:
        return DI_dic['DI11-15']
    elif 15 < num <= 20:
        return DI_dic['DI16-20']
    elif 20 < num:
        return DI_dic['DI20+']


#PRICE_RATE
PR_dic = {'PR50-':pd.Series([1],index=['PR50-']),'PR50-60':pd.Series([1],index=['PR50-60']),'PR60-70':pd.Series([1],index=['PR60-70']),'PR70-80':pd.Series([1],index=['PR70-80']),'PR80-90':pd.Series([1],index=['PR80-90']),'PR90+':pd.Series([1],index=['PR90+'])}

def price_rate_dummy(num):
    if num < 50:
        return PR_dic['PR50-']
    elif 50 <= num < 60:
        return PR_dic['PR50-60']
    elif 60 <= num < 70:
        return PR_dic['PR60-70']
    elif 70 <= num < 80:
        return PR_dic['PR70-80']
    elif 80 <= num < 90:
        return PR_dic['PR80-90']
    elif 90 <= num:
        return PR_dic['PR90+']

#VALIDPERIOD
VP_dic = {'VP0-20':pd.Series([1],index=['VP0-20']),'VP21-50':pd.Series([1],index=['VP21-50']),'VP51-80':pd.Series([1],index=['VP51-80']),'VP81-110':pd.Series([1],index=['VP81-110']),'VP111+':pd.Series([1],index=['VP111+'])}
def validperiod_dummy(num):
    if 0 <= num <= 20:
        return VP_dic['VP0-20']
    elif 20 < num <= 50:
        return VP_dic['VP21-50']
    elif 50 < num <= 80:
        return VP_dic['VP51-80']
    elif 80 < num <= 110:
        return VP_dic['VP81-110']
    elif 110 < num:
        return VP_dic['VP111+']








'''
Trainデータ作成
'''

#User_listに購入データをマージ
ulist_train1 = ulist.merge(cpdtr,on='USER_ID_hash',how='outer')

#ulist_listに閲覧記録をマージ
ulist_train2 = ulist.merge(cpvtr,on='USER_ID_hash',how='outer')

#ulist_trainの閲覧時に購入しなかったが、最終的に購入した商品を削除
ulist_train = pd.concat([ulist_train1,ulist_train2])
ulist_train = ulist_train.drop_duplicates(['USER_ID_hash','COUPON_ID_hash'])

#ulist_trainのクーポンにクーポン情報をマージ
ulist_train = ulist_train.merge(cpltr,on='COUPON_ID_hash',how='left')

#欠損値補間
ulist_train['ITEM_COUNT'] = ulist_train['ITEM_COUNT'].fillna(0)
ulist_train['PURCHASE_FLG'] = ulist_train['PURCHASE_FLG'].fillna(1)
ulist_train['CATALOG_PRICE'] = ulist_train['CATALOG_PRICE'].fillna(-999)
ulist_train['DISCOUNT_PRICE'] = ulist_train['DISCOUNT_PRICE'].fillna(-999)
ulist_train['PRICE_RATE'] = ulist_train['PRICE_RATE'].fillna(-999)
ulist_train['DISPPERIOD'] = ulist_train['DISPPERIOD'].fillna(-999)
ulist_train['VALIDPERIOD'] = ulist_train['VALIDPERIOD'].fillna(-999)


usable_label = [u'USABLE_DATE_MON', u'USABLE_DATE_TUE', u'USABLE_DATE_WED', u'USABLE_DATE_THU', u'USABLE_DATE_FRI', u'USABLE_DATE_SAT', u'USABLE_DATE_SUN', u'USABLE_DATE_HOLIDAY', u'USABLE_DATE_BEFORE_HOLIDAY']
ulist_train[usable_label] = ulist_train[usable_label].fillna(1)

#ulist_train = ulist_train.merge(cpatr,on='COUPON_ID_hash',how='left')

#Testデータ作成
ulist['dummy'] = 'A'
cplte['dummy'] = 'A'

ulist_test = ulist.merge(cplte,on='dummy',how='outer')
del ulist_test['dummy']

ulist_test[usable_label] = ulist_test[usable_label].fillna(1)

#I_DATEがNaNのデータを削除する
ulist_train = ulist_train.iloc[(~ulist_train[u'I_DATE'].isnull()).values,:]
'''
基本的なTrainとTestが整った
'''
#ここでTrainのI_DATEを見てデータ整形するか
#時間変数
time_label = ['REG_DATE','WITHDRAW_DATE','DISPFROM','DISPEND','VALIDFROM','VALIDEND']


f_date = lambda x: datetime.date(int(x[0:4]),int(x[5:7]),int(x[8:10])).toordinal() - 734318
#I_DATEがNaNのユーザを削除
#ulist_train = ulist_train.iloc[(ulist_train[u'I_DATE'].notnull()).values,:]
ulist_train[u'I_DATE'] = ulist_train[u'I_DATE'].fillna('2012-06-23 12:00:00')
ulist_train[u'I_DATE'] = ulist_train[u'I_DATE'].apply(f_date)

#20120301より後
#ulist_train = ulist_train.iloc[(ulist_train[u'I_DATE']>=(datetime.datetime(2012,01,01).toordinal() - 734318)).values,:]

#20120623より前
ulist_train = ulist_train.iloc[((datetime.datetime(2012,06,23).toordinal()- 734318)>=ulist_train[u'I_DATE']).values,:]

#ulist_train[u'I_DATE'] = ulist_train[u'I_DATE'] - ulist_train[u'I_DATE'].min() +1
#ulist_train[u'I_DATE'] = np.log10(ulist_train[u'I_DATE']/10)

'''
validation用にtrainデータの20120617より後の1週間にその1週間で推薦できるクーポンを付加する

a = cpltr.iloc[(cpltr['DISPFROM'].apply(f_date).values>=(datetime.datetime(2012,06,17).toordinal()- 734318)),:]
a = a.iloc[(a['DISPFROM'].apply(f_date).values<(datetime.datetime(2012,06,24).toordinal()- 734318)),:]

b = ulist_train.iloc[(ulist_train[u'I_DATE']>=(datetime.datetime(2012,06,17).toordinal() - 734318)).values,:].copy()
b['dummy'] = 'A'
a['dummy'] = 'A'

b.merge(a,on='dummy',how='outer')#memory error
'''


train_label = ulist_test.columns.tolist()
train_label.append('PURCHASE_FLG')
train_label.append('I_DATE')#validation用

ulist_train = ulist_train[train_label]

#ulist_train.merge(cpatr,on='COUPON_ID_hash',how='outer')



'''
trainとtestの結合
PREF_NAME　ユーザの県
u'large_area_name', u'ken_name', u'small_area_name' クーポンの地域
'CAPSULE_TEXT', u'GENRE_NAME' クーポンの情報
'''
#object型のカラムを抽出しダミー変数化
object_col = ulist_test.iloc[:,(ulist_test.dtypes==object).values].columns.drop(['USER_ID_hash','COUPON_ID_hash','CAPSULE_TEXT','large_area_name']+time_label)


#データが多すぎるのでTestデータのユーザの住んでる地域だけのクーポンを推薦するかの材料にする
def pref_to_large(data):
    if data in ['東京都','神奈川県','埼玉県','千葉県','茨城県','群馬県','栃木県']:
        return u'関東'
    elif data in ['大阪府','兵庫県','京都府','奈良県','滋賀県','和歌山県']:
        return u'関西'
    elif data in ['愛知県','静岡県','三重県','岐阜県']:
        return u'東海'
    elif data in ['福岡県','熊本県','沖縄県','長崎県','大分県','鹿児島県','宮崎県','佐賀県']:
        return u'九州・沖縄'
    elif data in ['北海道']:
        return u'北海道'
    elif data in ['長野県','山梨県','石川県','福井県','新潟県','富山県']:
        return u'北信越'
    elif data in ['宮城県','福島県','山形県','岩手県','秋田県','青森県']:
        return u'東北'
    elif data in ['広島県','岡山県','山口県','島根県','鳥取県']:
        return u'中国'
    elif data in ['高知県','愛媛県','香川県','徳島県']:
        return u'四国'
    else:
        return u'不明'

#testデータのユーザの県を地域に変換する
#ulist_test['large_area2'] = ulist_test['PREF_NAME'].apply(pref_to_large).values

'''
Testデータの削減
地域性に関して
'''
#testデータのユーザの地域とクーポンの地域が同じデータのみを抽出する
def large2_is_large(data):
    #print data['large_area2'].values[0]
    #print data['large_area2'].values[0] != '不明'
    if data['large_area2'].values[0] != '不明':
        #print  (data['large_area2'] == data['large_area_name']).tolist()
        #print (data['ken_name']=='東京都').tolist()
        bool_label = (data['large_area2'] == data['large_area_name']).tolist() or (data['ken_name']=='東京都').tolist()
        #print bool_label
        return data.iloc[bool_label,:]
    else:#ulistのユーザを落とさないために
        return data

#def pref_is_ken(data):
#    #print data['large_area2'].values[0]
#    if data['large_area2'].values[0] != '不明':
#        return data.iloc[(data['large_area2'] == data['large_area_name']).values,:]
#    else:#ulistのユーザを落とさないために
#        return data


#ulist_test = ulist_test.groupby('USER_ID_hash').apply(large2_is_large)
#ulist_test = pd.DataFrame(ulist_test.values,columns=ulist_test.columns)
#del ulist_test['large_area2']
#ulist_test.drop_duplicates(['USER_ID_hash','large_area2','large_area_name'])

all_data = pd.concat([ulist_train,ulist_test])
all_data.index = range(0,len(all_data))




for ob_label in object_col: 
    #欠損値補間（場所再考せよ！）
    all_data[ob_label] = all_data[ob_label].fillna('NaN')
    a, b = categorical(np.array(all_data[ob_label]),drop=True,dictnames=True)
    colname = ob_label + '_' + pd.Series(list(b.viewvalues()))
    #colname = ob_label + pd.Series(range(0,len(all_data[ob_label].value_counts()))).astype(str)
    a = pd.DataFrame(a,columns=colname)
    all_data = pd.concat([all_data,a],axis=1)
    print ob_label, 'done'
    del all_data[ob_label]



'''
AGEなどをダミー変数化する
'''
#all_data = pd.concat([all_data,all_data['AGE'].apply(age_dummy).fillna(0)],axis=1)

#all_data = pd.concat([all_data,all_data['CATALOG_PRICE'].apply(catalog_price_dummy).fillna(0)],axis=1)

#all_data = pd.concat([all_data,all_data['DISCOUNT_PRICE'].apply(discount_price_dummy).fillna(0)],axis=1)

#all_data = pd.concat([all_data,all_data['DISPPERIOD'].apply(dispperiod_dummy).fillna(0)],axis=1)

#all_data = pd.concat([all_data,all_data['PRICE_RATE'].apply(price_rate_dummy).fillna(0)],axis=1)

#all_data = pd.concat([all_data,all_data['VALIDPERIOD'].fillna(1).apply(validperiod_dummy).fillna(0)],axis=1)

#del all_data['AGE'],all_data['CATALOG_PRICE'],all_data['DISCOUNT_PRICE'],all_data['DISPPERIOD'],all_data['PRICE_RATE'],all_data['VALIDPERIOD']


#trainとtest分割
all_train = all_data.iloc[:len(ulist_train),:]
all_test = all_data.iloc[len(ulist_train):,:]
del all_test['PURCHASE_FLG']

#時間変数削除
for t_label in time_label:
    del all_train[t_label], all_test[t_label]

'''
#Pickleして保存(データ大きすぎてエラー)
all_data.to_pickle('/Users/IkkiTanaka/Documents/kaggle/Coupon_Purchase_Prediction/data/all_data.dump')

all_data = pd.read_pickle('/Users/IkkiTanaka/Documents/kaggle/Coupon_Purchase_Prediction/data/all_data.dump')
'''
#Feature Engineering

#何回そのユーザが閲覧したか
#cp_watched = all_train['USER_ID_hash'].value_counts()
#cp_watched = pd.DataFrame(cp_watched,columns=['cp_watched'])
#cp_watched['USER_ID_hash'] = cp_watched.index.values
#cp_watched.index = range(0,len(cp_watched))
#all_train = all_train.merge(cp_watched,on='USER_ID_hash',how='left')
#all_test = all_test.merge(cp_watched,on='USER_ID_hash',how='left')

#何回そのユーザが購入したか
#cp_bought = all_train.groupby('USER_ID_hash').sum()['PURCHASE_FLG']
#cp_bought = pd.DataFrame(cp_bought)
#cp_bought.columns = ['cp_bought']
#cp_bought['USER_ID_hash'] = cp_bought.index.values
#cp_bought.index = range(0,len(cp_bought))
#all_train = all_train.merge(cp_bought,on='USER_ID_hash',how='left')
#all_test = all_test.merge(cp_bought,on='USER_ID_hash',how='left')

'''
学習
'''
#データをshuffle
import random
all_train = all_train.iloc[np.random.permutation(len(all_train))]
test_user_cp = all_test[['USER_ID_hash','COUPON_ID_hash']]
#del all_train['USER_ID_hash'], all_test['USER_ID_hash']
#del all_train['COUPON_ID_hash'], all_test['COUPON_ID_hash']

#all_test['CATALOG_PRICE'] = all_test['CATALOG_PRICE'].fillna(1)
#all_test['DISCOUNT_PRICE'] = all_test['DISCOUNT_PRICE'].fillna(1)
#all_test['PRICE_RATE'] = all_test['PRICE_RATE'].fillna(1)
#all_test['DISPPERIOD'] = all_test['DISPPERIOD'].fillna(1)
#all_test['VALIDPERIOD'] = all_test['VALIDPERIOD'].fillna(1)
#all_test['cp_watched'] = all_test['cp_watched'].fillna(1)
#all_test['cp_bought'] = all_test['cp_bought'].fillna(1)

for i in all_test.columns:
    if True in all_test[i].isnull().values:
        print i
all_test['VALIDPERIOD'] = all_test['VALIDPERIOD'].fillna(-999)
#メモリのためにall_data削除
del all_data
'''
#ユーザの過去購入したクーポンの平均情報を付加
drop_label = [u'I_DATE', u'PURCHASE_FLG', u'SEX_ID_f', u'SEX_ID_m', 'PREF_NAME_NaN', 'PREF_NAME_三重県', 'PREF_NAME_京都府', 'PREF_NAME_佐賀県', 'PREF_NAME_兵庫県', 'PREF_NAME_北海道', 'PREF_NAME_千葉県', 'PREF_NAME_和歌山県', 'PREF_NAME_埼玉県', 'PREF_NAME_大分県', 'PREF_NAME_大阪府', 'PREF_NAME_奈良県', 'PREF_NAME_宮城県', 'PREF_NAME_宮崎県', 'PREF_NAME_富山県', 'PREF_NAME_山口県', 'PREF_NAME_山形県', 'PREF_NAME_山梨県', 'PREF_NAME_岐阜県', 'PREF_NAME_岡山県', 'PREF_NAME_岩手県', 'PREF_NAME_島根県', 'PREF_NAME_広島県', 'PREF_NAME_徳島県', 'PREF_NAME_愛媛県', 'PREF_NAME_愛知県', 'PREF_NAME_新潟県', 'PREF_NAME_東京都', 'PREF_NAME_栃木県', 'PREF_NAME_沖縄県', 'PREF_NAME_滋賀県', 'PREF_NAME_熊本県', 'PREF_NAME_石川県', 'PREF_NAME_神奈川県', 'PREF_NAME_福井県', 'PREF_NAME_福岡県', 'PREF_NAME_福島県', 'PREF_NAME_秋田県', 'PREF_NAME_群馬県', 'PREF_NAME_茨城県', 'PREF_NAME_長崎県', 'PREF_NAME_長野県', 'PREF_NAME_青森県', 'PREF_NAME_静岡県', 'PREF_NAME_香川県', 'PREF_NAME_高知県', 'PREF_NAME_鳥取県', 'PREF_NAME_鹿児島県']
a = all_train.groupby('USER_ID_hash').mean()
a = pd.DataFrame(a)
a['USER_ID_hash'] = a.index
a.index = range(0,len(a))
a = a[a.columns.drop(drop_label+['cp_watched','cp_bought'])]

all_train = all_train.merge(a,on='USER_ID_hash',how='left')
all_test = all_test.merge(a,on='USER_ID_hash',how='left')
'''


#20120617より後
all_train_val = all_train.iloc[(all_train[u'I_DATE']>=(datetime.datetime(2012,06,17).toordinal() - 734318)).values,:]
#20120617より前
all_train_tr = all_train.iloc[((datetime.datetime(2012,06,17).toordinal()- 734318)>all_train[u'I_DATE']).values,:]

del all_train_tr['I_DATE'], all_train_val['I_DATE']
del all_test['I_DATE'],all_train['I_DATE']




#XGBoost
import xgboost as xgb

#0を0.000000001に変更する関数
f_0to01 = lambda x: 0.0000001 if x == 0 else x

dtrain = xgb.DMatrix(all_train_tr[all_train_tr.columns.drop(['USER_ID_hash','COUPON_ID_hash','PURCHASE_FLG'])].astype(float), label=all_train_tr['PURCHASE_FLG'])

dval = xgb.DMatrix(all_train_val[all_train_val.columns.drop(['USER_ID_hash','COUPON_ID_hash','PURCHASE_FLG'])].astype(float), label=all_train_val['PURCHASE_FLG'])


random.seed(254433433+234432443)
param = {"objective" : "binary:logistic",#rank:pairwise
              #"eval_metric" : 'auc',
              "eval_metric" : "map@10-",
              #'scale_pos_weight': 1.2, #sum_wneg/sum_wpos
              "num_class" :  1,
              "nthread" :  8,
              "bst:max_depth" :  random.randint(5,5),
              "bst:eta" :  random.uniform(.3, 0.3),#step_size
              "bst:gamma" : round(random.uniform(0., 0.),2),#min_loss_reduction
              "bst:min_child_weight" : random.randint(1.,1.),
              "bst:subsample" :  round(random.uniform(.8, .9),2),#row_subsample
              "colsample_bytree" :  round(random.uniform(.7, .9),2),#column_subsample
              "silent": 1,
              #"base_score": round(random.uniform(.50, .50),2),
              #"max_delta_step": 1,
              #"scale_pos_weight": 0.05,
        }
num_round = int( round(random.uniform(10,10),0))
print param
print num_round
#param = param.items()
#param += [("eval_metric", "map@10-")]
#前1週間でvalidationする
evallist  = [(dval, 'eval'),(dtrain,'train')]
bst = xgb.train(param,dtrain, num_round,evallist)

'''
validationでMAP計算
'''
pred_val = bst.predict( dval )
pred_val = pd.DataFrame(pred_val,columns=['probability'])
pred_val[['USER_ID_hash','PURCHASE_FLG']] = all_train_val[['USER_ID_hash','PURCHASE_FLG']].reset_index().drop('index',axis=1)

import map
def map2(data):
    if len(data)<100:
        return 0
    else:
        true_list = data.iloc[(data['PURCHASE_FLG']==1).values,:].index.astype(str).tolist()
        #print true_list
        pred_list = data.sort('probability',ascending=False).index[:10].astype(str).tolist()
        #print pred_list
        #print  map.mapk([true_list],[pred_list])
        return map.mapk([true_list],[pred_list])

print np.mean(pred_val.groupby('USER_ID_hash').apply(map2))


#5-fold cross-validation
#watchlist  = [(dtrain,'train')]
#bst=xgb.cv(param, dtrain, num_round, nfold=5, seed = 1992, show_stdv = False)
#f_ = lambda x: float(x.split('\tcv-test-auc:')[1][:8])
#bst = pd.Series(bst).apply(f_)
#num_round_ = bst.argmax() + 180
#cc = bst.max()
'''
#全データで学習する
'''
#全データ
dtrain_all = xgb.DMatrix(all_train[all_train.columns.drop(['USER_ID_hash','COUPON_ID_hash','PURCHASE_FLG'])].astype(float), label=all_train['PURCHASE_FLG'])
dtest = xgb.DMatrix(all_test[all_test.columns.drop(['USER_ID_hash','COUPON_ID_hash'])].astype(float))
random.seed(254433433+234432443)
param = {"objective" : "binary:logistic",#rank:pairwise
              #"eval_metric" : 'auc',
              "eval_metric" : "map@10-",
              #'scale_pos_weight': 1.2, #sum_wneg/sum_wpos
              "num_class" :  1,
              "nthread" :  8,
              "bst:max_depth" :  random.randint(2,2),
              "bst:eta" :  random.uniform(0.01, 0.1),#step_size
              #"bst:gamma" : round(random.uniform(0., 0.),2),#min_loss_reduction
              #"bst:min_child_weight" : random.randint(1.,1.),
              "bst:subsample" :  round(random.uniform(.8, .9),2),#row_subsample
              #"colsample_bytree" :  round(random.uniform(.7, .9),2),#column_subsample
              "silent": 1,
              #"base_score": round(random.uniform(.50, .50),2),
              #"max_delta_step": 1,
              "scale_pos_weight": 0.05,
        }
num_round = int( round(random.uniform(30,30),0))
print param
print num_round

watchlist  = [(dtrain_all,'train')]
bst = xgb.cv(param,dtrain_all, num_round,nfold=5, seed = 1992, show_stdv = False)


watchlist  = [(dtrain_all,'train')]
np.random.seed(19920407)
bst = xgb.train(param,dtrain_all, num_round,watchlist)
pred = bst.predict( dtest )

#予測値を利用してTOP10のクーポンを抽出する
pred = pd.DataFrame(pred,columns=['probability'])
pred[['USER_ID_hash','COUPON_ID_hash']] = test_user_cp

def top10(data):
    data = data.sort('probability',ascending=False)
    #print data.columns
    top10 = data['COUPON_ID_hash'].values[:10]
    #print top10
    user = data['USER_ID_hash'].values[0]
    #print user
    return pd.Series([user,' '.join(top10)],index=[u'USER_ID_hash',u'PURCHASED_COUPONS'])

pred = pred.groupby('USER_ID_hash').apply(top10)
pred = pd.DataFrame(pred.values,columns=[u'USER_ID_hash',u'PURCHASED_COUPONS'])
pred.to_csv('/Users/IkkiTanaka/Documents/kaggle/Coupon_Purchase_Prediction/prediction/xgb1.csv',index=False)





import matplotlib.pyplot as plt
xgb.plot_tree(bst, num_trees=2)
plt.show()
'''
fastFM
'''
from fastFM import mcmc
import scipy as sp
#targetを{-1,1}にする
all_train_tr['PURCHASE_FLG'][all_train_tr['PURCHASE_FLG'] < np.mean(all_train_tr['PURCHASE_FLG'])] = -1
all_train_val['PURCHASE_FLG'][all_train_val['PURCHASE_FLG'] < np.mean(all_train_val['PURCHASE_FLG'])] = -1


#sparce行列にしてから入力する必要がある
fm_tr = sp.sparse.csc_matrix((all_train_tr[all_train_tr.columns.drop(['USER_ID_hash','COUPON_ID_hash','PURCHASE_FLG'])].astype(float)))
fm_la_tr = all_train_tr['PURCHASE_FLG']
fm_val = sp.sparse.csc_matrix(all_train_val[all_train_val.columns.drop(['USER_ID_hash','COUPON_ID_hash','PURCHASE_FLG'])].astype(float))

#build model
fm = mcmc.FMClassification(n_iter=100, init_stdev=0.1, rank=8)

#fit and predict
pred_val = fm.fit_predict_proba(fm_tr,fm_la_tr,fm_val)

#validation
#pred_val = fm.predict_proba(all_train_val[all_train_val.columns.drop(['USER_ID_hash','COUPON_ID_hash','PURCHASE_FLG'])].astype(float))
pred_val = pd.DataFrame(pred_val,columns=['probability'])
pred_val[['USER_ID_hash','PURCHASE_FLG']] = all_train_val[['USER_ID_hash','PURCHASE_FLG']].reset_index().drop('index',axis=1)

import map
def map2(data):
    if len(data)<100:
        return 0
    else:
        true_list = data.iloc[(data['PURCHASE_FLG']==1).values,:].index.astype(str).tolist()
        #print true_list
        pred_list = data.sort('probability',ascending=False).index[:10].astype(str).tolist()
        #print pred_list
        #print  map.mapk([true_list],[pred_list])
        return map.mapk([true_list],[pred_list])

print np.mean(pred_val.groupby('USER_ID_hash').apply(map2))


#全データ

all_train['PURCHASE_FLG'][all_train['PURCHASE_FLG'] < np.mean(all_train['PURCHASE_FLG'])] = -1


fm_tr_all = sp.sparse.csc_matrix(all_train[all_train.columns.drop(['USER_ID_hash','COUPON_ID_hash','PURCHASE_FLG'])].astype(float))
fm_la_tr_all = all_train['PURCHASE_FLG']
fm_te = sp.sparse.csc_matrix(all_test[all_test.columns.drop(['USER_ID_hash','COUPON_ID_hash'])].astype(float))

fm = mcmc.FMClassification(n_iter=1000, init_stdev=0.1, rank=16)
pred = fm.fit_predict_proba(fm_tr_all,fm_la_tr_all,fm_te)


#予測値を利用してTOP10のクーポンを抽出する
pred = pd.DataFrame(pred,columns=['probability'])
pred[['USER_ID_hash','COUPON_ID_hash']] = test_user_cp

def top10(data):
    data = data.sort('probability',ascending=False)
    #print data.columns
    top10 = data['COUPON_ID_hash'].values[:10]
    #print top10
    user = data['USER_ID_hash'].values[0]
    #print user
    return pd.Series([user,' '.join(top10)],index=[u'USER_ID_hash',u'PURCHASED_COUPONS'])

pred = pred.groupby('USER_ID_hash').apply(top10)
pred = pd.DataFrame(pred.values,columns=[u'USER_ID_hash',u'PURCHASED_COUPONS'])
pred.to_csv('/Users/IkkiTanaka/Documents/kaggle/Coupon_Purchase_Prediction/prediction/fastFM2.csv',index=False)











'''

 X   *   Y   
U×F   F×C

'''
a = all_train[all_train.columns.drop(['COUPON_ID_hash','PURCHASE_FLG'])].head(30).groupby('USER_ID_hash').mean().astype(float)
col_name_tr = a.columns
user_list_test = pd.DataFrame(a.index,columns=['USER_ID_hash'])
a = np.matrix(a)

b = all_test[all_test.columns.drop(['USER_ID_hash'])].head(30).groupby('COUPON_ID_hash').mean().astype(float)
print b.columns == col_name_tr
b = b[col_name_tr]
coupon_list_test = pd.DataFrame(b.index,columns=['COUPON_ID_hash'])
b = np.matrix(b)

#Standardization
a2 = np.nan_to_num((a - a.mean(0))/ a.std(0))
b2 = np.nan_to_num((b - b.mean(0))/ b.std(0))

#重み行列 DISCOUNT_PRICE PRICE_RATE USABLE_DATE_ GENRE_NAME ken_name small_area_name
x1 = np.array([1,0,0,3,3,4])
x2 = np.repeat(x1, [1,1,9,13,47,55], axis=0)

W = np.matrix(np.diag(x2))

score = a.dot(b.T)
score2 = a2.dot(b2.T)


#類似度に従ってクーポンをソートし、上位１０個を取る
def top10_CF(data,score):
    #print data.index[0]
    top10_cp = coupon_list_test['COUPON_ID_hash'].ix[score[data.index[0]].argsort().tolist()[0][::-1][:10]]
    top10_cp = ' '.join(top10_cp)
    return top10_cp

user_list_test['PURCHASED_COUPONS'] = user_list_test.groupby('USER_ID_hash').apply(top10_CF,score).values


#保存
user_list_test.to_csv('/Users/IkkiTanaka/Documents/kaggle/Coupon_Purchase_Prediction/prediction/simple_pred1.csv',index=False)



#LogisticRegression
from sklearn.linear_model import LogisticRegression

clf = LogisticRegression(class_weight='auto',verbose=2)
clf.fit(all_train[all_train.columns.drop('PURCHASE_FLG')],all_train['PURCHASE_FLG'])
#予測
pred = clf.predict_proba(all_test)[:,1]







#予測値を利用してTOP10のクーポンを抽出する
pred = pd.DataFrame(pred,columns=['probability'])
pred[['USER_ID_hash','COUPON_ID_hash']] = test_user_cp

def top10(data):
    data = data.sort('probability',ascending=False)
    #print data.columns
    top10 = data['COUPON_ID_hash'].values[:10]
    #print top10
    user = data['USER_ID_hash'].values[0]
    #print user
    return pd.Series([user,' '.join(top10)],index=[u'USER_ID_hash',u'PURCHASED_COUPONS'])

pred = pred.groupby('USER_ID_hash').apply(top10)
pred = pd.DataFrame(pred.values,columns=[u'USER_ID_hash',u'PURCHASED_COUPONS'])
pred.to_csv('/Users/IkkiTanaka/Documents/kaggle/Coupon_Purchase_Prediction/prediction/xgb1.csv',index=False)

'''
LR
3月-6月のデータ 0.000900

xgb
全期間データ xgb1 Local 0.002670 LB 0.003361

'''

