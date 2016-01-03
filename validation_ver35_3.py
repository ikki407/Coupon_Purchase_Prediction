import numpy as np
import pandas as pd
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
cpvtr.columns = [u'PURCHASE_FLG', u'I_DATE', u'PAGE_SERIAL', u'REFERRER_hash', u'COUPON_ID_hash', u'USER_ID_hash', u'SESSION_ID_hash', u'PURCHASEID_hash']

cpdtr = pd.read_csv('data/coupon_detail_train.csv')#the purchase log of users buying coupons during the training set time period. not provided this table for the test set period（クーポンの購入ログ）

ulist = pd.read_csv('data/user_list.csv')#the master list of users in the dataset（ユーザーのリスト、予測すべきユーザー）
ss = pd.read_csv('sample_submission.csv')

cpltr.loc[(cpltr[u'CAPSULE_TEXT']=='ビューティ').values,u'CAPSULE_TEXT'] = 'ビューティー'

f = lambda x:x.lower()
cpatr.columns = map(f,cpatr.columns.values)
cpate.columns = map(f,cpate.columns.values)
cpltr.columns = map(f,cpltr.columns.values)
cplte.columns = map(f,cplte.columns.values)
cpvtr.columns = map(f,cpvtr.columns.values)
cpdtr.columns = map(f,cpdtr.columns.values)
ulist.columns = map(f,ulist.columns.values)
ss.columns = map(f,ss.columns.values)

'''
%whosで変数確認できます
'''



####RF

train = cpdtr.groupby('coupon_id_hash').sum()['item_count']
train = pd.DataFrame(train,columns=['item_count'])
train['coupon_id_hash'] = train.index
train.index = range(0,len(train))
#trainの基本形は整った

#train,testを準備する
train = train.merge(cpltr,on='coupon_id_hash',how='left')
train = train[[u'item_count', u'coupon_id_hash', 
       u'price_rate', u'catalog_price', u'discount_price', u'dispperiod', u'validperiod',
       u'usable_date_before_holiday', u'large_area_name', u'ken_name',
       u'small_area_name','capsule_text']]

test = cplte[[u'coupon_id_hash', 
       u'price_rate', u'catalog_price', u'discount_price', u'dispperiod', u'validperiod',
       u'usable_date_before_holiday', u'large_area_name', u'ken_name',
       u'small_area_name','capsule_text']]

#del train[u'coupon_id']#, test[u'coupon_id']

#実際のdispperiod
f_week = lambda x: datatime.date(2012,6,30) - datetime.date(int(x[:4]),int(x[6:8]),int(x[10:12])) + 1
test['dispperiod'] = test['dispfrom'].apply(f_week)



print train.columns[1:] == test.columns

#train,testの中のカテゴリカル変数のバイナリ化
train_and_test = pd.concat([train,test])
train_and_test.index = range(0,len(train_and_test))




#capsule_textの特殊なのを消す
train_and_test = train_and_test.iloc[~(train_and_test['capsule_text'].isin(['イベント','公共の宿','ゲストハウス','ペンション','民宿','ビューティー','レッスン'])).values,:]

set(train['capsule_text']) and  set(test['capsule_text'])

#場所の変数のバイナリ化
for bi_label in ['large_area_name','ken_name','small_area_name','capsule_text']:
	bi_data = pd.get_dummies(train_and_test[bi_label])
	bi_data.columns = bi_label + '_' + bi_data.columns
	train_and_test = pd.concat([train_and_test, bi_data],axis=1)
	del train_and_test[bi_label]

train_and_test = train_and_test.fillna(0)
#train,testの分割
train = train_and_test.iloc[:-len(cplte),:]
test = train_and_test.iloc[-len(cplte):,:]

#discount_priceが0の試供品を抜く
train = train.iloc[(train['discount_price']!=0).values,:]
train['catalog-discount'] = train['catalog_price'] - train['discount_price']

test = test.iloc[(test['discount_price']!=0).values,:]
test['catalog-discount'] = test['catalog_price'] - test['discount_price']

print 'item_countをソートしてみると面白い情報が！？',train.sort('item_count')

#cp_topic50 = pd.read_csv('20150915_coupon_id_topic50.csv')
#cp_topic50['coupon_id_hash'] = cp_topic50['coupon_id_hash'].astype(str)

#train = train.merge(cp_topic50,on='coupon_id_hash',how='left')
#test = test.merge(cp_topic50,on='coupon_id_hash',how='left')

del test[u'item_count']

from sklearn.ensemble import RandomForestRegressor
from sklearn import cross_validation

train = train.fillna(0)
test = test.fillna(0)
test.index = range(0,len(test))
#test = test.astype(float)
not_using_label = [u'coupon_id_hash',u'dispperiod', u'validperiod',
       u'usable_date_before_holiday']

not_using_label_te = [u'coupon_id_hash',u'dispperiod', u'validperiod',
       u'usable_date_before_holiday']

a = train[train.columns.drop(not_using_label)]
a = a.iloc[(a['item_count']<200).values,:]




model = RandomForestRegressor(n_estimators=500, criterion='mse', max_depth=None, min_samples_split=2, 
	min_samples_leaf=1, min_weight_fraction_leaf=0.0, max_features='auto', 
	max_leaf_nodes=None, bootstrap=True, oob_score=False, n_jobs=8, random_state=19920407,
	 verbose=1, warm_start=False)
def get_score(clf, train_features, train_labels):
    X_train, X_test, y_train, y_test = cross_validation.train_test_split(train_features, train_labels, test_size=0.2, random_state=0)

    clf.fit(X_train, y_train)
    print clf.score(X_test, y_test) 
    print np.sum((clf.predict(X_test)-y_test.astype(float).values)*(clf.predict(X_test)-y_test.astype(float).values))/float(len(y_test))
    return pd.DataFrame(zip(clf.predict(X_test), y_test.values))

x = get_score(model,a[a.columns.drop('item_count')],a['item_count'])
print '予想売却数   実際の売却数', x.sort(0)
model.fit(a[a.columns.drop('item_count')],a['item_count'])

pred = model.predict(test[test.columns.drop(not_using_label_te)].fillna(0))

pred = pd.concat([pd.DataFrame(pred),test],axis=1)
print pred.sort(0)

imp = model.feature_importances_
imp = pd.DataFrame(imp,columns=['imp'])
imp = pd.concat([imp,pd.DataFrame(a.columns.drop('item_count').values)],axis=1)
print imp.sort('imp')
#pd.to_csv('20150916_pred_coupon_num.csv')

pred_coupon_num = pred


'''
いつもどおりの推薦
'''




#直近1週間の閲覧記録も興味があるとみなしてcpdtrにつける
cpvtr_1week = cpvtr.iloc[(cpvtr['i_date']>='2012-05-24 00:00:00').values,:]
#購入していて閲覧と購入のログが存在するものを削除
#cpdtrにつけるから
#cpvtr_1week = cpvtr_1week.sort('purchase_flg')
cpvtr_1week = cpvtr_1week.drop_duplicates(['user_id_hash','coupon_id_hash','session_id_hash'])
cpvtr_1week = cpvtr_1week.iloc[(cpvtr_1week['purchase_flg']==0).values,:]
#過去に一回でも見ていれば興味あるというコード
#cpvtr_1week = cpvtr_1week.drop_duplicates(['user_id_hash','coupon_id_hash'])
#過去に違う日で何回も見ていれば興味があるというコード
#cpvtr_1week = cpvtr_1week.drop_duplicates(['user_id_hash','coupon_id_hash','session_id_hash'])

#cpdtr2 = cpdtr.iloc[(cpdtr['i_date']>='2012-03-01 00:00:00').values,:]
cpdtr2 = cpdtr.drop_duplicates(['user_id_hash','coupon_id_hash'])
cpdtr_and_cpvtr = pd.concat([cpdtr2,cpvtr_1week])
cpdtr_and_cpvtr.index = range(0,len(cpdtr_and_cpvtr))

#閲覧回数1回の人にはvisitをつける
user_1detail = cpdtr_and_cpvtr['user_id_hash'].value_counts()
user_1detail = user_1detail[user_1detail==1].index.values

user_1detail_tr = cpvtr.iloc[(cpvtr['user_id_hash'].isin(user_1detail)).values,:]
user_1detail_tr = user_1detail_tr.drop_duplicates(['user_id_hash','coupon_id_hash','session_id_hash'])
user_1detail_tr = user_1detail_tr.iloc[(user_1detail_tr['purchase_flg']==0).values,:]

cpdtr_and_cpvtr = pd.concat([cpdtr_and_cpvtr,user_1detail_tr])
cpdtr_and_cpvtr.index = range(0,len(cpdtr_and_cpvtr))

#ここまで1回も現れていないユーザの情報
in_cpdtr_and_cpvtr_user = cpdtr_and_cpvtr['user_id_hash'].value_counts().index.values
not_detail_user = cpvtr.iloc[~(cpvtr['user_id_hash'].isin(in_cpdtr_and_cpvtr_user)).values,:]
not_detail_user = not_detail_user.drop_duplicates(['user_id_hash','coupon_id_hash','session_id_hash'])
not_detail_user = not_detail_user.iloc[(not_detail_user['purchase_flg']==0).values,:]

cpdtr_and_cpvtr = pd.concat([cpdtr_and_cpvtr,not_detail_user])
cpdtr_and_cpvtr.index = range(0,len(cpdtr_and_cpvtr))

cpltr_and_cplte = pd.concat([cpltr,cplte])

#userの購入情報にクーポンの詳細をつける
user_item_tr = cpdtr_and_cpvtr.merge(cpltr_and_cplte,on='coupon_id_hash',how='left')

#特徴量を選択する(small_area_name_xは購入者の場所)
feature_label = [u'coupon_id_hash', u'user_id_hash',u'genre_name',u'large_area_name', u'ken_name',u'small_area_name_y','discount_price','price_rate','catalog_price']
all_data = user_item_tr[feature_label]
all_data.columns = [u'coupon_id_hash', u'user_id_hash',u'genre_name',u'large_area_name', u'ken_name',u'small_area_name','discount_price','price_rate','catalog_price']

#cpltr_and_cplteに無いクーポンを見ているユーザを消すか否か
all_data = all_data.iloc[(all_data['coupon_id_hash'].isin(cpltr_and_cplte['coupon_id_hash'])).values,:]
all_data = all_data.iloc[~(all_data['user_id_hash'].isnull()).values,:]


#genre_nameがビューティのクーポンを消す
all_data = all_data.iloc[~(all_data['genre_name']=='ビューティー').values,:]



#trainとtestを結合する
feature_label_te = [u'coupon_id_hash',u'genre_name',u'large_area_name', u'ken_name',u'small_area_name','discount_price','price_rate','catalog_price']
#結合する前にTrainの長さを保存する
len_tr = len(all_data)
all_data = pd.concat([all_data,cplte[feature_label_te]])
all_data.index = range(0,len(all_data))

#宅配とその他のクーポンの地域をNaNにする
all_data.loc[(all_data['genre_name'].isin(['宅配','その他のクーポン'])).values,['large_area_name','ken_name','small_area_name']] = np.nan


#場所の変数のバイナリ化
for bi_label in ['genre_name','large_area_name','ken_name','small_area_name']:
	bi_data = pd.get_dummies(all_data[bi_label])
	bi_data.columns = bi_label + '_' + bi_data.columns
	all_data = pd.concat([all_data, bi_data],axis=1)
	del all_data[bi_label]

#discount_priceのバイナリ化
dp_dic = {'dp0':pd.Series([1],index=['dp0']),'dp1_500':pd.Series([1],index=['dp1_500']),
'dp500_1500':pd.Series([1],index=['dp500_1500']),'dp1500_2500':pd.Series([1],index=['dp1500_2500']),'dp2500_3500':pd.Series([1],index=['dp2500_3500']),
'dp3500_4500':pd.Series([1],index=['dp3500_4500']),'dp4500_5500':pd.Series([1],index=['dp4500_5500']),
'dp5500_6500':pd.Series([1],index=['dp5500_6500']),'dp6500_7500':pd.Series([1],index=['dp6500_7500']),
'dp7500_8500':pd.Series([1],index=['dp7500_8500']),'dp8500_9500':pd.Series([1],index=['dp8500_9500']),
'dp9500_10500':pd.Series([1],index=['dp9500_10500']),'dp10500_11500':pd.Series([1],index=['dp10500_11500']),
'dp11500_12500':pd.Series([1],index=['dp11500_12500']),'dp12500_13500':pd.Series([1],index=['dp12500_13500']),
'dp13500_14500':pd.Series([1],index=['dp13500_14500']),'dp14500_15500':pd.Series([1],index=['dp14500_15500']),
'dp15500_16500':pd.Series([1],index=['dp15500_16500']),'dp16500_17500':pd.Series([1],index=['dp16500_17500']),
'dp17500_18500':pd.Series([1],index=['dp17500_18500']),'dp18500_19500':pd.Series([1],index=['dp18500_19500']),
'dp19500_20500':pd.Series([1],index=['dp19500_20500']),'dp20500_25000':pd.Series([1],index=['dp20500_25000']),
'dp25000_35000':pd.Series([1],index=['dp25000_35000']),'dp35000_45000':pd.Series([1],index=['dp35000_45000']),
'dp45000_55000':pd.Series([1],index=['dp45000_55000']),'dp55000+':pd.Series([1],index=['dp55000+'])}
def discount_price_to_bi(data):
	#print data
	if data == 0:
		return dp_dic['dp0']
	elif 1 <= data <= 500:
		return dp_dic['dp1_500']
	elif 500 < data <= 1500:
		return dp_dic['dp500_1500']
	elif 1500 < data <= 2500:
		return dp_dic['dp1500_2500']
	elif 2500 < data <= 3500:
		return dp_dic['dp2500_3500']
	elif 3500 < data <= 4500:
		return dp_dic['dp3500_4500']
	elif 4500 < data <= 5500:
		return dp_dic['dp4500_5500']
	elif 5500 < data <= 6500:
		return dp_dic['dp5500_6500']
	elif 6500 < data <= 7500:
		return dp_dic['dp6500_7500']
	elif 7500 < data <= 8500:
		return dp_dic['dp7500_8500']
	elif 8500 < data <= 9500:
		return dp_dic['dp8500_9500']
	elif 9500 < data <= 10500:
		return dp_dic['dp9500_10500']
	elif 10500 < data <= 11500:
		return dp_dic['dp10500_11500']
	elif 11500 < data <= 12500:
		return dp_dic['dp11500_12500']
	elif 12500 < data <= 13500:
		return dp_dic['dp12500_13500']
	elif 13500 < data <= 14500:
		return dp_dic['dp13500_14500']
	elif 14500 < data <= 15500:
		return dp_dic['dp14500_15500']
	elif 15500 < data <= 16500:
		return dp_dic['dp15500_16500']
	elif 16500 < data <= 17500:
		return dp_dic['dp16500_17500']
	elif 17500 < data <= 18500:
		return dp_dic['dp17500_18500']
	elif 18500 < data <= 19500:
		return dp_dic['dp18500_19500']
	elif 19500 < data <= 20500:
		return dp_dic['dp19500_20500']
	elif 20500 < data <= 25000:
		return dp_dic['dp20500_25000']
	elif 25000 < data <= 35000:
		return dp_dic['dp25000_35000']
	elif 35000 < data <= 45000:
		return dp_dic['dp35000_45000']
	elif 45000 < data <= 55000:
		return dp_dic['dp45000_55000']
	elif 55000 < data:
		return dp_dic['dp55000+']

a = all_data['discount_price'].apply(discount_price_to_bi).fillna(0)
#columnsをソートしようかと思ったが意味はないのでやめた
#a = a[sorted(a.columns)]
all_data = pd.concat([all_data,a],axis=1)
#all_data[['dp0','dp10','dp100','dp1000','dp10000','dp100000']] = all_data['discount_price'].apply(discount_price_to_bi).fillna(0)

#price_rateのバイナリ化
pr_dic = {'pr45-':pd.Series([1],index=['pr45-']),'pr45_55':pd.Series([1],index=['pr45_55']),
'pr55_65':pd.Series([1],index=['pr55_65']),'pr65_75':pd.Series([1],index=['pr65_75']),'pr75_85':pd.Series([1],index=['pr75_85']),
'pr85_95':pd.Series([1],index=['pr85_95']),'pr95+':pd.Series([1],index=['pr95+'])}
def price_rate_to_bi(data):
	#print data
	if data <= 45:
		#print data
		return pr_dic['pr45-']
	elif 45 < data <= 55:
		return pr_dic['pr45_55']
	elif 55 < data <= 65:
		return pr_dic['pr55_65']
	elif 65 < data <= 75:
		return pr_dic['pr65_75']
	elif 75 < data <= 85:
		return pr_dic['pr75_85']
	elif 85 < data <= 95:
		return pr_dic['pr85_95']
	elif 95 < data:
		return pr_dic['pr95+']

all_data['price_rate'] = all_data['price_rate'].astype(int)
a = all_data['price_rate'].apply(price_rate_to_bi).fillna(0)
all_data = pd.concat([all_data,a],axis=1)


#all_data[['pr50-','pr50-60','pr60-70','pr70-80','pr80-90','pr90+']] = all_data['price_rate'].apply(price_rate_to_bi).fillna(0)

del all_data['price_rate']

#catalog_price - discount_priceのバイナリ化
all_data['catalog_price-discount_price'] = all_data['catalog_price'] - all_data['discount_price']
all_data['catalog_price-discount_price'] = all_data['catalog_price'] - all_data['discount_price']
cd_dic = {'cd0':pd.Series([1],index=['cd0']),'cd1_500':pd.Series([1],index=['cd1_500']),
'cd500_1500':pd.Series([1],index=['cd500_1500']),'cd1500_2500':pd.Series([1],index=['cd1500_2500']),'cd2500_3500':pd.Series([1],index=['cd2500_3500']),
'cd3500_4500':pd.Series([1],index=['cd3500_4500']),'cd4500_5500':pd.Series([1],index=['cd4500_5500']),
'cd5500_6500':pd.Series([1],index=['cd5500_6500']),'cd6500_7500':pd.Series([1],index=['cd6500_7500']),
'cd7500_8500':pd.Series([1],index=['cd7500_8500']),'cd8500_9500':pd.Series([1],index=['cd8500_9500']),
'cd9500_10500':pd.Series([1],index=['cd9500_10500']),'cd10500_11500':pd.Series([1],index=['cd10500_11500']),
'cd11500_12500':pd.Series([1],index=['cd11500_12500']),'cd12500_13500':pd.Series([1],index=['cd12500_13500']),
'cd13500_14500':pd.Series([1],index=['cd13500_14500']),'cd14500_15500':pd.Series([1],index=['cd14500_15500']),
'cd15500_16500':pd.Series([1],index=['cd15500_16500']),'cd16500_17500':pd.Series([1],index=['cd16500_17500']),
'cd17500_18500':pd.Series([1],index=['cd17500_18500']),'cd18500_19500':pd.Series([1],index=['cd18500_19500']),
'cd19500_20500':pd.Series([1],index=['cd19500_20500']),'cd20500_25000':pd.Series([1],index=['cd20500_25000']),
'cd25000_35000':pd.Series([1],index=['cd25000_35000']),'cd35000_45000':pd.Series([1],index=['cd35000_45000']),
'cd45000_55000':pd.Series([1],index=['cd45000_55000']),'cd55000+':pd.Series([1],index=['cd55000+'])}
def cat_dis_to_bi(data):
    #print data
    if data == 0:
        return cd_dic['cd0']
    elif 1 <= data <= 500:
        return cd_dic['cd1_500']
    elif 500 < data <= 1500:
        return cd_dic['cd500_1500']
    elif 1500 < data <= 2500:
        return cd_dic['cd1500_2500']
    elif 2500 < data <= 3500:
        return cd_dic['cd2500_3500']
    elif 3500 < data <= 4500:
        return cd_dic['cd3500_4500']
    elif 4500 < data <= 5500:
        return cd_dic['cd4500_5500']
    elif 5500 < data <= 6500:
        return cd_dic['cd5500_6500']
    elif 6500 < data <= 7500:
        return cd_dic['cd6500_7500']
    elif 7500 < data <= 8500:
        return cd_dic['cd7500_8500']
    elif 8500 < data <= 9500:
        return cd_dic['cd8500_9500']
    elif 9500 < data <= 10500:
        return cd_dic['cd9500_10500']
    elif 10500 < data <= 11500:
        return cd_dic['cd10500_11500']
    elif 11500 < data <= 12500:
        return cd_dic['cd11500_12500']
    elif 12500 < data <= 13500:
        return cd_dic['cd12500_13500']
    elif 13500 < data <= 14500:
        return cd_dic['cd13500_14500']
    elif 14500 < data <= 15500:
        return cd_dic['cd14500_15500']
    elif 15500 < data <= 16500:
        return cd_dic['cd15500_16500']
    elif 16500 < data <= 17500:
        return cd_dic['cd16500_17500']
    elif 17500 < data <= 18500:
        return cd_dic['cd17500_18500']
    elif 18500 < data <= 19500:
        return cd_dic['cd18500_19500']
    elif 19500 < data <= 20500:
        return cd_dic['cd19500_20500']
    elif 20500 < data <= 25000:
        return cd_dic['cd20500_25000']
    elif 25000 < data <= 35000:
        return cd_dic['cd25000_35000']
    elif 35000 < data <= 45000:
        return cd_dic['cd35000_45000']
    elif 45000 < data <= 55000:
        return cd_dic['cd45000_55000']
    elif 55000 < data:
        return cd_dic['cd55000+']

a = all_data['catalog_price-discount_price'].apply(cat_dis_to_bi).fillna(0)
all_data = pd.concat([all_data,a],axis=1)

#all_data[['cd0','cd1','cd100','cd1000','cd10000','cd100000']] = all_data['catalog_price-discount_price'].apply(cat_dis_to_bi).fillna(0)
del all_data['discount_price'],all_data['catalog_price']#,all_data['catalog_price-discount_price']

user_content = all_data.iloc[:len_tr,]
content_coupon = all_data.iloc[len_tr:,]

#差額が1円のクーポンは試供品で、Testにでないのと、8種類しかないのに25万個も売れているので取り除く。
user_content = user_content.iloc[~(user_content['catalog_price-discount_price']==1).values,:]
del user_content['catalog_price-discount_price']
content_coupon = content_coupon.iloc[~(content_coupon['catalog_price-discount_price']==1).values,:]
del content_coupon['catalog_price-discount_price']

#dp0を消す
#user_content = user_content.iloc[~(user_content['dp0']==1).values,:]
#del user_content['dp0']
#content_coupon = content_coupon.iloc[~(content_coupon['dp0']==1).values,:]
#del content_coupon['dp0']

#cd1_500を消す
#user_content = user_content.iloc[~(user_content['cd1_500']==1).values,:]
#del user_content['cd1_500']
#content_coupon = content_coupon.iloc[~(content_coupon['cd1_500']==1).values,:]
#del content_coupon['cd1_500']


del user_content['coupon_id_hash'], content_coupon['user_id_hash']

content_coupon = content_coupon.iloc[~(content_coupon['coupon_id_hash'].isin(['0c015306597566b632bebfb63b7e59f3','c988d799bc7db9254fe865ee6cf2d4ff'])).values,:]




'''
過去のユーザが買ったクーポンTF-IDFの平均情報
TF_jl = sum_M_r_jl / sum_L_M_r_jl

IDF = [IDF_1,...,IDF_L]
IDF_i = gamma / num_user

TF_IDF TF_jl.mul(IDF)#要素ごとにかける関数mul
'''
#まずは各特徴量のIDF_lを計算する
num_user = len(user_content['user_id_hash'].value_counts())#全ユーザ数
IDF = np.array([])#IDFを格納
for feature in user_content.columns.drop('user_id_hash'):
	gamma = len(user_content.iloc[(user_content[feature]==1).values,:].drop_duplicates('user_id_hash'))#γ^(i)_jlなるクーポンを含むユーザ数	
	IDF_l = -np.log2(gamma / float(num_user)) + 2
	IDF = np.append(IDF,IDF_l)

#次に各ユーザごとにTF-IDFを計算して、ユーザごとに集約（平均）する
def tf_idf(data):
	user_name = data['user_id_hash'].values[0]
	data_table = data[data.columns.drop('user_id_hash')]
	#print user_name, data_table
	sum_L_M_r_jl = data_table.sum().sum()#テーブルの全合計
	sum_M_r_jl = data_table.sum()#テーブルの列の合計
	TF_jl = sum_M_r_jl / float(sum_L_M_r_jl)#TFの値
	TF_IDF = TF_jl.mul(IDF)
	#print sum_L_M_r_jl,sum_M_r_jl,TF_jl
	#print TF_IDF
	return TF_IDF

a = user_content.groupby('user_id_hash').apply(tf_idf)

#a = user_content.groupby('user_id_hash').mean()
a = pd.DataFrame(a,columns=a.columns)
a['user_id_hash'] = a.index
#a.index = range(0,len(a))


'''
各ユーザーの過去に買ったクーポンとTestデータのクーポンの類似度計算
a: U*Content -> U
content_coupon: Content*Coupon -> V
'''
print a.columns.drop('user_id_hash') == content_coupon.columns.drop('coupon_id_hash')
U = a[a.columns.drop('user_id_hash')]
V = content_coupon[content_coupon.columns.drop('coupon_id_hash')]
V.index = content_coupon['coupon_id_hash']
print U.columns == V.columns

#直近1ヶ月に購入していないユーザーには宅配に重みをおいて推薦する
user_buy_1month = cpdtr.iloc[(cpdtr['i_date']>='2012-05-24').values,:]['user_id_hash'].drop_duplicates('user_id_hash')
#1ヶ月以内に購入していないユーザー
user_not_buy_1month = ulist.iloc[~(ulist['user_id_hash'].isin(user_buy_1month)).values]['user_id_hash'].values
#宅配クーポンの重み
takuhai_cp = cplte.iloc[(cplte['genre_name']=='宅配').values,:]['coupon_id_hash'].values

g = np.ones(len(V))
g[V.index.isin(takuhai_cp)] = 1.7



def convert_to_syuhen(data):
    if data in ['北海道']:
        return ['北海道','東京都','大阪府','沖縄県','神奈川県','千葉県']
    elif data in ['青森県']:
        return ['青森県','秋田県','岩手県','山形県','宮城県','北海道','東京都','大阪府','沖縄県','神奈川県','千葉県']
    elif data in ['秋田県','岩手県']:
        return ['青森県','秋田県','岩手県','山形県','宮城県','福島県','北海道','東京都','大阪府','沖縄県','神奈川県','千葉県']
    elif data in ['山形県','宮城県']:
        return ['秋田県','岩手県','山形県','宮城県','福島県','新潟県','北海道','東京都','大阪府','沖縄県','神奈川県','千葉県']
    elif data in ['福島県','新潟県']:
        return ['山形県','宮城県','福島県','新潟県','群馬県','栃木県','茨城県','北海道','東京都','大阪府','沖縄県','神奈川県','千葉県']
    elif data in ['群馬県','栃木県','茨城県']:
        return ['群馬県','栃木県','茨城県','福島県','新潟県','埼玉県','千葉県','長野県','北海道','東京都','大阪府','沖縄県','神奈川県','千葉県']
    elif data in ['埼玉県','千葉県','東京都','神奈川県']:
        return ['埼玉県','千葉県','東京都','神奈川県','山梨県','栃木県','北海道','東京都','大阪府','沖縄県','神奈川県','千葉県']
    elif data in ['山梨県','長野県','静岡県']:
        return ['埼玉県','千葉県','東京都','神奈川県','山梨県','栃木県','長野県','静岡県','北海道','東京都','大阪府','沖縄県','神奈川県','千葉県']
    elif data in ['愛知県','三重県','岐阜県']:
        return ['愛知県','三重県','岐阜県','奈良県','山梨県','長野県','和歌山県','北海道','東京都','大阪府','沖縄県','神奈川県','千葉県']
    elif data in ['富山県','石川県','福井県']:
        return ['富山県','石川県','福井県','新潟県','滋賀県','岐阜県','京都府','北海道','東京都','大阪府','沖縄県','神奈川県','千葉県']
    elif data in ['滋賀県','奈良県']:
        return ['滋賀県','京都府','奈良県','愛知県','三重県','岐阜県','大阪府','兵庫県','北海道','東京都','大阪府','沖縄県','神奈川県','千葉県']
    elif data in ['大阪府','和歌山県','京都府']:
        return ['滋賀県','京都府','奈良県','大阪府','和歌山県','京都府','兵庫県','北海道','東京都','大阪府','沖縄県','神奈川県','千葉県']
    elif data in ['兵庫県','岡山県','鳥取県']:
        return ['兵庫県','岡山県','広島県','京都府','奈良県','大阪府','京都府','鳥取県','島根県','北海道','東京都','大阪府','沖縄県','神奈川県','千葉県']
    elif data in ['島根県','広島県']:
        return ['兵庫県','岡山県','広島県','鳥取県','島根県','山口県','北海道','東京都','大阪府','沖縄県','神奈川県','千葉県']
    elif data in ['香川県','徳島県','愛媛県','高知県']:
        return ['香川県','徳島県','愛媛県','高知県','岡山県','大阪府','北海道','東京都','大阪府','沖縄県','神奈川県','千葉県']
    elif data in ['山口県']:
        return ['福岡県','佐賀県','長崎県','山口県','島根県','広島県','北海道','東京都','大阪府','沖縄県','神奈川県','千葉県']
    elif data in ['福岡県','佐賀県','長崎県']:
        return ['福岡県','佐賀県','長崎県','大分県','宮崎県','熊本県','鹿児島県','山口県','北海道','東京都','大阪府','沖縄県','神奈川県','千葉県']
    elif data in ['大分県','宮崎県','熊本県','鹿児島県']:
        return ['福岡県','佐賀県','長崎県','大分県','宮崎県','熊本県','鹿児島県','山口県','北海道','東京都','大阪府','沖縄県','神奈川県','千葉県']
    elif data in ['沖縄県']:
        return ['沖縄県','北海道','東京都','大阪府','沖縄県','神奈川県','千葉県']
    else:
        return []

#類似度計算する際にユーザの居住地周辺のクーポンから推薦する
def syuhen_rec(data):
	V2 = V.copy()
	
	#user_id
	user_id_ = data.name
	#user_idの居住地
	kyojuti = ulist.iloc[(ulist['user_id_hash']==user_id_).values,:]['pref_name'].values[0]
	print kyojuti
	if (kyojuti =='') or (type(kyojuti) == float):#居住地なし
		#coupon_listそのままで計算
                print type(kyojuti)
		return data.dot(V.T)
	else:
		#kyojuti周辺の県で掲載される(ユニークな)クーポンのリスト
		cplist_syuhen = cpate.iloc[(cpate['pref_name'].isin(convert_to_syuhen(kyojuti))).values,:].drop_duplicates('coupon_id_hash')['coupon_id_hash'].values
		V2.values[~(V2.index.isin(cplist_syuhen)),:]=0
		#print V2.iloc[~(V2.index.isin(cplist_syuhen)),:]
		print V2.sum(1).sum()/6.0#クーポン数 V2.sum(1),
		if user_id_ in user_not_buy_1month:
			#print user_id_
			return data.dot(V2.T)#.mul(g)#宅配に重みをかける
		else:
			return data.dot(V2.T)


#W_1month = np.diag(g)

#W_1month[V.index.isin(takuhai_cp),:]



cos_sim = U.apply(syuhen_rec,axis=1)








#cos_sim = ((U.T/((U*U).sum(1))**0.5).T).dot(((V.T/((V*V).sum(1))**0.5).T).T)

cp_top5_label = ['coupon_id_rank1','coupon_id_rank2','coupon_id_rank3','coupon_id_rank4','coupon_id_rank5','coupon_id_rank6','coupon_id_rank7','coupon_id_rank8','coupon_id_rank9','coupon_id_rank10']
def top5(data):
	#print rank
	#print data.Name
	#print type(data.argsort()[::-1])
	#print data.argsort()[::-1].values
	#print data[data.argsort()[::-1].values][:5].index.values
	return pd.Series(data[data.argsort()[::-1].values][:10].index.values, index=cp_top5_label)

prediction = cos_sim.apply(top5,axis=1)
prediction['user_id_hash'] = prediction.index

#trainに現れないユーザがいるのでとりあえずulistにMerge
print 'trainに現れないユーザ数:', len(ulist) - len(prediction) ,'ユーザ'
#trainに現れないユーザ
user_notin_tr = ulist.iloc[~(ulist['user_id_hash'].isin(prediction['user_id_hash'])).values,:]['user_id_hash'].values

pred_final = ulist.merge(prediction,on='user_id_hash',how='left')[['user_id_hash','coupon_id_rank1','coupon_id_rank2','coupon_id_rank3','coupon_id_rank4','coupon_id_rank5','coupon_id_rank6','coupon_id_rank7','coupon_id_rank8','coupon_id_rank9','coupon_id_rank10']]
pred_final = pred_final.sort('user_id_hash')
pred_final['user_id_hash'] = pred_final['user_id_hash'].astype(str)

'''
ユーザ間の類似度計算し
trainに現れないユーザに似ているユーザに推薦するアイテムを推薦する
'''
#trainに現れないユーザリスト
not_tr_user = pred_final.iloc[(pred_final['coupon_id_rank1'].isnull().values),:]['user_id_hash']

#userの行列作成
UU = ulist[[u'user_id_hash',  u'sex_id', u'age', u'pref_name']]
#変数のバイナリ化
for bi_label in ['sex_id', 'pref_name']:
	bi_data = pd.get_dummies(UU[bi_label])
	bi_data.columns = bi_label + '_' + bi_data.columns
	UU = pd.concat([UU, bi_data],axis=1)
	del UU[bi_label]

#ageのバイナリ化
age_dic = {'age10':pd.Series([1],index=['age10']),'age20':pd.Series([1],index=['age20']),
'age30':pd.Series([1],index=['age30']),'age40':pd.Series([1],index=['age40']),'age50':pd.Series([1],index=['age50']),
'age60':pd.Series([1],index=['age60']),'age70+':pd.Series([1],index=['age70+']),'age?':pd.Series([1],index=['age?'])}
def age_to_bi(data):
	#print data
	if data < 20:
		return age_dic['age10']
	elif 20 <= data < 30:
		return age_dic['age20']
	elif 30 <= data < 40:
		return age_dic['age30']
	elif 40 <= data < 50:
		return age_dic['age40']
	elif 50 <= data < 60:
		return age_dic['age50']
	elif 60 <= data < 70:
		return age_dic['age60']
	elif 70 <= data:
		return age_dic['age70+']
	else:
		return age_dic['age?']


UU = pd.concat([UU, UU['age'].apply(age_to_bi).fillna(0)],axis=1)
del UU['age']

UU.index = UU['user_id_hash'].values
del UU['user_id_hash']

user_sim = UU.dot(UU.T)

user_top10_label = ['user_id_rank1','user_id_rank2','user_id_rank3','user_id_rank4','user_id_rank5','user_id_rank6','user_id_rank7','user_id_rank8','user_id_rank9','user_id_rank10']
def top10_user(data):
	#print rank
	#print data.Name
	#print type(data.argsort()[::-1])
	#print data.argsort()[::-1].values
	#print data[data.argsort()[::-1].values][:5].index.values
	return pd.Series(data[data.argsort()[::-1].values][:10].index.values, index=user_top10_label)

user_sim2 = user_sim.apply(top10_user,axis=1)
user_sim2['user_id_hash'] = user_sim2.index

pred_final = pred_final.merge(user_sim2,on='user_id_hash',how='left')

#似ているユーザのクーポンを推薦する
num_user = 10
def sim_coupon(data):
	if type(data['coupon_id_rank1']) == str:
		print 0
		return 0
	data = data[user_top10_label].values
	for i in range(0,num_user):
		print type(pred_final.iloc[(pred_final['user_id_hash']==data[i]).values,:]['coupon_id_rank1'].values[0])
		if type(pred_final.iloc[(pred_final['user_id_hash']==data[i]).values,:]['coupon_id_rank1'].values[0]) == str:
			#print pred_final.iloc[(pred_final['user_id_hash']==data[i]).values,:][user_top10_label[:5]].values.tolist()[0]
			#print pd.Series(pred_final.iloc[(pred_final['user_id_hash']==data[i]).values,:][user_top10_label[:5]].values.tolist()[0],index=cp_top5_label)
			return pd.Series(pred_final.iloc[(pred_final['user_id_hash']==data[i]).values,:][cp_top5_label].values.tolist()[0],index=cp_top5_label)

pred_final.loc[(pred_final['coupon_id_rank1'].isnull().values),cp_top5_label] = pred_final.iloc[(pred_final['coupon_id_rank1'].isnull().values),:].apply(sim_coupon,axis=1).values
#trainに現れないユーザに挿入完了


def fff(data):
    a = data[['coupon_id_rank1','coupon_id_rank2','coupon_id_rank3','coupon_id_rank4','coupon_id_rank5','coupon_id_rank6','coupon_id_rank7','coupon_id_rank8','coupon_id_rank9','coupon_id_rank10']].values.tolist()[0]
    #print a
    print a
    b = ' '.join(a)
    #print b
    return pd.Series(b,index=['COUPON_ID_hash'])



o = pred_final
#'coupon_id_rank1','coupon_id_rank2','coupon_id_rank3','coupon_id_rank4','coupon_id_rank5']
user_notin_cpdtr = ulist.iloc[~(ulist['user_id_hash'].isin(cpdtr['user_id_hash'])).values,:]['user_id_hash']

o.loc[(o['user_id_hash'].isin(user_notin_cpdtr)).values,['coupon_id_rank3','coupon_id_rank4','coupon_id_rank5','coupon_id_rank6','coupon_id_rank7','coupon_id_rank8','coupon_id_rank9','coupon_id_rank10']] = o.loc[(o['user_id_hash'].isin(user_notin_cpdtr)).values,['coupon_id_rank1','coupon_id_rank2','coupon_id_rank3','coupon_id_rank4','coupon_id_rank5','coupon_id_rank6','coupon_id_rank7','coupon_id_rank8']].values
o.loc[(o['user_id_hash'].isin(user_notin_cpdtr)).values,['coupon_id_rank1','coupon_id_rank2']] = ['0c015306597566b632bebfb63b7e59f3','c988d799bc7db9254fe865ee6cf2d4ff']


pred_all = pd.DataFrame(columns=['user_id_hash','coupon_id_hash'])
pred_all = pd.concat([pred_all,pd.DataFrame(o[['user_id_hash','coupon_id_rank1']].values,columns=['user_id_hash','coupon_id_hash'])])
pred_all = pd.concat([pred_all,pd.DataFrame(o[['user_id_hash','coupon_id_rank2']].values,columns=['user_id_hash','coupon_id_hash'])])
pred_all = pd.concat([pred_all,pd.DataFrame(o[['user_id_hash','coupon_id_rank3']].values,columns=['user_id_hash','coupon_id_hash'])])
pred_all = pd.concat([pred_all,pd.DataFrame(o[['user_id_hash','coupon_id_rank4']].values,columns=['user_id_hash','coupon_id_hash'])])
pred_all = pd.concat([pred_all,pd.DataFrame(o[['user_id_hash','coupon_id_rank5']].values,columns=['user_id_hash','coupon_id_hash'])])
pred_all = pd.concat([pred_all,pd.DataFrame(o[['user_id_hash','coupon_id_rank6']].values,columns=['user_id_hash','coupon_id_hash'])])
pred_all = pd.concat([pred_all,pd.DataFrame(o[['user_id_hash','coupon_id_rank7']].values,columns=['user_id_hash','coupon_id_hash'])])
pred_all = pd.concat([pred_all,pd.DataFrame(o[['user_id_hash','coupon_id_rank8']].values,columns=['user_id_hash','coupon_id_hash'])])
pred_all = pd.concat([pred_all,pd.DataFrame(o[['user_id_hash','coupon_id_rank9']].values,columns=['user_id_hash','coupon_id_hash'])])
pred_all = pd.concat([pred_all,pd.DataFrame(o[['user_id_hash','coupon_id_rank10']].values,columns=['user_id_hash','coupon_id_hash'])])








pred_all['user_id_hash'] = pred_all['user_id_hash'].astype(str)












'''
高山コードとpred_coupon_numを結合しcount_weightを作成する
'''
#pred_allは最終推薦物で，bcが'coupon_id' と 'count', 'genre_name' などが格納されたDataFrame
ans = pred_all.copy()
ans=ans.sort('user_id_hash')
a2=ans.merge(cplte,how='left',on='coupon_id_hash')
a3=a2.merge(ulist,how='left',on = 'user_id_hash')
q=a3['coupon_id_hash'].value_counts()
r=pd.DataFrame()
r['coupon_id_hash']=q.index
r['count']=q.values
bc=r.merge(cplte,how="left",on='coupon_id_hash')

takayama = bc.copy()

pred_coupon_num['coupon_id_hash'] = pred_coupon_num['coupon_id_hash'].astype(str)
takayama = takayama.merge(pred_coupon_num, on ="coupon_id_hash",how="left")
pred_num = takayama[0].fillna(0).values
count_num = takayama['count'].values



count_weight = pd.DataFrame()
count_weight['coupon_id_hash'] = takayama['coupon_id_hash'].values
count_weight['count_weight'] = 1/(np.log10(count_num/(pred_num*(np.sum(count_num)/np.sum(pred_num))))/10 + 1)



count_weight1 = count_weight.copy()

'''
cos_simとcount_weightを掛け合わせて同じように推薦
'''
count_weight2 = count_weight1.iloc[~(count_weight1['coupon_id_hash'].isin(['0c015306597566b632bebfb63b7e59f3','c988d799bc7db9254fe865ee6cf2d4ff'])).values,:].copy()
count_weight2['count_weight'] = count_weight2['count_weight'].apply(lambda x: 5*x -4)
count_weight2.index = count_weight2['coupon_id_hash'].astype(str)

def top20_cos_sim_weight(data):
   top20_cos_sim_cp = data[data[data.argsort()[::-1].values][:15].index.values]
   count_weight_top = count_weight2.iloc[(count_weight2['coupon_id_hash'].isin(top20_cos_sim_cp.index.values)).values,:]
   return top20_cos_sim_cp.mul(count_weight_top['count_weight'].astype(float))

count_weight2['coupon_id_hash'] = count_weight2['coupon_id_hash'].astype(str)
cos_sim_weight1 = cos_sim.apply(top20_cos_sim_weight,axis=1).fillna(0)

count_weight2 = count_weight1.iloc[~(count_weight1['coupon_id_hash'].isin(['0c015306597566b632bebfb63b7e59f3','c988d799bc7db9254fe865ee6cf2d4ff'])).values,:].copy()
count_weight2['count_weight'] = count_weight2['count_weight'].apply(lambda x: 6*x -5)
count_weight2.index = count_weight2['coupon_id_hash'].astype(str)

def top20_cos_sim_weight(data):
   top20_cos_sim_cp = data[data[data.argsort()[::-1].values][:15].index.values]
   count_weight_top = count_weight2.iloc[(count_weight2['coupon_id_hash'].isin(top20_cos_sim_cp.index.values)).values,:]
   return top20_cos_sim_cp.mul(count_weight_top['count_weight'].astype(float))

count_weight2['coupon_id_hash'] = count_weight2['coupon_id_hash'].astype(str)
cos_sim_weight2 = cos_sim.apply(top20_cos_sim_weight,axis=1).fillna(0)

count_weight2 = count_weight1.iloc[~(count_weight1['coupon_id_hash'].isin(['0c015306597566b632bebfb63b7e59f3','c988d799bc7db9254fe865ee6cf2d4ff'])).values,:].copy()
count_weight2['count_weight'] = count_weight2['count_weight'].apply(lambda x: 4*x -3)
count_weight2.index = count_weight2['coupon_id_hash'].astype(str)

def top20_cos_sim_weight(data):
   top20_cos_sim_cp = data[data[data.argsort()[::-1].values][:15].index.values]
   count_weight_top = count_weight2.iloc[(count_weight2['coupon_id_hash'].isin(top20_cos_sim_cp.index.values)).values,:]
   return top20_cos_sim_cp.mul(count_weight_top['count_weight'].astype(float))

count_weight2['coupon_id_hash'] = count_weight2['coupon_id_hash'].astype(str)
cos_sim_weight3 = cos_sim.apply(top20_cos_sim_weight,axis=1).fillna(0)

count_weight2 = count_weight1.iloc[~(count_weight1['coupon_id_hash'].isin(['0c015306597566b632bebfb63b7e59f3','c988d799bc7db9254fe865ee6cf2d4ff'])).values,:].copy()
count_weight2['count_weight'] = count_weight2['count_weight'].apply(lambda x: 6*x -5)
count_weight2.index = count_weight2['coupon_id_hash'].astype(str)

def top20_cos_sim_weight(data):
   top20_cos_sim_cp = data[data[data.argsort()[::-1].values][:20].index.values]
   count_weight_top = count_weight2.iloc[(count_weight2['coupon_id_hash'].isin(top20_cos_sim_cp.index.values)).values,:]
   return top20_cos_sim_cp.mul(count_weight_top['count_weight'].astype(float))

count_weight2['coupon_id_hash'] = count_weight2['coupon_id_hash'].astype(str)
cos_sim_weight4 = cos_sim.apply(top20_cos_sim_weight,axis=1).fillna(0)

count_weight2 = count_weight1.iloc[~(count_weight1['coupon_id_hash'].isin(['0c015306597566b632bebfb63b7e59f3','c988d799bc7db9254fe865ee6cf2d4ff'])).values,:].copy()
count_weight2['count_weight'] = count_weight2['count_weight'].apply(lambda x: 4*x -3)
count_weight2.index = count_weight2['coupon_id_hash'].astype(str)

def top20_cos_sim_weight(data):
   top20_cos_sim_cp = data[data[data.argsort()[::-1].values][:20].index.values]
   count_weight_top = count_weight2.iloc[(count_weight2['coupon_id_hash'].isin(top20_cos_sim_cp.index.values)).values,:]
   return top20_cos_sim_cp.mul(count_weight_top['count_weight'].astype(float))

count_weight2['coupon_id_hash'] = count_weight2['coupon_id_hash'].astype(str)
cos_sim_weight5 = cos_sim.apply(top20_cos_sim_weight,axis=1).fillna(0)

count_weight2 = count_weight1.iloc[~(count_weight1['coupon_id_hash'].isin(['0c015306597566b632bebfb63b7e59f3','c988d799bc7db9254fe865ee6cf2d4ff'])).values,:].copy()
count_weight2['count_weight'] = count_weight2['count_weight'].apply(lambda x: 5*x -4)
count_weight2.index = count_weight2['coupon_id_hash'].astype(str)

def top20_cos_sim_weight(data):
   top20_cos_sim_cp = data[data[data.argsort()[::-1].values][:10].index.values]
   count_weight_top = count_weight2.iloc[(count_weight2['coupon_id_hash'].isin(top20_cos_sim_cp.index.values)).values,:]
   return top20_cos_sim_cp.mul(count_weight_top['count_weight'].astype(float))

count_weight2['coupon_id_hash'] = count_weight2['coupon_id_hash'].astype(str)
cos_sim_weight6 = cos_sim.apply(top20_cos_sim_weight,axis=1).fillna(0)







cos_sim_weight = (cos_sim_weight1 + cos_sim_weight2 + cos_sim_weight3 + cos_sim_weight4 + cos_sim_weight5 + cos_sim_weight6)/6.0


#cos_sim_weight = cos_sim_weight1
#cos_sim_weight = (cos_sim_weight1 + cos_sim_weight2 + cos_sim_weight3 + cos_sim_weight4 + cos_sim_weight5 + cos_sim_weight6 + cos_sim_weight7 + cos_sim_weight8 )/8.0

print cos_sim_weight


'''
新たに推薦し直す
'''
cp_top5_label = ['coupon_id_rank1','coupon_id_rank2','coupon_id_rank3','coupon_id_rank4','coupon_id_rank5','coupon_id_rank6','coupon_id_rank7','coupon_id_rank8','coupon_id_rank9','coupon_id_rank10']
def top5(data):
	#print rank
	#print data.Name
	#print type(data.argsort()[::-1])
	#print data.argsort()[::-1].values
	#print data[data.argsort()[::-1].values][:5].index.values
	return pd.Series(data[data.argsort()[::-1].values][:10].index.values, index=cp_top5_label)

prediction = cos_sim_weight.apply(top5,axis=1)
prediction['user_id_hash'] = prediction.index

#trainに現れないユーザがいるのでとりあえずulistにMerge
print 'trainに現れないユーザ数:', len(ulist) - len(prediction) ,'ユーザ'
#trainに現れないユーザ
user_notin_tr = ulist.iloc[~(ulist['user_id_hash'].isin(prediction['user_id_hash'])).values,:]['user_id_hash'].values

pred_final = ulist.merge(prediction,on='user_id_hash',how='left')[['user_id_hash','coupon_id_rank1','coupon_id_rank2','coupon_id_rank3','coupon_id_rank4','coupon_id_rank5','coupon_id_rank6','coupon_id_rank7','coupon_id_rank8','coupon_id_rank9','coupon_id_rank10']]
pred_final = pred_final.sort('user_id_hash')
pred_final['user_id_hash'] = pred_final['user_id_hash'].astype(str)

'''
ユーザ間の類似度計算し
trainに現れないユーザに似ているユーザに推薦するアイテムを推薦する
'''
#trainに現れないユーザリスト
not_tr_user = pred_final.iloc[(pred_final['coupon_id_rank1'].isnull().values),:]['user_id_hash']

#userの行列作成
UU = ulist[[u'user_id_hash',  u'sex_id', u'age', u'pref_name']]
#変数のバイナリ化
for bi_label in ['sex_id', 'pref_name']:
	bi_data = pd.get_dummies(UU[bi_label])
	bi_data.columns = bi_label + '_' + bi_data.columns
	UU = pd.concat([UU, bi_data],axis=1)
	del UU[bi_label]

#ageのバイナリ化
age_dic = {'age10':pd.Series([1],index=['age10']),'age20':pd.Series([1],index=['age20']),
'age30':pd.Series([1],index=['age30']),'age40':pd.Series([1],index=['age40']),'age50':pd.Series([1],index=['age50']),
'age60':pd.Series([1],index=['age60']),'age70+':pd.Series([1],index=['age70+']),'age?':pd.Series([1],index=['age?'])}
def age_to_bi(data):
	#print data
	if data < 20:
		return age_dic['age10']
	elif 20 <= data < 30:
		return age_dic['age20']
	elif 30 <= data < 40:
		return age_dic['age30']
	elif 40 <= data < 50:
		return age_dic['age40']
	elif 50 <= data < 60:
		return age_dic['age50']
	elif 60 <= data < 70:
		return age_dic['age60']
	elif 70 <= data:
		return age_dic['age70+']
	else:
		return age_dic['age?']


UU = pd.concat([UU, UU['age'].apply(age_to_bi).fillna(0)],axis=1)
del UU['age']

UU.index = UU['user_id_hash'].values
del UU['user_id_hash']

user_sim = UU.dot(UU.T)

user_top10_label = ['user_id_rank1','user_id_rank2','user_id_rank3','user_id_rank4','user_id_rank5','user_id_rank6','user_id_rank7','user_id_rank8','user_id_rank9','user_id_rank10']
def top10_user(data):
	#print rank
	#print data.Name
	#print type(data.argsort()[::-1])
	#print data.argsort()[::-1].values
	#print data[data.argsort()[::-1].values][:5].index.values
	return pd.Series(data[data.argsort()[::-1].values][:10].index.values, index=user_top10_label)

user_sim2 = user_sim.apply(top10_user,axis=1)
user_sim2['user_id_hash'] = user_sim2.index

pred_final = pred_final.merge(user_sim2,on='user_id_hash',how='left')

#似ているユーザのクーポンを推薦する
num_user = 10
def sim_coupon(data):
	if type(data['coupon_id_rank1']) == str:
		print 0
		return 0
	data = data[user_top10_label].values
	for i in range(0,num_user):
		print type(pred_final.iloc[(pred_final['user_id_hash']==data[i]).values,:]['coupon_id_rank1'].values[0])
		if type(pred_final.iloc[(pred_final['user_id_hash']==data[i]).values,:]['coupon_id_rank1'].values[0]) == str:
			#print pred_final.iloc[(pred_final['user_id_hash']==data[i]).values,:][user_top10_label[:5]].values.tolist()[0]
			#print pd.Series(pred_final.iloc[(pred_final['user_id_hash']==data[i]).values,:][user_top10_label[:5]].values.tolist()[0],index=cp_top5_label)
			return pd.Series(pred_final.iloc[(pred_final['user_id_hash']==data[i]).values,:][cp_top5_label].values.tolist()[0],index=cp_top5_label)

pred_final.loc[(pred_final['coupon_id_rank1'].isnull().values),cp_top5_label] = pred_final.iloc[(pred_final['coupon_id_rank1'].isnull().values),:].apply(sim_coupon,axis=1).values
#trainに現れないユーザに挿入完了


def fff(data):
    a = data[['coupon_id_rank1','coupon_id_rank2','coupon_id_rank3','coupon_id_rank4','coupon_id_rank5','coupon_id_rank6','coupon_id_rank7','coupon_id_rank8','coupon_id_rank9','coupon_id_rank10']].values.tolist()[0]
    #print a
    print a
    b = ' '.join(a)
    #print b
    return pd.Series(b,index=['COUPON_ID_hash'])



o = pred_final.copy()

#閲覧回数1会の人に人気商品トップ10を推薦する
a = all_data['user_id_hash'].value_counts()
times1_user = a[(a==1).values].index.values

top10_cps = o['coupon_id_rank1'].value_counts()[:10].index.values 

#o.loc[(o['user_id_hash'].isin(times1_user)).values,['coupon_id_rank1','coupon_id_rank2','coupon_id_rank3','coupon_id_rank4','coupon_id_rank5','coupon_id_rank6','coupon_id_rank7','coupon_id_rank8','coupon_id_rank9','coupon_id_rank10']] = top10_cps






#'coupon_id_rank1','coupon_id_rank2','coupon_id_rank3','coupon_id_rank4','coupon_id_rank5']
user_notin_cpdtr = ulist.iloc[~(ulist['user_id_hash'].isin(cpdtr['user_id_hash'])).values,:]['user_id_hash']

o.loc[(o['user_id_hash'].isin(user_notin_cpdtr)).values,['coupon_id_rank3','coupon_id_rank4','coupon_id_rank5','coupon_id_rank6','coupon_id_rank7','coupon_id_rank8','coupon_id_rank9','coupon_id_rank10']] = o.loc[(o['user_id_hash'].isin(user_notin_cpdtr)).values,['coupon_id_rank1','coupon_id_rank2','coupon_id_rank3','coupon_id_rank4','coupon_id_rank5','coupon_id_rank6','coupon_id_rank7','coupon_id_rank8']].values
o.loc[(o['user_id_hash'].isin(user_notin_cpdtr)).values,['coupon_id_rank1','coupon_id_rank2']] = ['0c015306597566b632bebfb63b7e59f3','c988d799bc7db9254fe865ee6cf2d4ff']




pred_ikki = o[['user_id_hash','coupon_id_rank1','coupon_id_rank2','coupon_id_rank3','coupon_id_rank4','coupon_id_rank5','coupon_id_rank6','coupon_id_rank7','coupon_id_rank8','coupon_id_rank9','coupon_id_rank10']].fillna('').groupby('user_id_hash').apply(fff)
pred_ikki['USER_ID_hash'] = pred_ikki.index
pred_ikki = pred_ikki[['USER_ID_hash','COUPON_ID_hash']]
pred_ikki.index = range(0,len(pred_ikki))
pred_ikki.columns = [['USER_ID_hash','PURCHASED_COUPONS']]
pred_ikki.to_csv('/Users/IkkiTanaka/Documents/kaggle/Coupon_Purchase_Prediction/prediction/ikki0911_pred42.csv',index=False)


'''
予測結果を見る

'''

t = pred_all['coupon_id_hash'][:(len(ulist)*2)].value_counts()
t = pd.DataFrame(t)
t['coupon_id_hash'] = t.index.values
t.merge(cplte,on='coupon_id_hash',how='left')


!diff /Users/IkkiTanaka/Documents/kaggle/Coupon_Purchase_Prediction/prediction/ikki0911_pred41.csv /Users/IkkiTanaka/Documents/kaggle/Coupon_Purchase_Prediction/prediction/ikki0911_pred35.csv


