import numpy as np
import pandas as pd
import psycopg2
'''
0907の
割引額が1円の試供品の情報を削除したTrainデータで
ユーザの情報を集約するときにTF-IDFを用いて計算

0909_1900追加分
discount_price,price_rateの細かなバイナリ化
'''

#edit user and password!!
conn = psycopg2.connect("dbname=intern host=itn-redshift-02.cb9dz5j1yugx.ap-northeast-1.redshift.amazonaws.com user=itn_tanaka_ikki password=t8)&a3YYxMyB%6f port=5439")

#answer_sample
cur = conn.cursor()
cur.execute("SELECT * FROM answer_sample;")
a = cur.fetchall()
colnames = [desc[0] for desc in cur.description]
answer_sample = pd.DataFrame(a, columns=colnames)
cur.close()

#dnld_coupon_area_test
cur = conn.cursor()
cur.execute("SELECT * FROM dnld_coupon_area_test;")
a = cur.fetchall()
colnames = [desc[0] for desc in cur.description]
cpate = pd.DataFrame(a, columns=colnames)
cur.close()

#dnld_coupon_area_train
cur = conn.cursor()
cur.execute("SELECT * FROM dnld_coupon_area_train;")
a = cur.fetchall()
colnames = [desc[0] for desc in cur.description]
cpatr = pd.DataFrame(a, columns=colnames)
cur.close()

#dnld_coupon_list_test
cur = conn.cursor()
cur.execute("SELECT * FROM dnld_coupon_list_test;")
a = cur.fetchall()
colnames = [desc[0] for desc in cur.description]
cplte = pd.DataFrame(a, columns=colnames)
cur.close()

#dnld_coupon_list_train
cur = conn.cursor()
cur.execute("SELECT * FROM dnld_coupon_list_train;")
a = cur.fetchall()
colnames = [desc[0] for desc in cur.description]
cpltr = pd.DataFrame(a, columns=colnames)
cur.close()

#dnld_coupon_txt_test
cur = conn.cursor()
cur.execute("SELECT * FROM dnld_coupon_txt_test;")
a = cur.fetchall()
colnames = [desc[0] for desc in cur.description]
cptte = pd.DataFrame(a, columns=colnames)
cur.close()

#dnld_coupon_txt_train
cur = conn.cursor()
cur.execute("SELECT * FROM dnld_coupon_txt_train;")
a = cur.fetchall()
colnames = [desc[0] for desc in cur.description]
cpttr = pd.DataFrame(a, columns=colnames)
cur.close()

#dnld_coupon_visits_train
cur = conn.cursor()
cur.execute("SELECT * FROM dnld_coupon_visits_train;")
a = cur.fetchall()
colnames = [desc[0] for desc in cur.description]
cpvtr = pd.DataFrame(a, columns=colnames)
cur.close()
cpvtr.columns = [u'session_id', u'user_id', u'coupon_id', u'purchase_flg',
 u'purchaseid', u'i_date', u'url', u'referrer', u'page_serial']
#dnld_coupon_detail_train
cur = conn.cursor()
cur.execute("SELECT * FROM dnld_coupon_detail_train;")
a = cur.fetchall()
colnames = [desc[0] for desc in cur.description]
cpdtr = pd.DataFrame(a, columns=colnames)
cur.close()

#dnld_user_list
cur = conn.cursor()
cur.execute("SELECT * FROM dnld_user_list;")
a = cur.fetchall()
colnames = [desc[0] for desc in cur.description]
ulist = pd.DataFrame(a, columns=colnames)
cur.close()

'''
%whosで変数確認できます
'''

#朝言ったユーザーの買ったアイテムと推薦するアイテムの類似度

#userの購入情報にクーポンの詳細をつける
user_item_tr = cpdtr.merge(cpltr,on='coupon_id',how='left')

#特徴量を選択する(small_area_name_xは購入者の場所)
feature_label = [u'coupon_id', u'user_id',u'genre_name',u'large_area_name', u'ken_name',u'small_area_name_y','discount_price','price_rate','catalog_price']
all_data = user_item_tr[feature_label]
all_data.columns = [u'coupon_id', u'user_id',u'genre_name',u'large_area_name', u'ken_name',u'small_area_name','discount_price','price_rate','catalog_price']

#trainとtestを結合する
feature_label_te = [u'coupon_id',u'genre_name',u'large_area_name', u'ken_name',u'small_area_name','discount_price','price_rate','catalog_price']
#結合する前にTrainの長さを保存する
len_tr = len(all_data)
all_data = pd.concat([all_data,cplte[feature_label_te]])
all_data.index = range(0,len(all_data))

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
cat_dis_dic = {'cd0':pd.Series([1],index=['cd0']),'cd1':pd.Series([1],index=['cd1']),
'cd10':pd.Series([1],index=['cd10']),'cd100':pd.Series([1],index=['cd100']),'cd1000':pd.Series([1],index=['cd1000']),
'cd10000':pd.Series([1],index=['cd10000']),'cd100000':pd.Series([1],index=['cd100000'])}

def cat_dis_to_bi(data):
	#print data
	if data == 0:
		return cat_dis_dic['cd0']
	elif 0 < data < 10:
		return cat_dis_dic['cd1']
	elif 10 <= data < 100:
		return cat_dis_dic['cd10']
	elif 100 <= data < 1000:
		return cat_dis_dic['cd100']
	elif 1000 <= data < 10000:
		return cat_dis_dic['cd1000']
	elif 10000 <= data < 100000:
		return cat_dis_dic['cd10000']
	elif 100000 <= data:
		return cat_dis_dic['cd100000']

#all_data[['cd0','cd1','cd100','cd1000','cd10000','cd100000']] = all_data['catalog_price-discount_price'].apply(cat_dis_to_bi).fillna(0)
del all_data['discount_price'],all_data['catalog_price']#,all_data['catalog_price-discount_price']

user_content = all_data.iloc[:len_tr,]
content_coupon = all_data.iloc[len_tr:,]

#差額が1円のクーポンは試供品で、Testにでないのと、8種類しかないのに25万個も売れているので取り除く。
user_content = user_content.iloc[~(user_content['catalog_price-discount_price']==1).values,:]
del user_content['catalog_price-discount_price']
content_coupon = content_coupon.iloc[~(content_coupon['catalog_price-discount_price']==1).values,:]
del content_coupon['catalog_price-discount_price']




del user_content['coupon_id'], content_coupon['user_id']

'''
過去のユーザが買ったクーポンTF-IDFの平均情報
TF_jl = sum_M_r_jl / sum_L_M_r_jl

IDF = [IDF_1,...,IDF_L]
IDF_i = gamma / num_user

TF_IDF TF_jl.mul(IDF)#要素ごとにかける関数mul
'''
#まずは各特徴量のIDF_lを計算する
num_user = len(user_content['user_id'].value_counts())#全ユーザ数
IDF = np.array([])#IDFを格納
for feature in user_content.columns.drop('user_id'):
	gamma = len(user_content.iloc[(user_content[feature]==1).values,:].drop_duplicates('user_id'))#γ^(i)_jlなるクーポンを含むユーザ数	
	IDF_l = -np.log2(gamma / float(num_user)) + 1
	IDF = np.append(IDF,IDF_l)

#次に各ユーザごとにTF-IDFを計算して、ユーザごとに集約（平均）する
def tf_idf(data):
	user_name = data['user_id'].values[0]
	data_table = data[data.columns.drop('user_id')]
	#print user_name, data_table
	sum_L_M_r_jl = data_table.sum().sum()#テーブルの全合計
	sum_M_r_jl = data_table.sum()#テーブルの列の合計
	TF_jl = sum_M_r_jl / float(sum_L_M_r_jl)#TFの値
	TF_IDF = TF_jl.mul(IDF)
	#print sum_L_M_r_jl,sum_M_r_jl,TF_jl
	#print TF_IDF
	return TF_IDF

a = user_content.groupby('user_id').apply(tf_idf)

#a = user_content.groupby('user_id').mean()
a = pd.DataFrame(a,columns=a.columns)
a['user_id'] = a.index
#a.index = range(0,len(a))


'''
各ユーザーの過去に買ったクーポンとTestデータのクーポンの類似度計算
a: U*Content -> U
content_coupon: Content*Coupon -> V
'''
print a.columns.drop('user_id') == content_coupon.columns.drop('coupon_id')
U = a[a.columns.drop('user_id')]
V = content_coupon[content_coupon.columns.drop('coupon_id')]
V.index = content_coupon['coupon_id']
print U.columns == V.columns
cos_sim = ((U.T/((U*U).sum(1))**0.5).T).dot(((V.T/((V*V).sum(1))**0.5).T).T)

cp_top5_label = ['coupon_id_rank1','coupon_id_rank2','coupon_id_rank3','coupon_id_rank4','coupon_id_rank5']
def top5(data):
	#print rank
	#print data.Name
	#print type(data.argsort()[::-1])
	#print data.argsort()[::-1].values
	#print data[data.argsort()[::-1].values][:5].index.values
	return pd.Series(data[data.argsort()[::-1].values][:5].index.values, index=cp_top5_label)

prediction = cos_sim.apply(top5,axis=1)
prediction['user_id'] = prediction.index

#trainに現れないユーザがいるのでとりあえずulistにMerge
print 'trainに現れないユーザ数:', len(ulist) - len(prediction) ,'ユーザ'
pred_final = ulist.merge(prediction,on='user_id',how='left')[['user_id','coupon_id_rank1','coupon_id_rank2','coupon_id_rank3','coupon_id_rank4','coupon_id_rank5']]
pred_final = pred_final.sort('user_id')
pred_final['user_id'] = pred_final['user_id'].astype(str)

'''
ユーザ間の類似度計算し
trainに現れないユーザに似ているユーザに推薦するアイテムを推薦する
'''
#trainに現れないユーザリスト
not_tr_user = pred_final.iloc[(pred_final['coupon_id_rank1'].isnull().values),:]['user_id']

#userの行列作成
UU = ulist[[u'user_id',  u'sex_id', u'age', u'pref_name']]
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

UU.index = UU['user_id'].values
del UU['user_id']

user_sim = UU.dot(UU.T)

user_top10_label = ['user_id_rank1','user_id_rank2','user_id_rank3','user_id_rank4','user_id_rank5',
'user_id_rank6','user_id_rank7','user_id_rank8','user_id_rank9','user_id_rank10']
def top10_user(data):
	#print rank
	#print data.Name
	#print type(data.argsort()[::-1])
	#print data.argsort()[::-1].values
	#print data[data.argsort()[::-1].values][:5].index.values
	return pd.Series(data[data.argsort()[::-1].values][:10].index.values, index=user_top10_label)

user_sim2 = user_sim.apply(top10_user,axis=1)
user_sim2['user_id'] = user_sim2.index

pred_final = pred_final.merge(user_sim2,on='user_id',how='left')

#似ているユーザのクーポンを推薦する
num_user = 10
def sim_coupon(data):
	if type(data['coupon_id_rank1']) == str:
		print 0
		return 0
	data = data[user_top10_label].values
	for i in range(0,num_user):
		print type(pred_final.iloc[(pred_final['user_id']==data[i]).values,:]['coupon_id_rank1'].values[0])
		if type(pred_final.iloc[(pred_final['user_id']==data[i]).values,:]['coupon_id_rank1'].values[0]) == str:
			#print pred_final.iloc[(pred_final['user_id']==data[i]).values,:][user_top10_label[:5]].values.tolist()[0]
			#print pd.Series(pred_final.iloc[(pred_final['user_id']==data[i]).values,:][user_top10_label[:5]].values.tolist()[0],index=cp_top5_label)
			return pd.Series(pred_final.iloc[(pred_final['user_id']==data[i]).values,:][cp_top5_label].values.tolist()[0],index=cp_top5_label)

pred_final.loc[(pred_final['coupon_id_rank1'].isnull().values),cp_top5_label] = pred_final.iloc[(pred_final['coupon_id_rank1'].isnull().values),:].apply(sim_coupon,axis=1).values
#trainに現れないユーザに挿入完了


pred_all = pd.DataFrame(columns=['user_id','coupon_id'])
pred_all = pd.concat([pred_all,pd.DataFrame(pred_final[['user_id','coupon_id_rank1']].values,columns=['user_id','coupon_id'])])
pred_all = pd.concat([pred_all,pd.DataFrame(pred_final[['user_id','coupon_id_rank2']].values,columns=['user_id','coupon_id'])])
pred_all = pd.concat([pred_all,pd.DataFrame(pred_final[['user_id','coupon_id_rank3']].values,columns=['user_id','coupon_id'])])
pred_all = pd.concat([pred_all,pd.DataFrame(pred_final[['user_id','coupon_id_rank4']].values,columns=['user_id','coupon_id'])])
pred_all = pd.concat([pred_all,pd.DataFrame(pred_final[['user_id','coupon_id_rank5']].values,columns=['user_id','coupon_id'])])

pred_all['user_id'] = pred_all['user_id'].astype(str)

pred_all.to_csv('output0909_1900.csv',index=False)

'''
97.XXXの結果と比較
x = pd.read_csv('output0909_1700.csv')
y = pd.read_csv('output0909_1500.csv')
z = pd.read_csv('output0907.csv')

x.iloc[(x['user_id']=='0004302b').values,:]
y.iloc[(y['user_id']=='0004302b').values,:]
'''


