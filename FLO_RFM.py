import datetime
###############################################################
# RFM ile Müşteri Segmentasyonu (Customer Segmentation with RFM)
###############################################################

###############################################################
# İş Problemi (Business Problem)
###############################################################
# FLO müşterilerini segmentlere ayırıp bu segmentlere göre pazarlama stratejileri belirlemek istiyor.
# Buna yönelik olarak müşterilerin davranışları tanımlanacak ve bu davranış öbeklenmelerine göre gruplar oluşturulacak..

###############################################################
# Veri Seti Hikayesi
###############################################################

# Veri seti son alışverişlerini 2020 - 2021 yıllarında OmniChannel(hem online hem offline alışveriş yapan) olarak yapan müşterilerin geçmiş alışveriş davranışlarından
# elde edilen bilgilerden oluşmaktadır.

# master_id: Eşsiz müşteri numarası
# order_channel : Alışveriş yapılan platforma ait hangi kanalın kullanıldığı (Android, ios, Desktop, Mobile, Offline)
# last_order_channel : En son alışverişin yapıldığı kanal
# first_order_date : Müşterinin yaptığı ilk alışveriş tarihi
# last_order_date : Müşterinin yaptığı son alışveriş tarihi
# last_order_date_online : Muşterinin online platformda yaptığı son alışveriş tarihi
# last_order_date_offline : Muşterinin offline platformda yaptığı son alışveriş tarihi
# order_num_total_ever_online : Müşterinin online platformda yaptığı toplam alışveriş sayısı
# order_num_total_ever_offline : Müşterinin offline'da yaptığı toplam alışveriş sayısı
# customer_value_total_ever_offline : Müşterinin offline alışverişlerinde ödediği toplam ücret
# customer_value_total_ever_online : Müşterinin online alışverişlerinde ödediği toplam ücret
# interested_in_categories_12 : Müşterinin son 12 ayda alışveriş yaptığı kategorilerin listesi

###############################################################
# GÖREVLER
###############################################################
from datetime import date
import datetime as dt
import pandas as pd
import numpy as np
pd.set_option('display.max_columns', None)
#pd.set_option('display.max_rows', None)
pd.set_option('display.float_format', lambda x: '%.3f' % x)


# VERİYİ ANLAMA (DATA UNDERSTANDING )VE HAZIRLAMA
#Adım1:   flo_data_20K.csv verisiniokuyunuz.Dataframe’inkopyasınıoluşturunuz.
df_ = pd.read_csv("/Users/ardaugurlu/Documents/miuul/crmAnalytics/datasets/flo_data_20k.csv")
df = df_.copy()

#Adım2:   a) Veri setindea. İlk 10 gözlem

df.head(10)

# b) Değişken isimleri
df.columns

# c) Betimsel istatistik

df.describe().T

# d) Boşdeğer

df.isnull().sum()

# e) değişken tipleri

df.info()
#Adım3 . Omnichannel müşterilerin hem online'dan hemde offline platformlardan alışveriş yaptığını ifade etmektedir.
# Herbir müşterinin toplam  alışveriş sayısı ve harcaması için yeni değişkenler oluşturun.

df["Total_order"] = df["order_num_total_ever_online"] + df["order_num_total_ever_offline"]
df["Total_value"] = df["customer_value_total_ever_online"] + df["customer_value_total_ever_offline"]

# 4. Değişken tiplerini inceleyiniz. Tarih ifade eden değişkenlerin tipini date'e çeviriniz.




df["first_order_date"] = pd.to_datetime(df["first_order_date"])
df["last_order_date"] =  pd.to_datetime(df["last_order_date"])
df["last_order_date_online"] =  pd.to_datetime(df["last_order_date_online"])
df["last_order_date_offline"] =  pd.to_datetime(df["last_order_date_offline"])
df.info()

# first_order_date : Müşterinin yaptığı ilk alışveriş tarihi
# last_order_date : Müşterinin yaptığı son alışveriş tarihi
# last_order_date_online : Muşterinin online platformda yaptığı son alışveriş tarihi
# last_order_date_offline : Muşterinin offline platformda yaptığı son alışveriş tarihi

# 5. Alışveriş kanallarındaki müşteri sayısının, ortalama alınan ürün sayısının ve ortalama harcamaların dağılımına bakınız.
df.groupby("order_channel").agg({"master_id" : lambda x : x.nunique(),
                                 "Total_order": "mean",
                                 "Total_value": "mean"})


# 6. En fazla kazancı getiren ilk 10 müşteriyi sıralayınız.
df.groupby("master_id").agg({"Total_value" : "sum"}).sort_values("Total_value", ascending=False).head(10)
#or
df.groupby("master_id")["Total_order"].sum().nlargest(10)

# 7. En fazla siparişi veren ilk 10 müşteriyi sıralayınız.

df.groupby("master_id").agg({"Total_order" : "sum"}).sort_values("Total_order", ascending=False).head(10)
df.groupby("master_id")["Total_order"].sum().nlargest(10)





###############################################################
# GÖREV 2: RFM Metriklerinin Hesaplanması
###############################################################
# Veri setindeki en son alışverişin yapıldığı tarihten 2 gün sonrasını analiz tarihi



df["last_order_date"].max()
today_date = dt.datetime(2021, 6, 2)

#customer_id, recency, frequnecy ve monetary değerlerinin yer aldığı yeni bir rfm dataframe

rfm = df.groupby("master_id").agg({"last_order_date" : lambda x : (today_date - x.max()).days,
                                  "Total_order" :  lambda Total_order: Total_order.sum(),
                                   "Total_value" : lambda Total_value : Total_value.sum()})

rfm.columns = ["recency", "frequency", "monetary"]

rfm.head()

###############################################################
# GÖREV 3: RF ve RFM Skorlarının Hesaplanması (Calculating RF and RFM Scores)
###############################################################

#  Recency, Frequency ve Monetary metriklerini qcut yardımı ile 1-5 arasında skorlara çevrilmesi ve
# Bu skorları recency_score, frequency_score ve monetary_score olarak kaydedilmesi

rfm["recency_score"] = pd.qcut(rfm["recency"], 5, labels=[5,4,3,2,1])

rfm["frequency_score"] = pd.qcut(rfm["frequency"].rank(method = "first"), 5, labels=[1,2,3,4,5])

rfm["monetary_score"] = pd.qcut(rfm["monetary"], 5, labels=[1,2,3,4,5])



# recency_score ve frequency_score’u tek bir değişken olarak ifade edilmesi ve RF_SCORE olarak kaydedilmesi

rfm["rf_score"] = (rfm["recency_score"].astype(str) +
                    rfm["frequency_score"].astype(str))


###############################################################
# GÖREV 4: RF Skorlarının Segment Olarak Tanımlanması
###############################################################
seg_map = {
    r'[1-2][1-2]': 'hibernating',
    r'[1-2][3-4]': 'at_Risk',
    r'[1-2]5': 'cant_loose',
    r'3[1-2]': 'about_to_sleep',
    r'33': 'need_attention',
    r'[3-4][4-5]': 'loyal_customers',
    r'41': 'promising',
    r'51': 'new_customers',
    r'[4-5][2-3]': 'potential_loyalists',
    r'5[4-5]': 'champions'}

rfm["segment"] = rfm["rf_score"].replace(seg_map, regex = True)

rfm[rfm["rf_score"]== "55"]

###############################################################
# GÖREV 5: Aksiyon zamanı!
###############################################################

# 1. Segmentlerin recency, frequnecy ve monetary ortalamalarını inceleyiniz.

rfm[["segment", "recency", "frequency", "monetary"]].groupby("segment").agg(["mean"])





# 2. RFM analizi yardımı ile 2 case için ilgili profildeki müşterileri bulunuz ve müşteri id'lerini csv ye kaydediniz.

# a. FLO bünyesine yeni bir kadın ayakkabı markası dahil ediyor. Dahil ettiği markanın ürün fiyatları genel müşteri tercihlerinin üstünde.
# Bu nedenle markanın
# tanıtımı ve ürün satışları için ilgilenecek profildeki müşterilerle özel olarak iletişime geçeilmek isteniliyor. Bu müşterilerin sadık  ve champion
# kadın kategorisinden alışveriş yapan kişiler olması planlandı. Müşterilerin id numaralarını csv dosyasına yeni_marka_hedef_müşteri_id.cvs
# olarak kaydediniz.


rfm.shape
rfm["interested_in_categories"] = df["interested_in_categories_12"].values


rfm[((rfm["segment"] == "loyal_customers") | (rfm["segment"] == "champions")) & (rfm["interested_in_categories"].str.contains("KADIN"))]

new_df =pd.DataFrame()

new_df["customer_id"] = rfm[((rfm["segment"] == "loyal_customers") | (rfm["segment"] == "champions")) & (rfm["interested_in_categories"]== "[KADIN]")].index

new_df.to_csv("customer_id_for_women_and_loyal")

# b. Erkek ve Çoçuk ürünlerinde %40'a yakın indirim planlanmaktadır. Bu indirimle ilgili kategorilerle ilgilenen geçmişte iyi müşterilerden olan ama uzun süredir
# alışveriş yapmayan ve yeni gelen müşteriler özel olarak hedef alınmak isteniliyor. Uygun profildeki müşterilerin id'lerini csv dosyasına indirim_hedef_müşteri_ids.csv
# olarak kaydediniz.


 new_df_1 = pd.DataFrame()

 new_df_1["new_customer_id"] =  rfm[((rfm["segment"] == "cant_loose") | (rfm["segment"] ==  "new_customers")) & ((rfm["interested_in_categories"].str.contains("ERKEK")) & (rfm["interested_in_categories"].str.contains("COCUK")))].index


#######################################################



##############################################################
# BG-NBD ve Gamma-Gamma ile CLTV Prediction
##############################################################

###############################################################
# İş Problemi (Business Problem)
###############################################################
# FLO satış ve pazarlama faaliyetleri için roadmap belirlemek istemektedir.
# Şirketin orta uzun vadeli plan yapabilmesi için var olan müşterilerin gelecekte şirkete sağlayacakları potansiyel değerin tahmin edilmesi gerekmektedir.


###############################################################
# Veri Seti Hikayesi
###############################################################

# Veri seti son alışverişlerini 2020 - 2021 yıllarında OmniChannel(hem online hem offline alışveriş yapan) olarak yapan müşterilerin geçmiş alışveriş davranışlarından
# elde edilen bilgilerden oluşmaktadır.

# master_id: Eşsiz müşteri numarası
# order_channel : Alışveriş yapılan platforma ait hangi kanalın kullanıldığı (Android, ios, Desktop, Mobile, Offline)
# last_order_channel : En son alışverişin yapıldığı kanal
# first_order_date : Müşterinin yaptığı ilk alışveriş tarihi
# last_order_date : Müşterinin yaptığı son alışveriş tarihi
# last_order_date_online : Muşterinin online platformda yaptığı son alışveriş tarihi
# last_order_date_offline : Muşterinin offline platformda yaptığı son alışveriş tarihi
# order_num_total_ever_online : Müşterinin online platformda yaptığı toplam alışveriş sayısı
# order_num_total_ever_offline : Müşterinin offline'da yaptığı toplam alışveriş sayısı
# customer_value_total_ever_offline : Müşterinin offline alışverişlerinde ödediği toplam ücret
# customer_value_total_ever_online : Müşterinin online alışverişlerinde ödediği toplam ücret
# interested_in_categories_12 : Müşterinin son 12 ayda alışveriş yaptığı kategorilerin listesi


###############################################################
# GÖREVLER
###############################################################

#!pip install lifetimes
import datetime as dt
import pandas as pd
import matplotlib.pyplot as plt
from lifetimes import BetaGeoFitter
from lifetimes import GammaGammaFitter
from lifetimes.plotting import plot_period_transactions

pd.set_option('display.max_columns', None)
pd.set_option('display.width', 500)
pd.set_option('display.float_format', lambda x: '%.4f' % x)
from sklearn.preprocessing import MinMaxScaler

# GÖREV 1: Veriyi Hazırlama
# 1. flo_data_20K.csv verisini okuyunuz.Dataframe’in kopyasını oluşturunuz.

df_ = pd.read_csv("/Users/ardaugurlu/Documents/miuul/crmAnalytics/datasets/flo_data_20k.csv")
df = df_.copy()
# 2. Aykırı değerleri baskılamak için gerekli olan outlier_thresholds ve replace_with_thresholds fonksiyonlarını tanımlayınız.
# Not: cltv hesaplanırken frequency değerleri integer olması gerekmektedir.Bu nedenle alt ve üst limitlerini round() ile yuvarlayınız.


def outlier_thresholds(dataframe, variable): ### kendine girilen değer için eşik değer belirler.
    quartile1 = dataframe[variable].quantile(0.01) # %25 likçeyrek değerler hesaplanır
    quartile3 = dataframe[variable].quantile(0.99) # %75 lik çeyrek değer hesaplanır
    interquantile_range = quartile3 - quartile1 # çeyrek değerlerin farkı hesaplanır
    up_limit = (quartile3 + 1.5 * interquantile_range).round() #üst eşik değer hesaplanır fark 1.5 ile çarpılır ve çeyrek değere eklenir
    low_limit = (quartile1 - 1.5 * interquantile_range).round() #alt eşik değer hesaplanır
    return low_limit, up_limit

def replace_with_thresholds(dataframe, variable): #aykırı değer baskılama fonksiyonu gelecekte kullanılabilir.
    low_limit, up_limit = outlier_thresholds(dataframe, variable)
    # dataframe.loc[(dataframe[variable] < low_limit), variable] = low_limit #negatif değer olursa açılabilir
    dataframe.loc[(dataframe[variable] > up_limit), variable] = up_limit




# 3. "order_num_total_ever_online","order_num_total_ever_offline","customer_value_total_ever_offline","customer_value_total_ever_online" değişkenlerinin
# aykırı değerleri varsa baskılayanız.

replace_with_thresholds(df,"order_num_total_ever_online" )
replace_with_thresholds(df,"order_num_total_ever_offline" )
replace_with_thresholds(df,"customer_value_total_ever_offline")
replace_with_thresholds(df,"customer_value_total_ever_online")





# 4. Omnichannel müşterilerin hem online'dan hemde offline platformlardan alışveriş yaptığını ifade etmektedir. Herbir müşterinin toplam
# alışveriş sayısı ve harcaması için yeni değişkenler oluşturun.

df["total_order"] = df["order_num_total_ever_online"] + df["order_num_total_ever_offline"]
df["total_value"] = df["customer_value_total_ever_offline"] + df["customer_value_total_ever_online"]

df.info()


# 5. Değişken tiplerini inceleyiniz. Tarih ifade eden değişkenlerin tipini date'e çeviriniz.


df["first_order_date"] = pd.to_datetime(df["first_order_date"])
df["last_order_date"] =  pd.to_datetime(df["last_order_date"])
df["last_order_date_online"] =  pd.to_datetime(df["last_order_date_online"])
df["last_order_date_offline"] =  pd.to_datetime(df["last_order_date_offline"])

# GÖREV 2: CLTV Veri Yapısının Oluşturulması
# 1.Veri setindeki en son alışverişin yapıldığı tarihten 2 gün sonrasını analiz tarihi olarak alınız.

df["last_order_date"].max()

analysis_date = dt.datetime(2021,6,2)

df.columns

 # 2.customer_id, recency_cltv_weekly, T_weekly, frequency ve monetary_cltv_avg değerlerinin yer aldığı yeni bir cltv dataframe'i oluşturunuz.
{  # # recency: Son satın alma üzerinden geçen zaman. Haftalık. (kullanıcı özelinde)(recency)
cltv_df = pd.DataFrame()
cltv_df["customer_id"] = df["master_id"]
cltv_df["recency_cltv_weekly"] = ((df["last_order_date"]- df["first_order_date"]).astype('timedelta64[D]')) / 7
cltv_df["T_weekly"] = ((analysis_date - df["first_order_date"]).astype('timedelta64[D]'))/7
cltv_df["frequency"] = df["total_order"]
cltv_df["monetary_cltv_avg"] = df["total_value"] / df["total_order"]





 # Monetary değeri satın alma başına ortalama değer olarak, recency ve tenure değerleri ise haftalık cinsten ifade edilecek.




# GÖREV 3: BG/NBD, Gamma-Gamma Modellerinin Kurulması, CLTV'nin hesaplanması
# 1. BG/NBD modelini fit ediniz.

bgf = BetaGeoFitter(penalizer_coef=0.001) # model kurmamızı sağlıyor verdiğimiz değişkenlere göre 0.0001 ise parametrelere uygulayacağı ceza

bgf.fit(cltv_df['frequency'],
        cltv_df['recency_cltv_weekly'],
        cltv_df['T_weekly'])
 # a. 3 ay içerisinde müşterilerden beklenen satın almaları tahmin ediniz ve exp_sales_3_month olarak cltv dataframe'ine ekleyiniz.

#EXPECTED SALES

cltv_df["expected_sales_3_months"] = bgf.conditional_expected_number_of_purchases_up_to_time(12,# 3 ay için aşağıdaki değişkenlere göre tahmin yap
                                                        cltv_df['frequency'],
                                                        cltv_df['recency_cltv_weekly'],
                                                        cltv_df['T_weekly']).sort_values(ascending=False)



 # b. 6 ay içerisinde müşterilerden beklenen satın almaları tahmin ediniz ve exp_sales_6_month olarak cltv dataframe'ine ekleyiniz.


cltv_df["expected_sales_6_months"] = bgf.conditional_expected_number_of_purchases_up_to_time(24,# 6 ay için aşağıdaki değişkenlere göre tahmin yap
                                                        cltv_df['frequency'],
                                                        cltv_df['recency_cltv_weekly'],
                                                        cltv_df['T_weekly']).sort_values(ascending=False)



# 2. Gamma-Gamma modelini fit ediniz. Müşterilerin ortalama bırakacakları değeri tahminleyip exp_average_value olarak cltv dataframe'ine ekleyiniz.
ggf = GammaGammaFitter(penalizer_coef=0.01)

ggf.fit(cltv_df['frequency'], cltv_df['monetary_cltv_avg'])

cltv_df["exp_average_value"] =ggf.conditional_expected_average_profit(cltv_df['frequency'], #her bir müşteri için beklenen kar#
                                                                       cltv_df['monetary_cltv_avg'])


 # 3. 6 aylık CLTV hesaplayınız ve cltv ismiyle dataframe'e ekleyiniz.

cltv = ggf.customer_lifetime_value(bgf,
                                   cltv_df['frequency'],
                                   cltv_df['recency_cltv_weekly'],
                                   cltv_df['T_weekly'],
                                   cltv_df['monetary_cltv_avg'],
                                   time=6,  # 6 aylık
                                   freq="W",  # T'nin frekans bilgisi.(haftalık mı aylık mı yazabiliriz w haftalık)
                                   discount_rate=0.01)#zaman içerisinde satılan ürünlerde indirim

cltv_df["cltv"] = cltv
 # b. Cltv değeri en yüksek 20 kişiyi gözlemleyiniz.

cltv_df.sort_values("cltv",ascending=False).head(20)




# GÖREV 4: CLTV'ye Göre Segmentlerin Oluşturulması
           # 1. 6 aylık tüm müşterilerinizi 4 gruba (segmente) ayırınız ve grup isimlerini veri setine ekleyiniz. cltv_segment ismi ile dataframe'e ekleyiniz.



cltv_df

cltv_df["segment"] = pd.qcut(cltv_df["cltv"], 4, labels=["D", "C", "B", "A"])

cltv_df.sort_values(by="cltv", ascending=False).head(50) #ilk 50 gözlem

cltv_df.groupby("segment").agg(
    {"count", "mean", "sum"})


# 2. 4 grup içerisinden seçeceğiniz 2 grup için yönetime kısa kısa 6 aylık aksiyon önerilerinde bulununuz

#elimizdeki yeni veri stedine baktığımız zaman cltv değerlerimize A segmentindeki müşterilerimizden beklenen
#kazanç daha fazladır. Bunun için bu müşteriler üzerine odaklanmalıyız. Ortalama kazanç bakımından baktığımızda
#A segmentini açık ara önde görmekteyiz. Bu yüzden a segmentine özel kampanlayalar ile beklentiyi elde etmeliyiz.







