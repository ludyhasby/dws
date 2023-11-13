import streamlit as st
from PIL import Image
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import altair as alt
# library progress
import time
import re
from datetime import date, timedelta, datetime
import plotly.express as px
import pandas as pd # dataset processing
import numpy as np # array
import seaborn as sns # visualisasi
import matplotlib.pyplot as plt # visualisasi
import plotly.express as px # untuk figures
from sklearn.preprocessing import LabelEncoder # untuk encode label cat
from imblearn.over_sampling import SMOTE # untuk handling imbalance
from sklearn.model_selection import train_test_split # untuk bagi data
from sklearn.metrics import confusion_matrix # buat c matrix
from sklearn.metrics import roc_auc_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
import xgboost as xgb
from xgboost import XGBClassifier
import math # untuk transformasi
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier # untuk melakukan modelling d tree
from sklearn.ensemble import RandomForestClassifier # random forest
import lightgbm as lgb
from lightgbm import LGBMClassifier
# import lazypredict
# from lazypredict.Supervised import LazyClassifier
# setting page nya jadi gede
if "center" not in st.session_state:
    layout = "wide"
else:
    layout = "centered" if st.session_state.center else "wide"
st.set_page_config(layout=layout)

# judul dan sub bagiannya
bag1, bag2 = st.columns([4,1])
with bag1:
    st.markdown("<h1 style='text-align: center; color: black;'>Dashboard Analisis Churn Label</h1>", unsafe_allow_html=True)
    progress_text = "Operation in progress. Please wait."
    my_bar = st.progress(0, text=progress_text)
    for percent_complete in range(100):
        time.sleep(0.1)
        my_bar.progress(percent_complete + 1, text=progress_text)    
    sub1, sub2, sub3 = st.columns(3)
    with sub1 :
        dtime = datetime.now()
        dtime = dtime + timedelta(hours=7)
        current_dateTime = dtime.date()
        st.write(":blue[Tanggal Sekarang adalah:]", current_dateTime, "_waktu setempat_")
    with sub2:
        st.checkbox("Viewing on a mobile?", key="center", value=st.session_state.get("center", False))
    with sub3: 
        st.markdown("<h6 style='text-align: right; '> Tim Demeter </h6>", unsafe_allow_html=True)
with bag2:
    image = Image.open('logo.png')
    st.image(image)

data = pd.read_excel('Telco_customer_churn_adapted_v2.xlsx')

tab1, tab2, tab3 = st.tabs(["Data Visualization", "Data Preprocessing", "Data Modelling"])
with tab1:
    # visualisasi Data Keseluruhan
    st.title("#Visualisasi Data Berdasar Kolom")
    kat = data.iloc[:, [2, 3, 4, 5, 6, 7, 8, 9, 10, 12]]
    for i in kat :
        chart = alt.Chart(data).mark_bar().encode(x=i,y='count():Q').configure_mark(opacity=0.2, color='red')
        st.title(f'Distribusi Data berdasar {i}')
        st.altair_chart(chart, use_container_width=True)

    # agregat produk yang lebih sering dipakai
    st.title("#Produk Yang Paling Sering Digunakan")
    coba_gabung = pd.DataFrame(columns=['Kolom', 'Yes', 'No', 'No Internet'])
    produk = kat.iloc[:, [2, 3, 4, 6]]
    for i in produk.columns:
        coba = produk[i].value_counts()
        coba_gabung = coba_gabung.append({'Kolom': i, 'Yes': coba.get('Yes', 0), 'No': coba.get('No', 0), 'No Internet': coba.get('No internet service', 0)}, ignore_index=True)
    produk_2 = coba_gabung.iloc[:, [0,1]]
    fig, ax = plt.subplots()
    ax.bar(produk_2['Kolom'], produk_2['Yes'], color='grey', width=0.4)
    ax.set_title("Perbandingan Pengguna Beberapa Produk")
    st.pyplot(fig)

    # Visualisasi Data dengan Hue Churn Label
    st.title("#Visualisasi Data dengan Hue Churn Label")
    # distribusi churn pada setiap kategori
    kat = kat.iloc[:, :9]
    for i in kat :
        chart = alt.Chart(data).mark_bar().encode(x=i,y='count():Q', color = "Churn Label")
        st.title(f'Distribusi Churn Label pada {i}')
        st.altair_chart(chart, use_container_width=True)
    
    # Distribusi Churn Label Pada Data Numerik
    st.title("#Distribusi Churn Label Pada Data Numerik")
    bag1, bag2 = st.columns([2,1]) 
    with bag1:
        tenure = px.histogram(data, x="Tenure Months", color="Churn Label", marginal="box")
        st.plotly_chart(tenure)
    with bag2:
        st.write(data.groupby("Churn Label")['Tenure Months'].mean())
        st.write("dapat dilihat ada hubungan yang tampak, pelanggan relatif churn ketika tenure relatif rendah dengan rata-rata melakukan churn pada 18 bulan atau 1,5 tahun")
    bag1, bag2 = st.columns([2,1]) 
    with bag1:
        monthly_purchase = px.histogram(data, x= "Monthly Purchase (Thou. IDR)", color = "Churn Label", marginal = "box")
        st.plotly_chart(monthly_purchase)
    with bag2:
        st.write(data.groupby("Churn Label")['Monthly Purchase (Thou. IDR)'].mean())
        st.write("dapat dilihat, yang melakukan churn relatif melakukan pembelian lebih tinggi dibanding yang tidak. dengan yang melakukan churn rata-rata dalam sebulan membeli 100 ribu sedang yang tidak churn hanya 80 ribu. walau demikian data terlihat -> negatif skew")
    bag1, bag2 = st.columns([2,1])
    with bag1:
        cltv = px.histogram(data, x= "CLTV (Predicted Thou. IDR)", color = "Churn Label", marginal = "box")
        st.plotly_chart(cltv)
    with bag2:
        st.write(data.groupby("Churn Label")['CLTV (Predicted Thou. IDR)'].mean())
        st.write("Dapat dilihat cltv untuk pelanggan yang tidak churn lebih tinggi dibandingkan dengan pelanggan yang melakukan churn")

    # Korelasi pada Data Kategorik dengan One Hot Encoding
    st.set_option('deprecation.showPyplotGlobalUse', False)
    st.title("# Korelasi pada Data Kategorik dengan One Hot Encoding")
    kor = data.copy() # melakukan copying dulu
    kor['Churn Label'].replace(to_replace="Yes", value = 1, inplace=True)
    kor['Churn Label'].replace(to_replace="No", value = 0, inplace=True)
    kor_dummies = pd.get_dummies(kor[['Location', 'Device Class', 'Games Product', 'Music Product',
       'Education Product', 'Call Center', 'Video Product', 'Use MyApp',
       'Payment Method', 'Churn Label']])
    plt.figure(figsize=(7,7))
    sns.heatmap(kor_dummies.corr(), annot=False)
    st.pyplot()

    bag1, bag2 = st.columns([2,1]) 
    with bag1:
        bar_kor = px.bar(kor_dummies.corr()['Churn Label'].sort_values(ascending = False), color = 'value')
        st.plotly_chart(bar_kor)
    with bag2:
        st.write("Conclusion")
        st.write("- games product_no, device class_high end, payment Method_pulsa, music_product_no mempunyai korelasi positif dengan churn")
        st.write("- sementara itu device low class, use myapp_no internet, (vidio, music, games, educ) no internet mempunyai korelasi negatif dengan churn")

with tab2:
    # Data Preprocessing
    st.title("#Drop Data Yang Tidak Perlu")
    code = '''#dropping kolom lat, long, custid, lokasi (karena hampir nggak ada kor)
    data = data.drop(['Customer ID', 'Latitude', 'Longitude', 'Location'], axis = 1)'''
    st.code(code, language='python')
    data = data.drop(['Customer ID', 'Latitude', 'Longitude', 'Location'], axis = 1)
    st.title("#Encode pada Data yang Membutuhkan")
    code = '''# label encoder
    data['Churn Label'].replace(to_replace = 'Yes', value = 1, inplace = True)
    data['Churn Label'].replace(to_replace = 'No', value = 0, inplace = True)
    # ini untuk label encode data kategorik
    def label_encode(dataframe_series):
        if dataframe_series.dtype == 'object':
            dataframe_series = LabelEncoder().fit_transform(dataframe_series)
        return dataframe_series'''
    st.code(code, language='python')
    # label encoder
    data['Churn Label'].replace(to_replace = 'Yes', value = 1, inplace = True)
    data['Churn Label'].replace(to_replace = 'No', value = 0, inplace = True)

    def label_encode(dataframe_series):
        if dataframe_series.dtype == 'object':
            dataframe_series = LabelEncoder().fit_transform(dataframe_series)
        return dataframe_series
        
    data = data.apply(lambda x:label_encode(x))
    kor_lab = px.bar(data.corr()['Churn Label'].sort_values(ascending = False), color = 'value')
    st.plotly_chart(kor_lab)

    st.title("#Data Balancing")
    bag1, bag2 = st.columns([1,1]) 
    with bag1:
        st.table(data.groupby('Churn Label')['Churn Label'].count())
        st.write("akan digunakan SMOTE (Synthetic Minority Oversampling Technique) untuk mengatasi data imbalance")
    with bag2:
        st.write("Teknik SMOTE akan menduplikasi minoritas dengan cara memilih secara random dengan kNN lalu menghitungnya yang akan menjadi sample sintesis")
        over = SMOTE(sampling_strategy = 1)
        x = data.drop("Churn Label", axis = 1).values
        y = data['Churn Label'].values
        x, y = over.fit_resample(x, y)
        unique_values, counts = np.unique(y, return_counts=True)
        st.write(counts)

with tab3:
    st.title("#Pendefinisan Modelling dan Evaluasinya")
    # Split the data into training and testing sets (assuming you already have 'x' and 'y')
    x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=26, test_size=0.2)
    # bangun definisi model dulu
    def model(methodnyaapa, x_train, y_train, x_test, y_test):
        # untuk data training
        methodnyaapa.fit(x_train, y_train)
        # membuat prediksi
        predictions = methodnyaapa.predict(x_test)
        c_matrix = confusion_matrix(y_test, predictions)

        # buat persentase dan label nya
        # Calculate label percentages and create label strings with counts and percentages
        percentages = (c_matrix / np.sum(c_matrix, axis=1)[:, np.newaxis]).round(2) * 100
        labels = [[f"{c_matrix[i, j]} ({percentages[i, j]:.2f}%)" for j in range(c_matrix.shape[1])] for i in range(c_matrix.shape[0])]
        labels = np.asarray(labels)

        # Plot confusion matrix with labeled counts and percentages
        plt.figure(figsize=(4,4))
        sns.heatmap(c_matrix, annot=labels, fmt='', cmap='Blues')
        st.pyplot()

        # Evaluate model performance and print results
        st.write("ROC AUC: ", '{:.2%}'.format(roc_auc_score(y_test, predictions)))
        st.write("Model accuracy: ", '{:.2%}'.format(accuracy_score(y_test, predictions)))
        classification_report_str = classification_report(y_test, predictions)
        st.write(classification_report_str)
    
    code ='''def model(methodnyaapa, x_train, y_train, x_test, y_test):
    # untuk data training
    methodnyaapa.fit(x_train, y_train)
    # membuat prediksi
    predictions = methodnyaapa.predict(x_test)
    c_matrix = confusion_matrix(y_test, predictions)
    # buat persentase dan label nya
    # Calculate label percentages and create label strings with counts and percentages
    percentages = (c_matrix / np.sum(c_matrix, axis=1)[:, np.newaxis]).round(2) * 100
    labels = [[f"{c_matrix[i, j]} ({percentages[i, j]:.2f}%)" for j in range(c_matrix.shape[1])] for i in range(c_matrix.shape[0])]
    labels = np.asarray(labels)
    # Plot confusion matrix with labeled counts and percentages
    sns.heatmap(c_matrix, annot=labels, fmt='', cmap='Blues')
    # Evaluate model performance and print results
    print("ROC AUC: ", '{:.2%}'.format(roc_auc_score(y_test, predictions)))
    print("Model accuracy: ", '{:.2%}'.format(accuracy_score(y_test, predictions)))
    print(classification_report(y_test, predictions))'''
    st.code(code, language='python')

    # clf = LazyClassifier(verbose=0, ignore_warnings=True, custom_metric = None)
    # models,predictions = clf.fit(x_train,  x_test, y_train, y_test)
    # print(models)

    st.title("#XGBClassifier")
    xgb_model = XGBClassifier(learning_rate = 0.01, max_depth = 3, n_estimators = 1000)
    model(xgb_model, x_train, y_train, x_test, y_test)
    st.image('xgb.png')

    st.write("Feature Importannce")
    feature_importance = xgb_model.feature_importances_
    feature_importance_df = pd.DataFrame({'Feature': data.drop("Churn Label", axis = 1).columns, 'Importance': feature_importance})
    feature_importance_df.index = feature_importance_df['Feature']
    feat = px.bar(y=feature_importance_df['Feature'], x=feature_importance_df['Importance'].sort_values(ascending = True))
    st.plotly_chart(feat)
    st.title("#Random Forest")
    rf_model = RandomForestClassifier(random_state=26, n_jobs=-1, max_depth=10,
                                       n_estimators=200, oob_score=True)
    model(rf_model, x_train, y_train, x_test, y_test)
    st.image("rf.png")

    st.title("#LightGBM Classifier")
    lg = LGBMClassifier()
    model(lg, x_train, y_train, x_test, y_test)

    st.title("#LazyClassifier")
    st.write("adalah library di python yang memudahkan untuk mengomparasi beberapa model sekaligus")
    code ='''clf = LazyClassifier(verbose=0, ignore_warnings=True, custom_metric = None)
    models,predictions = clf.fit(x_train_e,  x_test_e, y_train_e,y_test_e)
    print(models)'''
    st.code(code, language='python')
    st.image('lazy.png')
