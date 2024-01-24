
# Customer Credit Risk Assessment: Customer Analysis with Machine Learning Based Classification Model

# Bu veri seti içerisindeki müşterilerimizi ID durumlarına göre iki veri seti tanımlanmıştır.
# İlk veri seti application_record olarak tanımlanmış olup aşağıda ayrıntılı bilgileri bulunmaktadır.

# application_record : Cinsiyet, gelir, yaş, çalışma süresi vb. gibi kullanıcı özellikleriyle ilgili tüm bilgileri içermektedir.

# Detaylar:
# ID  - Client number
# CODE_GENDER Gender / Cinsiyet
# FLAG_OWN_CAR	Is there a car / Arabası var mı?
# FLAG_OWN_REALTY	Is there a property	/ Bir mülkü var mı?
# CNT_CHILDREN	Number of children	/ Çocuk sayısı
# AMT_INCOME_TOTAL	Annual income	/ Yıllık geliri
# NAME_INCOME_TYPE	Income category	/ Gelir tipi
# NAME_EDUCATION_TYPE	Education level	/ Eğitim seviyesi
# NAME_FAMILY_STATUS	Marital status	/ Medeni durumu
# NAME_HOUSING_TYPE	Way of living	/ Ev Tipi
# DAYS_BIRTH	Birthday	Count backwards from current day (0), -1 means yesterday / Doğum günü
# DAYS_EMPLOYED	Start date of employment	Count backwards from current day(0). If positive, it means the person currently unemployed. /
# FLAG_MOBIL	Is there a mobile phone	 / Cep telefonu var mı?
# FLAG_WORK_PHONE	Is there a work phone	/ İş telefonu var mı?
# FLAG_PHONE	Is there a phone	/ Telefonu var mı?
# FLAG_EMAIL	Is there an email	/ Email var mı?
# OCCUPATION_TYPE	Occupation	/ Meslek
# CNT_FAM_MEMBERS	Family size / Aile büyüklüğü

# ---------------------------------------------------------------------------------------------------------------------------
# İkinci veri setimiz ise credit_record olarak tanımlanmış olup aşağıda ayrıntılı bilgileri bulabilirsiniz.

# credit_record : Kullanıcı etkinliği ile ilgili tüm bilgileri içerir,

# MONTHS_BALANCE, çıkarılan verinin ayıdır. Başlangıç noktasıdır, geriye doğru, 0 geçerli aydır, -1 önceki aydır vb.

# STATUS değişkeni şu bilgileri içerir:
#
# 0: 1-29 days past due ( vadesi 1-29 gün geçmiş)
# 1: 30-59 days past due (vadenin 30-59 gün geçmesi)
# 2: 60-89 days overdue(vadenin 60-89 gün geçmesi
# 3: 90-119 days overdue(vadenin 90-119 gün geçmesi)
# 4: 120-149 days overdue(vadenin 120-149 gün geçmesi)
# 5: Overdue or bad debts, write-offs for more than 150 days(Vadesi geçmiş veya şüpheli borçlar, 150 günden uzun süredir silinen alacaklar)
# C: paid off that month(o ay borcumu ödedim)
# X: No loan for the month(ay için kredi yok)

# Bu kapsamda ilgili veri setlerini kullanarak müşterilerimizin krediye uygun olup olmadığına(riskli/risksiz) uygun bir sınıflandırma modeli geliştireceğiz.


#####################################################################
#                  Importing Necessary Libraries                    #
#####################################################################

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

import joblib
import pandas as pd
import numpy as np
from lightgbm import LGBMClassifier
import xgboost as xgb
import mlflow
import mlflow.sklearn

from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier, AdaBoostClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_validate, GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import MinMaxScaler, LabelEncoder, StandardScaler, RobustScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split, cross_val_score,GridSearchCV
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.metrics import confusion_matrix, precision_recall_curve, roc_curve, auc, log_loss
from catboost import CatBoostClassifier
import lightgbm as lgb
from sklearn.model_selection import RandomizedSearchCV


from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.model_selection import cross_val_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import RocCurveDisplay
from sklearn.metrics import accuracy_score
from sklearn.utils import class_weight


from sklearn.metrics import confusion_matrix
import seaborn as sns
from sklearn.metrics import classification_report
from matplotlib import pyplot as plt


#####################################################################
#                      Defining Functions                           #
#####################################################################

# -------------------------------------------- #
#         Exploratory Data Analysis
# -------------------------------------------- #

def check_df(dataframe, head=5):
    print("##################### Shape #####################")
    print(dataframe.shape)
    print("##################### Types #####################")
    print(dataframe.dtypes)
    print("##################### Head #####################")
    print(dataframe.head(head))
    print("##################### Tail #####################")
    print(dataframe.tail(head))
    print("##################### NA #####################")
    print(dataframe.isnull().sum())
    print("##################### Quantiles #####################")
    print(dataframe.quantile([0, 0.05, 0.50, 0.95, 0.99, 1]).T)


def grab_col_names(dataframe, cat_th=10, car_th=20):
    # cat_cols, cat_but_car
    cat_cols = [col for col in dataframe.columns if dataframe[col].dtypes == "O"]
    num_but_cat = [col for col in dataframe.columns if dataframe[col].nunique() < cat_th and
                   dataframe[col].dtypes != "O"]
    cat_but_car = [col for col in dataframe.columns if dataframe[col].nunique() > car_th and
                   dataframe[col].dtypes == "O"]
    cat_cols = cat_cols + num_but_cat
    cat_cols = [col for col in cat_cols if col not in cat_but_car]

    # num_cols
    num_cols = [col for col in dataframe.columns if dataframe[col].dtypes != "O"]
    num_cols = [col for col in num_cols if col not in num_but_cat]

    # print(f"Observations: {dataframe.shape[0]}")
    # print(f"Variables: {dataframe.shape[1]}")
    # print(f'cat_cols: {len(cat_cols)}')
    # print(f'num_cols: {len(num_cols)}')
    # print(f'cat_but_car: {len(cat_but_car)}')
    # print(f'num_but_cat: {len(num_but_cat)}')
    return cat_cols, num_cols, cat_but_car


def target_vs_category_visual(dataframe, target, categorical_col):
    plt.figure(figsize=(15, 8))
    sns.histplot(x=target, hue=categorical_col, data=dataframe, element="step", multiple="dodge")
    plt.title("State of Categorical Variables according to Churn ")
    plt.show()

def cat_summary(dataframe, col_name, plot=False):
    if plot:
        sns.countplot(x=dataframe[col_name], data=dataframe)
        plt.show(block=True)
        print(pd.DataFrame({col_name: dataframe[col_name].value_counts(),
                            "Ratio": 100 * dataframe[col_name].value_counts() / len(dataframe)}))
        print("##########################################")


def target_summary_with_cat(dataframe, target, categorical_col):
    print(pd.DataFrame({"RISK STATUS": dataframe.groupby(categorical_col)[target].mean()}))
    print(20 * "-")


def target_summary_with_num(dataframe, target, numerical_col):
    print(dataframe.groupby(target).agg({numerical_col: "mean"}), end="\n\n")
    print("###################################")


def get_numerical_summary(dataframe):
    total = df.shape[0]
    missing_columns = [col for col in df.columns if df[col].isnull().sum() > 0]
    missing_percent = {}
    for col in missing_columns:
        null_count = df[col].isnull().sum()
        per = (null_count / total) * 100
        missing_percent[col] = per
        print("{} : {} ({}%)".format(col, null_count, round(per, 3)))
    return missing_percent

# -------------------------------------------- #
#              Data Preparation
# -------------------------------------------- #


def outlier_thresholds(dataframe, col_name, q1=0.05, q3=0.95):
    quartile1 = dataframe[col_name].quantile(q1)
    quartile3 = dataframe[col_name].quantile(q3)
    interquantile_range = quartile3 - quartile1
    up_limit = quartile3 + 1.5 * interquantile_range
    low_limit = quartile1 - 1.5 * interquantile_range
    return low_limit, up_limit


def missing_values_table(dataframe, na_name=False):
    na_columns = [col for col in dataframe.columns if dataframe[col].isnull().sum() > 0]

    n_miss = dataframe[na_columns].isnull().sum().sort_values(ascending=False)
    ratio = (dataframe[na_columns].isnull().sum() / dataframe.shape[0] * 100).sort_values(ascending=False)
    missing_df = pd.concat([n_miss, np.round(ratio, 2)], axis=1, keys=['n_miss', 'ratio'])
    print(missing_df, end="\n")

    if na_name:
        return na_columns


def missing_vs_target(dataframe, target, na_columns):
    temp_df = dataframe.copy()
    for col in na_columns:
        temp_df[col + '_NA_FLAG'] = np.where(temp_df[col].isnull(), 1, 0)

    na_flags = temp_df.loc[:, temp_df.columns.str.contains("_NA_")].columns

    for col in na_flags:
        print(pd.DataFrame({"TARGET_MEAN": temp_df.groupby(col)[target].mean(),
                            "Count": temp_df.groupby(col)[target].count()}), end="\n\n\n")


def quick_missing_imp(dataframe, target, num_method="median", cat_length=20):
    variables_with_na = [col for col in dataframe.columns if
                         dataframe[col].isnull().sum() > 0]  # Eksik değere sahip olan değişkenler listelenir

    temp_target = dataframe[target]

    print("# BEFORE")
    print(dataframe[variables_with_na].isnull().sum(),
          "\n\n")  # Uygulama öncesi değişkenlerin eksik değerlerinin sayısı

    # değişken object ve sınıf sayısı cat_lengthe eşit veya altındaysa boş değerleri mode ile doldur
    dataframe = dataframe.apply(
        lambda x: x.fillna(x.mode()[0]) if (x.dtype == "O" and len(x.unique()) <= cat_length) else x, axis=0)

    # num_method mean ise tipi object olmayan değişkenlerin boş değerleri ortalama ile dolduruluyor
    if num_method == "mean":
        dataframe = dataframe.apply(lambda x: x.fillna(x.mean()) if x.dtype != "O" else x, axis=0)
    # num_method median ise tipi object olmayan değişkenlerin boş değerleri ortalama ile dolduruluyor
    elif num_method == "median":
        dataframe = dataframe.apply(lambda x: x.fillna(x.median()) if x.dtype != "O" else x, axis=0)

    dataframe[target] = temp_target

    print("# AFTER \n Imputation method is 'MODE' for categorical variables!")
    print(" Imputation method is '" + num_method.upper() + "' for numeric variables! \n")
    print(dataframe[variables_with_na].isnull().sum(), "\n\n")

    return dataframe


def outlier_th(dataframe, col_name, q1=0.05, q3=0.95):
    quartile1 = dataframe[col_name].quantile(q1)
    quartile3 = dataframe[col_name].quantile(q3)
    interquantile_range = quartile3 - quartile1
    up_limit = quartile3 + 1.5 * interquantile_range
    low_limit = quartile1 - 1.5 * interquantile_range
    return low_limit, up_limit

# Define a Function about checking outlier for data columns
def check_outlier(dataframe, col_name):
    low_limit, up_limit = outlier_th(dataframe, col_name)
    if dataframe[(dataframe[col_name] > up_limit) | (dataframe[col_name] < low_limit)].any(axis=None):
        return True
    else:
        return False

# Define a Function about replace with threshold for data columns
def replace_with_thresholds(dataframe, variable):
    low_limit, up_limit = outlier_th(dataframe, variable)
    dataframe.loc[(dataframe[variable] < low_limit), variable] = low_limit
    dataframe.loc[(dataframe[variable] > up_limit), variable] = up_limit


def missing_values_table(dataframe, na_name=False):
    na_columns = [col for col in dataframe.columns if dataframe[col].isnull().sum() > 0]

    n_miss = dataframe[na_columns].isnull().sum().sort_values(ascending=False)
    ratio = (dataframe[na_columns].isnull().sum() / dataframe.shape[0] * 100).sort_values(ascending=False)
    missing_df = pd.concat([n_miss, np.round(ratio, 2)], axis=1, keys=['n_miss', 'ratio'])
    print(missing_df, end="\n")

    if na_name:
        return na_columns


def one_hot_encoder(dataframe, categorical_cols, drop_first=False):
    dataframe = pd.get_dummies(dataframe, columns=categorical_cols, drop_first=drop_first)
    return dataframe


# -------------------------------------------- #
#         Function for Modeling
# -------------------------------------------- #


def train_test(X, y, test_size=0.20):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)

    print("Base Models....")
    classifiers = [  #('LR', LogisticRegression()),
        # ('KNN', KNeighborsClassifier()),
        # ("SVC", SVC()),
        ("CART", DecisionTreeClassifier(max_depth=4, random_state=0)),
        ("RF", RandomForestClassifier(random_state=0, max_features='sqrt')),
        # ('Adaboost', AdaBoostClassifier(random_state=0)),
        # ('GBM', GradientBoostingClassifier(max_depth=4,random_state=0)),
        ('XGBoost', XGBClassifier(use_label_encoder=False, eval_metric='logloss')),
        ('LightGBM', LGBMClassifier(random_state=0, verbose=-1)),
        #('CatBoost', CatBoostClassifier(verbose=False))
    ]
    print(classifiers)
    return X_train, X_test, y_train, y_test, classifiers

def models(classfiers, X, y):
    for name, classifier in classifiers:
        classifier.fit(X_train, y_train)
        prediction = classifier.predict(X_test)

        cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=1, random_state=1)
        # Calculating Cross-Validation scores for different metrics
        accuracy_cv = cross_val_score(classifier, X_train, y_train, cv=cv, scoring='accuracy',n_jobs=-1).mean()
        f1_cv = cross_val_score(classifier, X_train, y_train, cv=cv, scoring='f1',n_jobs=-1).mean()
        precision_cv = cross_val_score(classifier, X_train, y_train, cv=cv, scoring='precision',n_jobs=-1).mean()
        recall_cv = cross_val_score(classifier, X_train, y_train, cv=cv, scoring='recall',n_jobs=-1).mean()

        # Printing Cross-Validation scores
        print(f"Classifier: {name}")
        print("Cross Validation Scores:")
        print("Accuracy : ", '{0:.2%}'.format(accuracy_cv))
        print("F1 : ", '{0:.2%}'.format(f1_cv))
        print("Precision : ", '{0:.2%}'.format(precision_cv))
        print("Recall : ", '{0:.2%}'.format(recall_cv))

        # Accuracy on test data
        test_accuracy = accuracy_score(y_test, prediction)
        print("Test Accuracy : ", '{0:.2%}'.format(test_accuracy))


def model_evaluation(classifiers, X_test, y_test, X_train, y_train):
    for name, classifier in classifiers:
        classifier.fit(X_train, y_train)
        cm = confusion_matrix(y_test, classifier.predict(X_test))
        names = ['True Neg', 'False Pos', 'False Neg', 'True Pos']
        counts = [value for value in cm.flatten()]
        percentages = ['{0:.2%}'.format(value) for value in cm.flatten() / np.sum(cm)]
        labels = [f'{v1}\n{v2}\n{v3}' for v1, v2, v3 in zip(names, counts, percentages)]
        labels = np.asarray(labels).reshape(2, 2)

        # Her sınıflandırıcı için ayrı bir grafik çiz
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=labels, cmap='Blues', fmt='', square=True)
        plt.title(f'Confusion Matrix for {name}')
        plt.ylabel('True label')
        plt.xlabel('Predicted label')

        # Göster
        plt.show(block=True)

        # Sınıflandırma raporunu yazdır
        print(f'Classification Report for {name}:\n')
        print(classification_report(y_test, classifier.predict(X_test)))


def feature_importances(classifiers, X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)

    for name, classifier in classifiers:
        classifier.fit(X_train, y_train)
        y_pred = classifier.predict(X_test)
        acc_score = accuracy_score(y_test, y_pred)
        feature_imp = pd.Series(classifier.feature_importances_,
                                index=X.columns).sort_values(ascending=False)

        sns.barplot(x=feature_imp, y=feature_imp.index)
        plt.xlabel('Değişken Önem Skorları')
        plt.ylabel('Değişkenler')
        plt.title(name)
        plt.show(block=True)


def hyperparameter_optimization(X, y, classifiers, cv=5, main_scoring='accuracy'):
    print("Hyperparameter Optimization....")
    best_models = {}
    scoring_metrics = ['accuracy', 'f1', 'recall', 'precision']

    for name, classifier, params in classifiers:
        print(f"########## {name} ##########")

        initial_scores = {}
        for metric in scoring_metrics:
            cv_results = cross_validate(classifier, X, y, cv=cv, scoring=metric)
            mean_score = round(cv_results['test_score'].mean(), 4)
            initial_scores[metric] = mean_score
            print(f"{metric} (Before): {mean_score}")

        # GridSearchCV ile hiperparametre optimizasyonu
        # RandomSearchCV
        gs_best = RandomizedSearchCV(classifier, params, cv=cv, scoring=main_scoring, n_jobs=-1, verbose=False).fit(X, y)
        final_model = classifier.set_params(**gs_best.best_params_)
        print(f"{name} best params: {gs_best.best_params_}")

        # Optimizasyon sonrası skorları hesaplama
        optimized_scores = {}
        for metric in scoring_metrics:
            cv_results = cross_validate(final_model, X, y, cv=cv, scoring=metric)
            mean_score = round(cv_results['test_score'].mean(), 4)
            optimized_scores[metric] = mean_score
            print(f"{metric} (After): {mean_score}")

        best_models[name] = {
            'final_model': final_model,
            'initial_scores': initial_scores,
            'optimized_scores': optimized_scores
        }

    return best_models


#####################################################################
#                      Reading Datasets                             #
#####################################################################

credit_data = pd.read_csv("8Week_ML_Part2/halkbank/credit_record.csv")
app_data = pd.read_csv("8Week_ML_Part2/halkbank/application_record.csv")


# - Defining ID Variable as unique

creadit_data_df = pd.DataFrame(credit_data.groupby(["ID"])["MONTHS_BALANCE"].agg(min))
creadit_data_df = creadit_data_df.rename(columns={'MONTHS_BALANCE': 'MONTHS_BEGINNING'})

data = app_data.merge(creadit_data_df , on='ID' , how='inner')


# - Creating Target Variable
credit_data['STATUS'].value_counts()
credit_data = credit_data[credit_data["STATUS"] != "X"]

map_status = {'C' : 0,
              '0' : 0,
              '1' : 1,
              '2' : 1,
              '3' : 1,
              '4' : 1,
              '5' : 1}

credit_data["STATUS"] = credit_data['STATUS'].map(map_status)



credit_target_df = pd.DataFrame(credit_data.groupby(['ID'])['STATUS'].agg(max))
credit_target_df.rename(columns={'STATUS':'TARGET'})


# - Adding "STATUS" Target Variable to Dataframe

df = data.merge(credit_target_df , on='ID' , how='inner')

check_df(df)


#####################################################################
#                      Exploratory Data Analysis                    #
#####################################################################


# - Rename Variables in the dataset

df.columns
df.rename(columns={'CODE_GENDER':'Gender','FLAG_OWN_CAR':'Own_Car','FLAG_OWN_REALTY':'Own_Reality',
                   "DAYS_BIRTH":"Birthday", "DAYS_EMPLOYED":"Employment_Date","FLAG_MOBIL":"Own_Mobile",
                         'CNT_CHILDREN':'ChldNo','AMT_INCOME_TOTAL':'income',
                         'NAME_EDUCATION_TYPE':'educ_tp','NAME_FAMILY_STATUS':'famtp',
                        'NAME_HOUSING_TYPE':'houstp','FLAG_EMAIL':'email',
                         'NAME_INCOME_TYPE':'incometp','FLAG_WORK_PHONE':'wkphone',
                         'FLAG_PHONE':'phone','CNT_FAM_MEMBERS':'famsize',
                        'OCCUPATION_TYPE':'occupty', 'MONTHS_BEGINNING': 'mntbalance', 'STATUS' :'status'
                        },inplace=True)



# - Defining Categorical and Numerical Variables

cat_cols, num_cols, cat_but_car = grab_col_names(df)


# - Summary for Categorical Variables

for col in cat_cols:
    cat_summary(df, col,plot=True)


#  - Target Variable Analysis with Categorical Variable

for col in cat_cols:
    target_summary_with_cat(df,"status",col)



#  - Correlation Analysis

plt.figure(figsize = (20,5))
sns.heatmap(df.corr(),annot = True);
plt.show(block=True)


corr = df.corrwith(df['status']).sort_values(ascending = False).to_frame()
corr.columns = ['Correlations']
plt.subplots(figsize = (5,5))
sns.heatmap(corr,annot = True,linewidths = 0.4,linecolor = 'black')
plt.title('Correlation w.r.t Status')
plt.show(block=True)


#####################################################################
#                      Data Preparation                             #
#####################################################################

df.nunique()
df = df.drop(['ID',"Own_Mobile"],axis=1)

df["status"].value_counts()

# -Observing Missing Data
missing_values_table(df)


# - Filling in missing data by assigning with independent variables

df.loc[(df["occupty"].isnull()) & (df["educ_tp"] == "Secondary / secondary special") &
       (df["incometp"] == "Pensioner"),"occupty" ] = "pensioner_labors"


df.loc[(df["occupty"].isnull()) & (df["educ_tp"] == "Secondary / secondary special") &
       (df["incometp"] == "Working"),"occupty"] = "Laborers"

df.loc[(df["occupty"].isnull()) & (df["educ_tp"] == "Higher education") &
       (df["incometp"] == "Working"),"occupty"] = "High skill tech staff"

df.loc[(df["occupty"].isnull()) & (df["educ_tp"] == "Secondary / secondary special") &
       (df["incometp"] == "Commercial associate"),"occupty"] = "Core staff"

df['occupty'] =df['occupty'].replace(np.nan,'others')

df["occupty"].value_counts()

missing_values_table(df)


# -Performing Outlier Analysis

check_outlier(df,"income")
replace_with_thresholds(df,"income")
check_outlier(df,"income")


#####################################################################
#               Feature Extractions/ Interactions                   #
#####################################################################


# Creating Age variable by Birthday variable
df["Age"] = round((df.Birthday/365)*-1)
df.head()

# Creating new age categorical variable by Age
bins = [21, 34, 45, 57, 69]
labels = ['21-34', '35-45', '46-57', '58-69']
df['Age_category'] = pd.cut(df['Age'], bins=bins, labels=labels, include_lowest=True)


# Creating Experience variable by Employment_Date variable
df["Experience"] = df.Employment_Date/365
df['Experience']=df['Experience'].apply(lambda v : int(v*-1) if v <0 else 0)


# Creating new experience categorical variable from experience by using spec. function
def map_experience_to_category(experience):
    if experience == 0:
        return 'No_experience'
    elif experience == 1:
        return '1 year_exp'
    elif 2 <= experience<= 3:
        return '2-3 years_exp'
    elif 4 <= experience<= 5:
        return '4-5 years_exp'
    elif 6 <= experience<= 10:
        return '6-10 years_exp'
    elif 11 <= experience<= 15:
        return '11-15 years_exp'

    elif 16 <= experience<= 20:
        return '16-20 years_exp'

    elif 21 <= experience<= 30:
        return '21-30 years_exp'

    else:
        return '30+ years_experience'

df['Experience_category'] = df['Experience'].apply(map_experience_to_category)
df["Experience_category"].value_counts()

df.groupby("Experience_category").agg({"status": ["mean","count"]})


# Creating new month_balance categorical variable from month_balance by using spec. function
def map_month_to_category(month):
    if month == 0:
        return 'Current Month'
    elif month == -1:
        return '1 month ago'
    elif month == -2:
        return '2 months ago'
    elif month == -3:
        return '3 months ago'
    elif month == -4:
        return '4 months ago'
    elif -12 <= month <= -5:
        return '5-12 months ago'

    elif -24 <= month < -12:
        return '12-24 months ago'

    elif -36 <= month < 24:
        return '25-36 months ago'

    else:
        return 'More than 36 months ago'

df['monthbalance_category'] = df['mntbalance'].apply(map_month_to_category)
df['monthbalance_category'].value_counts()


# Creating new income categorical variable by Age
df['income_category'] = pd.qcut(df['income'], q=4, labels=['Low', 'Medium', 'High','Very High'])
df['income_category'].value_counts()


# Replace variables
df["Gender"] =  df['Gender'].replace(['F','M'],[0,1])
df["Own_Car"] = df["Own_Car"].replace(["Y","N"],[1,0])
df["Own_Reality"] = df["Own_Reality"].replace(["Y","N"],[1,0])
df["Is_Working"] = df["incometp"].replace(["Working","Commercial associate","State servant","Pensioner","Student"],[1,1,1,0,0])


# Defining famsize variables again
df['famsize'] = df['famsize'].astype(object)
df.loc[df['famsize'] >= 4,'famsize']='4More'
df["famsize"].value_counts()



# Defining feature interaction for occupty and houstp variables
df.loc[(df['occupty']=='Cleaning staff') | (df['occupty']=='Cooking staff') | (df['occupty']=='Drivers') |
       (df['occupty']=='Laborers') | (df['occupty']=='Low-skill Laborers') | (df['occupty']=='Security staff') |
       (df['occupty']=='Waiters/barmen staff') ,'occupty']='Laborwk'


df.loc[(df['occupty']=='Accountants') | (df['occupty']=='Core staff') | (df['occupty']=='HR staff') |
       (df['occupty']=='Medicine staff') | (df['occupty']=='Private service staff') | (df['occupty']=='Realty agents') |
       (df['occupty']=='Sales staff') | (df['occupty']=='Secretaries'),'occupty']='officewk'

df.loc[(df['occupty']=='Managers') | (df['occupty']=='High skill tech staff') | (df['occupty']=='IT staff'),'occupty']='hightecwk'

df["occupty"].value_counts()

df.groupby("occupty").agg({"status": ["mean","count"]})


housing_type = {'House / apartment' : 'House / apartment',
                   'With parents': 'With parents',
                    'Municipal apartment' : 'House / apartment',
                    'Rented apartment': 'House / apartment',
                    'Office apartment': 'House / apartment',
                    'Co-op apartment': 'House / apartment'}

df["houstp"] = df['houstp'].map(housing_type)


# Drop Employment_Date and Birthday variables
df = df.drop(columns=['Employment_Date','Birthday'])
df.head()




#####################################################################
#                      Encoding / Scaling                           #
#####################################################################
df1 = df.copy()

# Performing one hot enconding for categorical variables
cat_cols, num_cols, cat_but_car = grab_col_names(df1)
cat_cols = [col for col in cat_cols if "status" not in col]

df1 = one_hot_encoder(df1, cat_cols, drop_first=True)
df1.head()


# Performing scaling for numerical variables
X_scaled = MinMaxScaler().fit_transform(df1[num_cols])
df1[num_cols] = pd.DataFrame(X_scaled, columns=df1[num_cols].columns)
missing_values_table(df1)



#################################################
# F STATISTICS


from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2,mutual_info_classif
from sklearn.feature_selection import f_classif


def score_func_by_target(variables,score,target):
    features = df.loc[:, variables]
    target = df.loc[:, target]

    best_features = SelectKBest(score_func=score, k='all')
    fit = best_features.fit(features, target)

    featureScores = pd.DataFrame(data=fit.scores_, index=list(features.columns), columns=['Chi Squared Score'])

    plt.subplots(figsize=(15, 5))
    sns.heatmap(featureScores.sort_values(ascending=False, by='Chi Squared Score'), annot=True, linewidths=0.4,
                linecolor='black', fmt='.2f');
    plt.title('Selection of Categorical Features')
    plt.show(block=True)


score_func_by_target(num_cols,f_classif,"status")

cat_cols, num_cols, cat_but_car = grab_col_names(df1)



#  - Correlation Analysis

corr = df1.corrwith(df['status']).sort_values(ascending = False).to_frame()
corr.columns = ['Correlations']
plt.subplots(figsize = (5,5))
sns.heatmap(corr,annot = True,linewidths = 0.4,linecolor = 'black')
plt.title('Correlation w.r.t Status')
plt.show(block=True)

#####################################################################
#                     MODELLING                                     #
#####################################################################

y = df1["status"]
X = df1.drop(["status"], axis=1)

X.head()

# Splitting the model into train/test split
X_train, X_test, y_train, y_test,classifiers = train_test(X,y)

# Model success evaluation with Hold-Out via the complexity matrix
model_evaluation(classifiers,X_test,y_test,X_train,y_train)

# Model success evaluation with K-Fold Cross Validation
models(classifiers,X,y)

# Feature Importance chart of each classifier model
feature_importances(classifiers,X,y)



######################
# Bonus : Dengesiz Veri Setlerine Özel Çalışma
#####################

# GBM Algorithm

weights = class_weight.compute_sample_weight('balanced', y_train)
gbm_model = GradientBoostingClassifier(n_estimators=500,random_state=42)
gbm_model.fit(X_train, y_train, sample_weight=weights)
y_pred = gbm_model.predict(X_test)
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))



# LightGBM

params = {
    'boosting_type': 'gbdt',
    'objective': 'binary',
    'metric': 'binary_logloss',
    'is_unbalance': True,
    'learning_rate': 0.05,
    'num_leaves': 30,
}

z_train = lgb.Dataset(X_train, label=y_train)

lgbm = lgb.train(params, z_train, num_boost_round=1000)

y_pred = lgbm.predict(X_test, num_iteration=lgbm.best_iteration)

y_pred_binary = [1 if x >= 0.5 else 0 for x in y_pred]

# Performans değerlendirme
print(confusion_matrix(y_test, y_pred_binary))
print(classification_report(y_test, y_pred_binary))



##########################################################################
#                   BONUS : MLFLOW
##########################################################################

# MLflow deneyini başlat
mlflow.set_experiment("mlflow_model_performance")



def train_and_log_model(name, model, X_train, X_test, y_train, y_test):
    with mlflow.start_run(run_name=name):
        # Modeli eğit
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        # Performans metriklerini hesapla
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)

        # Metrikleri logla
        mlflow.log_metric("accuracy", accuracy)
        mlflow.log_metric("precision", precision)
        mlflow.log_metric("recall", recall)
        mlflow.log_metric("f1", f1)

        # Modeli logla
        mlflow.sklearn.log_model(model, name)

        return model


X_train, X_test, y_train, y_test, classifiers = train_test(X, y)

for name, model in classifiers:
    train_and_log_model(name, model, X_train, X_test, y_train, y_test)



# MLflow deneyini başlat
mlflow.set_experiment("mlflow_confusion_evalution")


def log_confusion_matrix(cm, labels, name):
    # Confusion Matrix'i kaydetmek için özel bir fonksiyon
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(cm, annot=labels, cmap='Blues', fmt='', square=True)
    plt.title(f'Confusion Matrix for {name}')
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()  # Düzeni ayarla

    # MLflow'a resmi kaydet
    temp_file = f"{name}_confusion_matrix.png"
    plt.savefig(temp_file)
    mlflow.log_artifact(temp_file)

def model_evaluation_with_mlflow(classifiers, X_test, y_test, X_train, y_train):
    for name, classifier in classifiers:
        with mlflow.start_run(run_name=f"{name}_evaluation"):
            # Modeli eğit
            classifier.fit(X_train, y_train)
            y_pred = classifier.predict(X_test)

            # Confusion Matrix hesapla ve logla
            cm = confusion_matrix(y_test, y_pred)
            names = ['True Neg', 'False Pos', 'False Neg', 'True Pos']
            counts = [value for value in cm.flatten()]
            percentages = ['{0:.2%}'.format(value) for value in cm.flatten() / np.sum(cm)]
            labels = np.asarray([f'{v1}\n{v2}\n{v3}' for v1, v2, v3 in zip(names, counts, percentages)]).reshape(2, 2)
            log_confusion_matrix(cm, labels, name)

            # Sınıflandırma raporunu kaydet
            report = classification_report(y_test, y_pred, output_dict=True)
            mlflow.log_metrics({f"{name}_precision": report['weighted avg']['precision'],
                                f"{name}_recall": report['weighted avg']['recall'],
                                f"{name}_f1-score": report['weighted avg']['f1-score']})

            plt.show(block=True)



model_evaluation_with_mlflow(classifiers,X_test,y_test,X_train,y_train)





#####################################################################
#             UNDERSAMPLING/OVERSAMPLING                            #
#####################################################################

from imblearn.under_sampling import RandomUnderSampler
from collections import Counter
undersample = RandomUnderSampler(random_state=0)
X_undersample, y_undersample = undersample.fit_resample(X, y)

counter = Counter(y_undersample)


# Splitting the model into train/test split
X_train, X_test, y_train, y_test,classifiers = train_test(X_undersample,y_undersample)


# Model success evaluation with Hold-Out via the complexity matrix
model_evaluation(classifiers,X_test,y_test,X_train,y_train)

# Model success evaluation with K-Fold Cross Validation
models(classifiers,X_undersample,y_undersample)

# Feature Importance chart of each classifier model
feature_importances(classifiers,X_undersample,y_undersample)


# MLFLOW UNDERSAMPLING

mlflow.set_experiment("undersampling")

def train_and_log_model(name, model, X_train, X_test, y_train, y_test):
    with mlflow.start_run(run_name=name):
        # Modeli eğit
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        # Performans metriklerini hesapla
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)

        # Metrikleri logla
        mlflow.log_metric("accuracy", accuracy)
        mlflow.log_metric("precision", precision)
        mlflow.log_metric("recall", recall)
        mlflow.log_metric("f1", f1)

        # Modeli logla
        mlflow.sklearn.log_model(model, name)

        return model


for name, model in classifiers:
    train_and_log_model(name, model, X_train, X_test, y_train, y_test)

###############################################################3
# TOMEK LINKS Yöntemi Uygulayarak Undersampling

# Birbirine en yakın çoğunluk ve azınlık sınıfı örnek çiftlerini (Tomek Links) bulur ve bu çiftlerden çoğunluk sınıfına ait olanları siler.
# Bu, sınırların daha net olmasını sağlayarak azınlık sınıfının sınıflandırılmasını kolaylaştırabilir.

from imblearn.under_sampling import TomekLinks

# Tomek Links uygulayalım
tl = TomekLinks()
X_resampled, y_resampled = tl.fit_resample(X_train, y_train)



# Splitting the model into train/test split
X_train, X_test, y_train, y_test,classifiers = train_test(X_resampled,y_resampled)


# Model success evaluation with Hold-Out via the complexity matrix
model_evaluation(classifiers,X_test,y_test,X_train,y_train)

# Model success evaluation with K-Fold Cross Validation
models(classifiers,X_undersample,y_undersample)

# Feature Importance chart of each classifier model
feature_importances(classifiers,X_undersample,y_undersample)



#####################################################################
#                 HYPERPARAMETER OPTIMIZATION                       #
#####################################################################

# - Creating default parameter values of the ML Models

cart_params = {'max_depth': range(1, 20),
               "min_samples_split": range(2, 30) }

rf_params = {"max_depth": [8, 15, None],
             "max_features": [5, 7, "sqrt"],
             "min_samples_split": [15, 20],
             "n_estimators": [200, 300,500]}

xgboost_params = {"learning_rate": [0.1, 0.01],
                  "max_depth": [5, 8],
                  "n_estimators": [100,200,500],
                  "colsample_bytree": [0.5, 1],}

lightgbm_params = {"learning_rate": [0.01, 0.1],
                   "n_estimators": [300, 500,1000],
                   "colsample_bytree": [0.7, 1]}

catboost_params = {
    'learning_rate': [0.01, 0.05, 0.1, 0.3],
    'depth': [4, 6, 8, 10],
    'iterations': [100, 250, 500, 1000],
    'l2_leaf_reg': [1, 3, 5, 7, 9],
    'border_count': [32, 64, 128, 255],
    'bagging_temperature': [0.0, 0.5, 1.0],
    'auto_class_weights': ['None', 'Balanced', 'SqrtBalanced']
}


classifiers = [#('KNN', KNeighborsClassifier(), knn_params),
               ("CART", DecisionTreeClassifier(random_state=42), cart_params),
               ("RF", RandomForestClassifier(random_state=42), rf_params),
               ('XGBoost', xgb.XGBClassifier(eval_metric='logloss',random_state=42), xgboost_params),
               ('LightGBM', LGBMClassifier(random_state=42,verbose=-1), lightgbm_params),]
                #('CatBoost', CatBoostClassifier(verbose=False),catboost_params)]


best_models = hyperparameter_optimization(X_undersample,y_undersample,classifiers)



#########################################################################


def hyperparameter_optimization_with_mlflow(X, y, classifiers, cv=5, main_scoring='accuracy'):
    best_models = {}
    for name, classifier, params in classifiers:
        with mlflow.start_run(run_name=f"{name}_hyperparameter_optimization"):
            # Hiperparametre optimizasyonu
            gs_best = GridSearchCV(classifier, params, cv=cv, scoring=main_scoring, n_jobs=-1, verbose=False).fit(X, y)

            # En iyi parametreleri logla
            mlflow.log_params(gs_best.best_params_)

            # Final modeli oluştur
            final_model = classifier.set_params(**gs_best.best_params_)

            # Cross-validation ile modeli değerlendir
            cv_results = cross_validate(final_model, X, y, cv=cv, scoring=['accuracy', 'f1', 'recall', 'precision'])
            for metric in cv_results:
                if 'test_' in metric:
                    avg_score = np.mean(cv_results[metric])
                    mlflow.log_metric(metric, avg_score)

            best_models[name] = final_model

    return best_models


hyperparameter_optimization_with_mlflow(X,y,classifiers,cv=5,main_scoring="accuracy")

#####################################################################
#                       OPTIMIZATION RESULT                         #
#####################################################################


model_results_df = pd.DataFrame(columns=['Model', 'Metric', 'Before Optimization', 'After Optimization'])

for name, model_info in best_models.items():
    for metric in model_info['initial_scores']:
        before_score = model_info['initial_scores'][metric]
        after_score = model_info['optimized_scores'][metric]
        model_results_df = model_results_df.append({
            'Model': name,
            'Metric': metric,
            'Before Optimization': before_score,
            'After Optimization': after_score
        }, ignore_index=True)

print(model_results_df)



