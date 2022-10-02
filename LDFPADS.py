import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import sklearn
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score,precision_score,recall_score,f1_score, confusion_matrix, roc_curve, auc, classification_report
from lightgbm import LGBMClassifier
import catboost
from catboost import CatBoostClassifier
from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE

data = pd.read_csv("Data.csv")
data
data.describe()
data.shape
data.head()
data.info()
data.describe(include = "all")
cat = data.select_dtypes(include='object').columns.to_list() 

for i in cat:
    print("Name of column : ", i)
    print("No. of Unique Values : ", data[i].nunique())
    print("Unique Values : \n", data[i].unique()) 
    print()
    print()
    print('*'*60)
    print()
    print()

def encoding(df):
    
    tag_encoder = LabelEncoder()
    sequence_encoder =  LabelEncoder()
    activity_encoder = LabelEncoder()

    df['Tag Identificator'] = tag_encoder.fit_transform(df['Tag Identificator'])
    df['Sequence Name'] = sequence_encoder.fit_transform(df['Sequence Name'])
    df['Activity'] = activity_encoder.fit_transform(df['Activity'])
    
    return "Successful"

for i in data.select_dtypes(include = ["int64" , 'float64']):
    sns.boxplot(data[i])
    plt.show()

fig = plt.figure()
ax = plt.axes(projection = '3d')
ax.scatter3D(data['X Cordinate'],
            data['Y Cordinate'],
            data['Z Cordinate'],
            c = data['Z Cordinate'], 
            cmap = 'Greens')

def new_col(df):
    df['Data Format'] = pd.to_datetime(df['Data Format'], format = '%d.%m.%Y %H:%M:%S:%f')
    df['Day'] = df['Data Format'].dt.day
    df['Month'] = df['Data Format'].dt.month
    df['Year'] = df['Data Format'].dt.year
    df['Hour'] = df['Data Format'].dt.hour
    df['Minute'] = df['Data Format'].dt.minute
    df['Second'] = df['Data Format'].dt.second
    df['Microsecond'] = df['Data Format'].dt.microsecond
    
    del df['Data Format']
    
    print("Seccuessfull")
    
encoding(data)
new_col(data)

data.head()

plt.figure(figsize = (20 , 12))
sns.heatmap(data.corr(), annot = True)
plt.show()

data.describe(include = 'all')

col = ['Day', 'Month', 'Year', 'Hour', 'Minute', 'Second', 'Microsecond']
for i in col : 
    print('Name of the column : ', i)
    print('No of the unique values : ', data[i].nunique())
    print('Unique Values : ', data[i].unique())
    print()
    print()
    print('*'*60)
    print()
    print()

data.drop(['Day', 'Month', 'Year'], axis = 1, inplace = True)
col = ['Hour', 'Minute', 'Second']

for i in col:
    print("Name of the column : ", i)
    print("No. of Unique Values : ", data[i].nunique())
    print("Unique Values : ", data[i].unique())
    print()
    print()
    print('*'*60)
    print()
    print()

for i in col:
    sns.displot(data[i])
    plt.show()

data.shape
data.info()
data['Activity'].value_counts()

X = data.drop(['Activity'], axis = 1)
y = data['Activity']

smote = SMOTE()
X_tf , y_tf = smote.fit_resample(X , y)
X_tf.shape , y_tf.shape

scaler = RobustScaler()
x = scaler.fit_transform(X_tf)

x_train,x_test,y_train,y_test = train_test_split(x, y_tf, test_size=.1)

print(x_train.shape[0], x_test.shape[0])

accuracy = {}

def train_model(model , model_name):
    print(model_name)
    model = model.fit(x_train, y_train)
    pred = model.predict(x_test)
    acc = accuracy_score(y_test , pred)*100
    accuracy[model_name] = acc
    print('Accuracy Score', acc)
    print()
    print('Classification Report')
    print(classification_report(y_test , pred))

lgbm = LGBMClassifier(n_estimators = 720, n_jobs = -1 , max_depth = 15 , min_child_weight = 5 , min_child_samples = 5 , num_leaves = 10 , learning_rate = 0.15)
train_model(lgbm, 'LGBMClassifier')

cat = CatBoostClassifier(verbose = 0 , n_estimators = 1000)
train_model(cat, 'Cat Boost')
xgb = XGBClassifier(n_estimators = 1500, nthread  = 4, max_depth = 15, min_child_weight = 5, learning_rate=0.01)
train_model(xgb, 'XGBClassifier')

rfc = RandomForestClassifier(n_estimators = 1500, n_jobs=-1, max_depth=15, min_samples_split=5, min_samples_leaf=3)
train_model(rfc, 'Random Forest Classifier')
