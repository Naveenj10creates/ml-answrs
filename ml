
EXP 2:
import pandas as pd
import scipy
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
df=pd.read_csv("/content/diabetes.csv")
print(df)
df.shape
df.info()
df.describe()
df.describe().T
df.isnull().sum()
df.Insulin.isnull().sum()

#hist 
df.hist(bins=20,figsize=(15,10),color='PURPLE',edgecolor='black')
plt.suptitle('Histogram of Diabetes Dataset')
plt.show()

#boxplot
fig,axs=plt.subplots(9,1,dpi=95,figsize=(7,17))
i=0
for col in df.columns:
  axs[i].boxplot(df[col],vert=False)
  axs[i].set_ylabel(col)
  i+=1
plt.show()

#correlation
corr=df.corr()
sns.heatmap(corr,annot=True,fmt='.2f',cmap='Blues')
plt.show()

corr['Outcome'].sort_values(ascending=False)

#pie chart
plt.pie(df.Outcome.value_counts(),labels=["not Diabetes",'diabetes'],autopct='%.2f%%')
plt.title('Diabetes')
plt.show()

#seperate array
x=df.drop(columns=['Outcome'])
y=df.Outcome
x.head()

from sklearn.preprocessing import MinMaxScaler
scaler=MinMaxScaler(feature_range=(0,1))
rescaledX=scaler.fit_transform(x)
rescaledX[:5]

from sklearn.preprocessing import StandardScaler
scaler=StandardScaler().fit(x)
rescaledX=scaler.fit_transform(x)
rescaledX[:5]


EXP:3
HANDLING MISSING VALUES
import pandas as pd
import numpy as np
data={'first score':[100,90,np.nan,95,75,87],
      'second score':[55,30,45,np.nan,89,98],
      'third score':[np.nan,55,67,87,45,np.nan]
      }
df=pd.DataFrame(data)
df
df1=df.dropna()
df1

df2=df.fillna(0)
df2
df3=df.mean()
df3

df4=df.fillna(df.mean())
df4
df5=df.ffill()
df5
df6=df.bfill()
df6
df7=df.median()
df7
df8=df.fillna(df.median())
df8
from pandas._libs import missing
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
#
numerical_features=df.select_dtypes(include=['int64','float64']).columns
numeric_transformer=Pipeline([
    ('impute',SimpleImputer(missing_values=np.nan,strategy='mean')),
    ('scale',StandardScaler())
])
preprocessor_pipeline=ColumnTransformer(transformers=[('num',numeric_transformer,numerical_features)])
preprocessor_pipeline.fit_transform(df)
df_data=preprocessor_pipeline.fit_transform(df)
df_data

EXP:4
PCA 

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

#sync data
np.random.seed(42)
x=np.random.randint(100,size=(100,3))
X_dataset=pd.DataFrame(x)
print(X_dataset.head())

#step:1 standardize data
scaler=StandardScaler()
X_std=scaler.fit_transform(X_dataset)
print(np.mean(X_std))
print(np.std(X_std))
X_std_dataset=pd.DataFrame(X_std)
print(X_std_dataset.head())

#step 2-5 PCA
pca=PCA(n_components=2)
x_pca=pca.fit_transform(X_std_dataset)
print(x_pca.shape)
print(x_pca)

#plot explained variance  ratio
explained_variance=pca.explained_variance_ratio_
print('Explained variance per principal component:{}'.format(pca.explained_variance_ratio_))
cumulative_var_ratio=np.cumsum(explained_variance)
print('Cumulative explained variance ratio:{}'.format(cumulative_var_ratio))

plt.plot(range(1,len(cumulative_var_ratio)+1),cumulative_var_ratio,marker='o')
plt.xlabel('Number of principal components')
plt.ylabel('Cumulative explained variance ratio')
plt.title('cumulative variance ratio vs number of principal components ')
plt.show()


EXP:5
grid and randomized search

import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split,GridSearchCV
from sklearn.svm import SVC
from sklearn.metrics import classification_report,accuracy_score

#data
iris=datasets.load_iris()
X=iris.data
y=iris.target

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=42)
model=SVC()

hyperparameters={
    'C':[0.1,1,10,100],
    'kernel':['linear','rbf','poly'],
    'gamma':['scale','auto']
}

grid_search=GridSearchCV(estimator=model,param_grid=hyperparameters,scoring='accuracy',cv=5,verbose=2,n_jobs=-1)
grid_search.fit(X_train,y_train)

best_hyperparameters=grid_search.best_params_
best_model=grid_search.best_score_
print("best hyperparameters:",best_hyperparameters)
print("Best Cross_validation Score:",best_model)

best_model=grid_search.best_estimator_
y_pred=best_model.predict(X_test)

print("classification report:")
print(classification_report(y_test,y_pred))
accuracy=accuracy_score(y_test,y_pred)
print("accuracy:",accuracy)

RANDOMIZED

import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split,RandomizedSearchCV
from sklearn.svm import SVC
from sklearn.metrics import classification_report,accuracy_score

#data
iris=datasets.load_iris()
X=iris.data
y=iris.target

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=42)
model=SVC()

hyperparameters={
    'C':[0.1,1,10,100],
    'kernel':['linear','rbf','poly'],
    'gamma':['scale','auto']
}

Random_search=RandomizedSearchCV(estimator=model,param_distributions=hyperparameters,scoring='accuracy',cv=5,verbose=2,n_jobs=-1)
Random_search.fit(X_train,y_train)

best_hyperparameters=Random_search.best_params_
best_model=Random_search.best_score_
print("best hyperparameters:",best_hyperparameters)
print("Best Cross_validation Score:",best_model)

best_model=Random_search.best_estimator_
y_pred=best_model.predict(X_test)

print("classification report:")
print(classification_report(y_test,y_pred))
accuracy=accuracy_score(y_test,y_pred)
print("accuracy:",accuracy)

EXP:6
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

np.random.seed(42)

#defining x and y value
x=np.linspace(0,10,100)
y=np.sin(x)+np.random.normal(0,0.3,size=x.shape)

#reshape x for the model
x_reshaped=x.reshape(-1,1)

#model
model=LinearRegression()
model.fit(x_reshaped,y)

#pred
y_pred=model.predict(x_reshaped)
df=pd.DataFrame(y_pred)
print(df.head(10))

plt.figure(figsize=(8,6))
plt.scatter(x,y,color='blue',label='Data')
plt.plot(x,y_pred,color='red',label='underfitting(linear rigression)')
plt.title('underfitting Ex')
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.show()

OVWEFITTING:
#Overfitting EX
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline

np.random.seed(42)

#defining x and y value
x=np.linspace(0,10,100)
y=np.sin(x)+np.random.normal(0,0.3,size=x.shape)

#reshape x for the model
x_reshaped=x.reshape(-1,1)

#model
model_overfit=make_pipeline(PolynomialFeatures(degree=15),LinearRegression())
model_overfit.fit(x_reshaped,y)

#pred
y_pred_overfit=model_overfit.predict(x_reshaped)

plt.figure(figsize=(8,6))
plt.scatter(x,y,color='blue',label='Data')
plt.plot(x,y_pred_overfit,color='red',label='overfitting(polynomial Degree )')
plt.title('overfitting Ex')
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.show()
