<H3>ENTER YOUR NAME: THAANESH.V</H3>
<H3>ENTER YOUR REGISTER NO.: 212223230228</H3>
<H3>EX. NO.1</H3>
<H3>DATE: 21.08.2025</H3>
<H1 ALIGN =CENTER> Introduction to Kaggle and Data preprocessing</H1>

## AIM:
To perform Data preprocessing in a data set downloaded from Kaggle

## EQUIPMENTS REQUIRED:
Hardware – PCs
Anaconda – Python 3.7 Installation / Google Colab /Jupiter Notebook

## RELATED THEORETICAL CONCEPT:

**Kaggle :**
Kaggle, a subsidiary of Google LLC, is an online community of data scientists and machine learning practitioners. Kaggle allows users to find and publish data sets, explore and build models in a web-based data-science environment, work with other data scientists and machine learning engineers, and enter competitions to solve data science challenges.

**Data Preprocessing:**

Pre-processing refers to the transformations applied to our data before feeding it to the algorithm. Data Preprocessing is a technique that is used to convert the raw data into a clean data set. In other words, whenever the data is gathered from different sources it is collected in raw format which is not feasible for the analysis.
Data Preprocessing is the process of making data suitable for use while training a machine learning model. The dataset initially provided for training might not be in a ready-to-use state, for e.g. it might not be formatted properly, or may contain missing or null values.Solving all these problems using various methods is called Data Preprocessing, using a properly processed dataset while training will not only make life easier for you but also increase the efficiency and accuracy of your model.

**Need of Data Preprocessing :**

For achieving better results from the applied model in Machine Learning projects the format of the data has to be in a proper manner. Some specified Machine Learning model needs information in a specified format, for example, Random Forest algorithm does not support null values, therefore to execute random forest algorithm null values have to be managed from the original raw data set.
Another aspect is that the data set should be formatted in such a way that more than one Machine Learning and Deep Learning algorithm are executed in one data set, and best out of them is chosen.


## ALGORITHM:
STEP 1:Importing the libraries<BR>
STEP 2:Importing the dataset<BR>
STEP 3:Taking care of missing data<BR>
STEP 4:Encoding categorical data<BR>
STEP 5:Normalizing the data<BR>
STEP 6:Splitting the data into test and train<BR>

##  PROGRAM:
```python
import pandas as pd                  
from sklearn.preprocessing import MinMaxScaler 
from sklearn.model_selection import train_test_split

df = pd.read_csv("Churn_Modelling.csv")
print(df)

x = df.iloc[:, :-1].values
x

y = df.iloc[:, -1].values
y

print(df.isnull().sum())
df.duplicated()
df.describe()

df = df.drop(['Surname', 'Geography', 'Gender'], axis=1)
scaler = MinMaxScaler()
df1 = pd.DataFrame(scaler.fit_transform(df))
print(df1)

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)

print(x_train)
print(len(x_train))

print(x_test)
print(len(x_test))
```
## OUTPUT:
### Dataset Preview:
<img width="638" height="702" alt="1 dataset preview" src="https://github.com/user-attachments/assets/cf6ff8fa-e6ca-47ba-ad2f-151a7a2e7740" />

### Feature Matrix:
<img width="715" height="163" alt="2 feature matrix" src="https://github.com/user-attachments/assets/51d95e22-5590-48a9-b3e9-73b6d5620c77" />

### Target Vector:
<img width="318" height="50" alt="3 target vector" src="https://github.com/user-attachments/assets/40238658-5fda-4376-a4bb-0d134b762814" />

### Check for missing values:
<img width="236" height="338" alt="4 missing values check" src="https://github.com/user-attachments/assets/87db4ee0-b397-45b6-9cbc-62e3e7a842da" />

### Check for duplicate values:
<img width="362" height="542" alt="5 duplicate check" src="https://github.com/user-attachments/assets/d856bfac-e9a7-45af-bd1e-c3f0baab27c1" />

### Dataset Statistical Summary:
<img width="1556" height="362" alt="6 describe dataset" src="https://github.com/user-attachments/assets/3d544a2d-faf5-44e1-beef-00ee57c79a13" />

### Normalized Dataset:
<img width="773" height="582" alt="7 normalise" src="https://github.com/user-attachments/assets/ffc6b8df-d788-4ea2-ae39-48bcc7de6f08" />

### Training Data:
<img width="488" height="185" alt="8 train data" src="https://github.com/user-attachments/assets/b1e04dce-1a94-445f-9a5c-7a2f630bc3da" />

### Testing Data:
<img width="458" height="185" alt="9 test data" src="https://github.com/user-attachments/assets/6b492d6e-f7bf-4a97-a2e1-057be148ffcf" />

## RESULT:
Thus, Implementation of Data Preprocessing is done in python  using a data set downloaded from Kaggle.


