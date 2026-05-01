import pandas as pd
import numpy as np
import pymysql


# STEP 1: Connection
# -----------------------------
conn = pymysql.connect(
    host="localhost",
    user="root",
    password="Akanksha@123",
    database="akankshadb"
)

print("Connected")


# STEP 2: Load data
# -----------------------------
df = pd.read_sql("SELECT * FROM churn_modelling", conn)


print(df.head())


# STEP 3: Drop unnecessary columns
# -----------------------------
df = df.drop(columns=['RowNumber','CustomerId','Surname'])


# STEP 4: Encoding 
# -----------------------------
df['Gender'] = df['Gender'].map({"Female":1, "Male":0})

df['Geography'] = df['Geography'].map({"France":1,"Spain":2,
    "Germany":3
})


# STEP 5: X and Y
# -----------------------------
x = df.drop(columns=['Exited'])
y = df['Exited']


# STEP 6: Train Test Split
# -----------------------------
from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.2, random_state=42
)


# STEP 7: Standardization
# -----------------------------
from sklearn.preprocessing import StandardScaler

sc = StandardScaler()

x_train_sc = sc.fit_transform(x_train)

x_train_new = pd.DataFrame(x_train_sc, columns=x_train.columns)


# STEP 8: Output 
# -----------------------------

print(x_train_new.head())


print(np.round(x_train_new.describe(),2))



