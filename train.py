"""This is a demo for residemt city prediction using ensemble learning.
The training data is generated randomly, and the model is trained using XGBoost, TensorFlow, and Logistic Regression.
The final prediction is made by stacking the predictions from XGBoost and TensorFlow and using Logistic Regression as a meta-learner.
"""
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
import xgboost as xgb
import tensorflow as tf
from tensorflow.keras.layers import Input, Embedding, Flatten, Dense, Concatenate
from tensorflow.keras.models import Model


### Step 1: Prepare Data
# Sample data, 10k users, 700 cities
data = pd.DataFrame({
    'user_id': np.random.randint(0, 10000, 10000),  # 10k unique users
    'avg_purchase_amount': np.random.normal(50, 15, 10000),
    'weekly_activity': np.random.randint(1, 100, 10000),
    'resident_city': np.random.randint(0, 700, 10000),  # 700 cities
    'date': pd.date_range('2020-01-01', periods=10000)
}).sort_values('date').reset_index(drop=True)

# Encode target
label_encoder = LabelEncoder()
data['resident_city'] = label_encoder.fit_transform(data['resident_city'])

# Temporal split data
X_train, X_test, y_train, y_test = train_test_split(
    data[['user_id', 'avg_purchase_amount', 'weekly_activity']],
    data['resident_city'],
    test_size=0.2,
    shuffle=False
)

### Step 2: Train XGboost Models
# XGBoost features (structured data)
X_train_xgb = X_train[['avg_purchase_amount', 'weekly_activity']]
X_test_xgb = X_test[['avg_purchase_amount', 'weekly_activity']]

# Train XGBoost
xgb_model = xgb.XGBClassifier(
    objective='multi:softprob',
    num_class=700,
    n_estimators=300,
    learning_rate=0.1,
    max_depth=6
)
xgb_model.fit(X_train_xgb, y_train)

# Get XGBoost predicted probabilities
xgb_train_probs = xgb_model.predict_proba(X_train_xgb)
xgb_test_probs = xgb_model.predict_proba(X_test_xgb)


### Step 3: Train TensorFlow Embedding Model
# TensorFlow features (user_id as integer)
user_id_train = X_train['user_id'].values
user_id_test = X_test['user_id'].values

# Define embedding model
num_users = 10000  # Unique user IDs
embedding_dim = 16

# Input layer for user_id
user_input = Input(shape=(1,), name='user_input')
user_embed = Embedding(input_dim=num_users, output_dim=embedding_dim)(user_input)
user_flatten = Flatten()(user_embed)

# Combine with other features (optional: add more inputs)
dense = Dense(32, activation='relu')(user_flatten)
output = Dense(700, activation='softmax')(dense)

model = Model(inputs=user_input, outputs=output)
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy')

# Train embedding model
model.fit(
    x=user_id_train,
    y=y_train,
    epochs=10,
    batch_size=128,
    validation_split=0.1
)

# Get embedding model probabilities
tf_train_probs = model.predict(user_id_train)
tf_test_probs = model.predict(user_id_test)


# Concatenated predictions
stacked_train = np.hstack([xgb_train_probs, tf_train_probs])
stacked_test = np.hstack([xgb_test_probs, tf_test_probs])


### Step 4: Train Meta-Learner (Logistic Regression)
# Train meta-learner (Logistic Regression)
lr_model = LogisticRegression(
    multi_class='multinomial',
    solver='lbfgs',
    max_iter=1000
)
lr_model.fit(stacked_train, y_train)

# Evaluate
print("Ensemble Test Accuracy:", lr_model.score(stacked_test, y_test))