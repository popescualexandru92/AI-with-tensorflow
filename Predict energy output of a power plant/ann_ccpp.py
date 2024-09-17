from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
import tensorflow as tf

import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

dataset = pd.read_excel("Folds5x2_pp.xlsx")

"""Variable Name	Role	Type	Description	Units	Missing Values
AT	Feature	Continuous	in the range 1.81°C and 37.11°C	C	no
V	Feature	Continuous	in teh range 25.36-81.56 cm Hg	cm Hg	no
AP	Feature	Continuous	in the range 992.89-1033.30 milibar	milibar	no
RH	Feature	Continuous	in the range 25.56% to 100.16%	%	no
PE	Target	Continuous	420.26-495.76 MW	MW	no"""

x = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values


X_train, X_test, y_train, y_test = train_test_split(
    x, y, test_size=0.2, random_state=0)

ann = tf.keras.models.Sequential()
ann.add(tf.keras.layers.Dense(units=6, activation='relu'))
ann.add(tf.keras.layers.Dense(units=6, activation='relu'))
ann.add(tf.keras.layers.Dense(units=1))

ann.compile(optimizer='adam', loss='mean_squared_error')

ann.fit(X_train, y_train, batch_size=32, epochs=100)

y_pred = ann.predict(X_test)
np.set_printoptions(precision=2)
print(np.concatenate((y_pred.reshape(len(y_pred), 1), y_test.reshape(len(y_test), 1)), 1))

ann.save("ccpp.keras")
