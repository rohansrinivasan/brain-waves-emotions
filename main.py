import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix,classification_report

# Data Load
data=pd.read_csv('/content/emotions.csv')
data
sample = data.loc[0, 'fft_0_b':'fft_749_b']
plt.figure(figsize= (16,10))
plt.plot(range(len(sample)),sample)
plt.title('Features fft_0_b through fft_749_b')
plt.show()
data['label'].value_counts()
label_mapping = {'NEGATIVE':0, 'NEUTRAL': 1, 'POSITIVE': 2}

# Data Pre-Process
def preprocess_inputs(df):
  df = df.copy()
  df['label'] = df['label'].replace(label_mapping)

  y = df['label'].copy()
  x = df.drop('label', axis=1).copy()

  x_train, x_test, y_train, y_test = train_test_split(x,y,train_size=0.7, random_state=123) # Train Test Split

  return x_train, x_test, y_train, y_test

x_train, x_test, y_train, y_test = preprocess_inputs(data)
x_train

# Model 
inputs = tf.keras.Input(shape=(x_train.shape[1],))

expand_dims = tf.expand_dims (inputs, axis=2)

gru = tf.keras.layers.GRU (256, return_sequences=True) (expand_dims)

flatten = tf.keras.layers.Flatten()(gru)
outputs = tf.keras.layers.Dense(3, activation='softmax') (flatten)

model = tf.keras.Model(inputs=inputs, outputs=outputs) 
print(model.summary())

model.compile(
    optimizer='adam',
    loss= 'sparse_categorical_crossentropy',
    metrics=['accuracy']
)

history = model.fit(
    x_train,
    y_train,
    validation_split=0.2,
    batch_size=32,
    epochs=50,
    callbacks=[
               tf.keras.callbacks.EarlyStopping(
                   monitor='val_loss',
                   patience=5,
                   restore_best_weights=True
               )
    ]
)

# Model Accuracy
model_acc = model.evaluate(x_test, y_test, verbose=0)[1]
print("Test Accuracy: {:.3f}%".format(model_acc * 100))

# Model Performace 
# Train vs Test Accuracy 
import matplotlib.pyplot as plt
%matplotlib inline
# Model accuracy
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model Accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'])
plt.show()

# Train vs Test Loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model Loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'])
plt.show()

# Confusion Matrix y_pred = np.array(list(map(lambda x: np.argmax(x), model.predict(x_test))))
cm = confusion_matrix(y_test, y_pred)
clr = classification_report(y_test, y_pred, target_names=label_mapping.keys())

plt.figure(figsize=(8,8))
sns.heatmap(cm, annot=True, vmin=0, fmt='g', cbar=False, cmap ='Spectral_r')
plt.xticks(np.arange(3) + 0.5, label_mapping.keys())
plt.yticks(np.arange(3) + 0.5, label_mapping.keys())
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.show()

print("Classification Report:\n----\n", clr)