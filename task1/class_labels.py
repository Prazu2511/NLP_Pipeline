import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MultiLabelBinarizer
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping

df = pd.read_csv("./task1/multiclass_calls_dataset_with_rand.csv")


X = df['text_snippet'].values  
y = df['labels'].values  


from sklearn.feature_extraction.text import TfidfVectorizer
vectorizer = TfidfVectorizer(stop_words='english', max_features=5000)
X = vectorizer.fit_transform(X).toarray()  


mlb = MultiLabelBinarizer()
y = mlb.fit_transform([labels.split(", ") for labels in y]) 

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

model = Sequential()


model.add(Dense(512, activation='relu', input_dim=X_train.shape[1]))  
model.add(Dropout(0.5))  


model.add(Dense(512, activation='relu'))
model.add(Dropout(0.5))


model.add(Dense(y_train.shape[1], activation='sigmoid')) 


model.compile(optimizer=Adam(learning_rate=0.0001), loss='binary_crossentropy', metrics=['accuracy'])


early_stop = EarlyStopping(monitor='val_loss', patience=2, restore_best_weights=True)


history = model.fit(X_train, y_train, epochs=100, batch_size=32, validation_data=(X_val, y_val), callbacks=[early_stop])


loss, accuracy = model.evaluate(X_val, y_val)
print(f"Validation Loss: {loss}")
print(f"Validation Accuracy: {accuracy}")


y_pred = model.predict(X_val)
y_pred_labels = mlb.inverse_transform(y_pred > 0.5)  

print(f"Predicted labels for first validation instance: {y_pred_labels[0]}")

import matplotlib.pyplot as plt

plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.legend()
plt.show()

plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.legend()
plt.show()

custom_input = None
custom_input = input()
custom_input_vectorized = vectorizer.transform([custom_input]).toarray() 


y_pred_custom = model.predict(custom_input_vectorized)

predicted_labels = mlb.inverse_transform(y_pred_custom > 0.5) 


with open('./task1/class_labels.txt', 'w') as file:
    file.write(f'{predicted_labels[0]}')



import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score, classification_report, multilabel_confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

y_pred_binary = (y_pred > 0.5).astype(int)


precision = precision_score(y_val, y_pred_binary, average='macro')
recall = recall_score(y_val, y_pred_binary, average='macro')
f1 = f1_score(y_val, y_pred_binary, average='macro')

print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1 Score: {f1:.4f}")

print("\nClassification Report:")
print(classification_report(y_val, y_pred_binary, target_names=mlb.classes_))


conf_matrix = multilabel_confusion_matrix(y_val, y_pred_binary)

class_index = 0  
class_name = mlb.classes_[class_index]

conf_matrix_class = conf_matrix[class_index]

plt.figure(figsize=(6, 5))
sns.heatmap(
    conf_matrix_class,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=["Not " + class_name, class_name],
    yticklabels=["Not " + class_name, class_name]
)
plt.title(f"Confusion Matrix for Class: {class_name}")
plt.xlabel("Predicted")
plt.ylabel("True")
plt.show()



