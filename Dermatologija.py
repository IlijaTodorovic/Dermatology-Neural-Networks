from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from keras.models import Sequential
import matplotlib.pyplot as plt
from keras.layers import Dense
import numpy as np
import keras.utils
import sklearn
from sklearn.metrics import confusion_matrix


k = open("derm.txt", "r").read().split("\n")
l = [[int(y) if y.isnumeric() else 0 for y in x.split(",")] for x in k]
prosek = sum([x[-2] for x in l]) / len(l)

for x in l:
  if x[-2] == 0:
    x[-2] = int(prosek)

ulazi = np.array([x[:-1] for x in l] , dtype = np.float32)
izlazi = np.array([x[-1] for x in l] , dtype = np.float32)

plt.hist([ulazi[:,i] for i in range(5)])
plt.xlabel('Vrenost parametra za prvih 5 atributa')
plt.ylabel('Broj pojavljivanja u data setu')
plt.show()


x_train ,x_test ,y_train ,y_test  = sklearn.model_selection.train_test_split(ulazi, izlazi, test_size=0.2, train_size=0.8, random_state=42, shuffle=True, stratify=None)

one_hot = LabelEncoder()
one_hot.fit(y_train)
one_hot_1 = one_hot.transform(y_train)
one_hot_2 = keras.utils.to_categorical(one_hot_1)

test_one_hot = LabelEncoder()
test_one_hot.fit(y_test)
test_one_hot_1 = test_one_hot.transform(y_test)
test_one_hot_2 = keras.utils.to_categorical(test_one_hot_1)

bs = [8,16,32]
lr = [0.01, 0.001, 0.0003]
br_neu = [10,15,30] 

sve = []

for b in bs: 
  for l in lr:
    for n in br_neu:
      
      adam = keras.optimizers.Adam(learning_rate=l)
      model = Sequential() 
      model.add(Dense(5, input_shape = (34,) , activation="relu"))
      model.add(Dense(n, activation="relu"))
      model.add(Dense(6, activation = "softmax"))

      model.compile(loss = "categorical_crossentropy" , optimizer=adam , metrics = ["accuracy"])
      model.fit(x_train , one_hot_2 , epochs = 100 , batch_size = b , verbose=0)
      
      predictions = model.predict(x_test)
      predicted_labels = np.argmax(predictions, axis=1)
      true_labels_one_hot = keras.utils.to_categorical(test_one_hot_1)
      accuracy = np.sum(predicted_labels == np.argmax(true_labels_one_hot, axis=1)) / len(y_test)

      sve.append(((b,l,n) , accuracy))    
      print(f"bs={b} ,lr={l} , br_neu={n} ,accuracy={accuracy}")

print(f"sortirano po acc") 
print(sorted(sve , key = lambda x: x[-1]))

print(f"Accuracy on the test set: {accuracy}")

conf_matrix = confusion_matrix(np.argmax(true_labels_one_hot, axis=1), predicted_labels)
print("Confusion Matrix:")
print(conf_matrix)