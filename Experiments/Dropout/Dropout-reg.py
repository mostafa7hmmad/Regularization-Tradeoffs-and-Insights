

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation
from tensorflow.keras import regularizers

model = Sequential([
    Dense(32, input_shape=(X_train.shape[1],), 
          activation='relu'),
    Dropout(0.3),
    Dense(16, activation='relu'),
    Dropout(0.2),
    Dense(6, activation='relu'),
    Dropout(0.1),
    Dense(3, activation='softmax')  
])


# In[17]:


model.summary()


# In[18]:


model.compile(loss=CategoricalCrossentropy(),optimizer=Adam(0.001),metrics=['accuracy'])


# In[19]:


history = model.fit(
    X_train, y_train,
    validation_data=(X_test, y_test),
    epochs=25,
    batch_size=32,
)





