

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation
from tensorflow.keras import regularizers

model1 = Sequential([
    Dense(64, input_shape=(X_train.shape[1],),kernel_regularizer=regularizers.l2(0.2)),
    Activation('relu'),

    Dense(32,kernel_regularizer=regularizers.l2(0.002)),
    Activation('relu'),

    Dense(16,kernel_regularizer=regularizers.l2(0.005)),
    Activation('relu'),

    Dense(1, activation='sigmoid')
])


# In[18]:


model1.compile(
    optimizer=tf.keras.optimizers.Adam(0.001),
    loss=tf.keras.losses.BinaryCrossentropy(),
    metrics=['accuracy']
)


# In[19]:


model1.summary()


# In[20]:


history = model1.fit(
    X_train, y_train,
    validation_data=(X_test, y_test),
    epochs=25,
    batch_size=32,
    class_weight=class_weight   
)


# In[21]:




