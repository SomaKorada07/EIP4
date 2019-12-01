# Final Accuracy for base network

**Best Accuracy:**

Epoch 46/50

390/390 [==============================] - 7s 18ms/step - loss: 0.3310 - acc: 0.8897 - val_loss: 0.5589 - val_acc: 0.8319



**Last Accuracy:**

Epoch 50/50

390/390 [==============================] - 7s 18ms/step - loss: 0.3105 - acc: 0.8967 - val_loss: 0.6007 - val_acc: 0.8245



# New model definition (model.add... ) with output channel size and receptive field

model = Sequential()

model.add(SeparableConv2D(32, 3, 3, border_mode='same', input_shape=(32, 32, 3), depthwise_regularizer=l2(1e-4))) # 32, 3

model.add(BatchNormalization())

model.add(Activation('relu'))

model.add(Dropout(0.1))

model.add(BatchNormalization())

model.add(SeparableConv2D(48, 3, 3, border_mode='same', depthwise_regularizer=l2(1e-4))) # 32, 5

model.add(Activation('relu'))

model.add(Dropout(0.1))

model.add(BatchNormalization())

model.add(SeparableConv2D(64, 3, 3, depthwise_regularizer=l2(1e-4))) # 30, 7

model.add(Activation('relu'))

model.add(Dropout(0.1))

model.add(MaxPooling2D(pool_size=(2, 2))) # 15, 8

model.add(BatchNormalization())

model.add(Convolution2D(32, 1, 1)) #15, 8

model.add(Activation('relu'))

model.add(Dropout(0.1))

model.add(BatchNormalization())

model.add(SeparableConv2D(64, 3, 3, border_mode='same', depthwise_regularizer=l2(1e-4))) # 15, 12

model.add(Activation('relu'))

model.add(Dropout(0.1))

model.add(BatchNormalization())

model.add(SeparableConv2D(128, 3, 3, depthwise_regularizer=l2(1e-4))) # 13, 16

model.add(Activation('relu'))

model.add(Dropout(0.1))

model.add(MaxPooling2D(pool_size=(2, 2))) # 6, 18

model.add(BatchNormalization())

model.add(Convolution2D(64, 1, 1)) # 6, 18

model.add(Activation('relu'))

model.add(Dropout(0.1))

model.add(BatchNormalization())

model.add(SeparableConv2D(64, 3, 3, border_mode='same', depthwise_regularizer=l2(1e-4))) # 6, 26

model.add(Activation('relu'))

model.add(Dropout(0.1))

model.add(BatchNormalization())

model.add(SeparableConv2D(128, 3, 3, border_mode='same', depthwise_regularizer=l2(1e-4))) # 6, 34

model.add(Activation('relu'))

model.add(Convolution2D(10, 1, 1))

model.add(GlobalAveragePooling2D())

model.add(Activation('softmax'))



# 50 epoch logs of new model

```
Epoch 1/50
/usr/local/lib/python3.6/dist-packages/keras_preprocessing/image/image_data_generator.py:716: UserWarning: This ImageDataGenerator specifies `featurewise_center`, but it hasn't been fit on any training data. Fit it first by calling `.fit(numpy_data)`.
  warnings.warn('This ImageDataGenerator specifies '
/usr/local/lib/python3.6/dist-packages/keras_preprocessing/image/image_data_generator.py:724: UserWarning: This ImageDataGenerator specifies `featurewise_std_normalization`, but it hasn't been fit on any training data. Fit it first by calling `.fit(numpy_data)`.
  warnings.warn('This ImageDataGenerator specifies '
Learning rate (from LearningRateScheduler):  0.01
390/390 [==============================] - 29s 76ms/step - loss: 1.4878 - acc: 0.4534 - val_loss: 2.3691 - val_acc: 0.3796
Epoch 2/50
Learning rate (from LearningRateScheduler):  0.01
390/390 [==============================] - 24s 61ms/step - loss: 1.1667 - acc: 0.5822 - val_loss: 1.1906 - val_acc: 0.5951
Epoch 3/50
Learning rate (from LearningRateScheduler):  0.01
390/390 [==============================] - 24s 60ms/step - loss: 1.0319 - acc: 0.6376 - val_loss: 0.9791 - val_acc: 0.6668
Epoch 4/50
Learning rate (from LearningRateScheduler):  0.01
390/390 [==============================] - 24s 60ms/step - loss: 0.9428 - acc: 0.6727 - val_loss: 1.1209 - val_acc: 0.6470
Epoch 5/50
Learning rate (from LearningRateScheduler):  0.01
390/390 [==============================] - 23s 60ms/step - loss: 0.8934 - acc: 0.6927 - val_loss: 0.8650 - val_acc: 0.7009
Epoch 6/50
Learning rate (from LearningRateScheduler):  0.01
390/390 [==============================] - 23s 60ms/step - loss: 0.8466 - acc: 0.7111 - val_loss: 0.9020 - val_acc: 0.7041
Epoch 7/50
Learning rate (from LearningRateScheduler):  0.01
390/390 [==============================] - 23s 60ms/step - loss: 0.8133 - acc: 0.7240 - val_loss: 0.8745 - val_acc: 0.7087
Epoch 8/50
Learning rate (from LearningRateScheduler):  0.01
390/390 [==============================] - 23s 60ms/step - loss: 0.7806 - acc: 0.7375 - val_loss: 0.8365 - val_acc: 0.7213
Epoch 9/50
Learning rate (from LearningRateScheduler):  0.01
390/390 [==============================] - 23s 60ms/step - loss: 0.7654 - acc: 0.7416 - val_loss: 0.7800 - val_acc: 0.7450
Epoch 10/50
Learning rate (from LearningRateScheduler):  0.01
390/390 [==============================] - 23s 59ms/step - loss: 0.7391 - acc: 0.7516 - val_loss: 0.9104 - val_acc: 0.7048
Epoch 11/50
Learning rate (from LearningRateScheduler):  0.01
390/390 [==============================] - 23s 59ms/step - loss: 0.7330 - acc: 0.7530 - val_loss: 0.8126 - val_acc: 0.7385
Epoch 12/50
Learning rate (from LearningRateScheduler):  0.005
390/390 [==============================] - 23s 59ms/step - loss: 0.6467 - acc: 0.7842 - val_loss: 0.6731 - val_acc: 0.7773
Epoch 13/50
Learning rate (from LearningRateScheduler):  0.005
390/390 [==============================] - 23s 60ms/step - loss: 0.6326 - acc: 0.7906 - val_loss: 0.6650 - val_acc: 0.7823
Epoch 14/50
Learning rate (from LearningRateScheduler):  0.005
390/390 [==============================] - 23s 59ms/step - loss: 0.6252 - acc: 0.7940 - val_loss: 0.5997 - val_acc: 0.8040
Epoch 15/50
Learning rate (from LearningRateScheduler):  0.005
390/390 [==============================] - 23s 59ms/step - loss: 0.6179 - acc: 0.7941 - val_loss: 0.6714 - val_acc: 0.7845
Epoch 16/50
Learning rate (from LearningRateScheduler):  0.005
390/390 [==============================] - 23s 59ms/step - loss: 0.6023 - acc: 0.7991 - val_loss: 0.6484 - val_acc: 0.7898
Epoch 17/50
Learning rate (from LearningRateScheduler):  0.005
390/390 [==============================] - 23s 59ms/step - loss: 0.5998 - acc: 0.7998 - val_loss: 0.5886 - val_acc: 0.8075
Epoch 18/50
Learning rate (from LearningRateScheduler):  0.005
390/390 [==============================] - 23s 59ms/step - loss: 0.5914 - acc: 0.8031 - val_loss: 0.7069 - val_acc: 0.7712
Epoch 19/50
Learning rate (from LearningRateScheduler):  0.005
390/390 [==============================] - 23s 58ms/step - loss: 0.5863 - acc: 0.8043 - val_loss: 0.5777 - val_acc: 0.8116
Epoch 20/50
Learning rate (from LearningRateScheduler):  0.005
390/390 [==============================] - 23s 59ms/step - loss: 0.5749 - acc: 0.8090 - val_loss: 0.5746 - val_acc: 0.8158
Epoch 21/50
Learning rate (from LearningRateScheduler):  0.005
390/390 [==============================] - 23s 59ms/step - loss: 0.5705 - acc: 0.8099 - val_loss: 0.5849 - val_acc: 0.8092
Epoch 22/50
Learning rate (from LearningRateScheduler):  0.001
390/390 [==============================] - 23s 59ms/step - loss: 0.5164 - acc: 0.8280 - val_loss: 0.5537 - val_acc: 0.8210
Epoch 23/50
Learning rate (from LearningRateScheduler):  0.001
390/390 [==============================] - 23s 59ms/step - loss: 0.5012 - acc: 0.8322 - val_loss: 0.5328 - val_acc: 0.8262
Epoch 24/50
Learning rate (from LearningRateScheduler):  0.001
390/390 [==============================] - 23s 59ms/step - loss: 0.4966 - acc: 0.8339 - val_loss: 0.5509 - val_acc: 0.8232
Epoch 25/50
Learning rate (from LearningRateScheduler):  0.001
390/390 [==============================] - 23s 59ms/step - loss: 0.4902 - acc: 0.8380 - val_loss: 0.5278 - val_acc: 0.8294
Epoch 26/50
Learning rate (from LearningRateScheduler):  0.001
390/390 [==============================] - 23s 58ms/step - loss: 0.4845 - acc: 0.8380 - val_loss: 0.5600 - val_acc: 0.8200
Epoch 27/50
Learning rate (from LearningRateScheduler):  0.001
390/390 [==============================] - 23s 59ms/step - loss: 0.4865 - acc: 0.8371 - val_loss: 0.5643 - val_acc: 0.8190
Epoch 28/50
Learning rate (from LearningRateScheduler):  0.001
390/390 [==============================] - 23s 59ms/step - loss: 0.4817 - acc: 0.8385 - val_loss: 0.5436 - val_acc: 0.8260
Epoch 29/50
Learning rate (from LearningRateScheduler):  0.001
390/390 [==============================] - 23s 59ms/step - loss: 0.4823 - acc: 0.8393 - val_loss: 0.5793 - val_acc: 0.8142
Epoch 30/50
Learning rate (from LearningRateScheduler):  0.001
390/390 [==============================] - 23s 59ms/step - loss: 0.4799 - acc: 0.8400 - val_loss: 0.5385 - val_acc: 0.8282
Epoch 31/50
Learning rate (from LearningRateScheduler):  0.001
390/390 [==============================] - 23s 59ms/step - loss: 0.4751 - acc: 0.8417 - val_loss: 0.5563 - val_acc: 0.8192
Epoch 32/50
Learning rate (from LearningRateScheduler):  0.0005
390/390 [==============================] - 23s 59ms/step - loss: 0.4663 - acc: 0.8438 - val_loss: 0.5481 - val_acc: 0.8220
Epoch 33/50
Learning rate (from LearningRateScheduler):  0.0005
390/390 [==============================] - 23s 58ms/step - loss: 0.4661 - acc: 0.8441 - val_loss: 0.5310 - val_acc: 0.8292
Epoch 34/50
Learning rate (from LearningRateScheduler):  0.0005
390/390 [==============================] - 23s 59ms/step - loss: 0.4564 - acc: 0.8453 - val_loss: 0.5147 - val_acc: 0.8345
Epoch 35/50
Learning rate (from LearningRateScheduler):  0.0005
390/390 [==============================] - 23s 59ms/step - loss: 0.4608 - acc: 0.8482 - val_loss: 0.5146 - val_acc: 0.8331
Epoch 36/50
Learning rate (from LearningRateScheduler):  0.0005
390/390 [==============================] - 23s 59ms/step - loss: 0.4545 - acc: 0.8488 - val_loss: 0.5306 - val_acc: 0.8294
Epoch 37/50
Learning rate (from LearningRateScheduler):  0.0005
390/390 [==============================] - 23s 60ms/step - loss: 0.4615 - acc: 0.8443 - val_loss: 0.5227 - val_acc: 0.8309
Epoch 38/50
Learning rate (from LearningRateScheduler):  0.0005
390/390 [==============================] - 24s 61ms/step - loss: 0.4528 - acc: 0.8495 - val_loss: 0.5282 - val_acc: 0.8308
Epoch 39/50
Learning rate (from LearningRateScheduler):  0.0005
390/390 [==============================] - 24s 60ms/step - loss: 0.4527 - acc: 0.8492 - val_loss: 0.5077 - val_acc: 0.8339
Epoch 40/50
Learning rate (from LearningRateScheduler):  0.0005
390/390 [==============================] - 24s 61ms/step - loss: 0.4534 - acc: 0.8474 - val_loss: 0.5175 - val_acc: 0.8299
Epoch 41/50
Learning rate (from LearningRateScheduler):  0.0005
390/390 [==============================] - 24s 61ms/step - loss: 0.4497 - acc: 0.8493 - val_loss: 0.5260 - val_acc: 0.8293
Epoch 42/50
Learning rate (from LearningRateScheduler):  0.0001
390/390 [==============================] - 24s 61ms/step - loss: 0.4487 - acc: 0.8506 - val_loss: 0.5147 - val_acc: 0.8333
Epoch 43/50
Learning rate (from LearningRateScheduler):  0.0001
390/390 [==============================] - 24s 61ms/step - loss: 0.4447 - acc: 0.8510 - val_loss: 0.5109 - val_acc: 0.8344
Epoch 44/50
Learning rate (from LearningRateScheduler):  0.0001
390/390 [==============================] - 24s 60ms/step - loss: 0.4415 - acc: 0.8527 - val_loss: 0.5101 - val_acc: 0.8331
Epoch 45/50
Learning rate (from LearningRateScheduler):  0.0001
390/390 [==============================] - 23s 59ms/step - loss: 0.4421 - acc: 0.8516 - val_loss: 0.5172 - val_acc: 0.8314
Epoch 46/50
Learning rate (from LearningRateScheduler):  0.0001
390/390 [==============================] - 23s 59ms/step - loss: 0.4453 - acc: 0.8508 - val_loss: 0.5188 - val_acc: 0.8317
Epoch 47/50
Learning rate (from LearningRateScheduler):  0.0001
390/390 [==============================] - 23s 59ms/step - loss: 0.4404 - acc: 0.8503 - val_loss: 0.5148 - val_acc: 0.8341
Epoch 48/50
Learning rate (from LearningRateScheduler):  0.0001
390/390 [==============================] - 23s 60ms/step - loss: 0.4410 - acc: 0.8519 - val_loss: 0.5172 - val_acc: 0.8321
Epoch 49/50
Learning rate (from LearningRateScheduler):  0.0001
390/390 [==============================] - 23s 58ms/step - loss: 0.4447 - acc: 0.8511 - val_loss: 0.5183 - val_acc: 0.8321
Epoch 50/50
Learning rate (from LearningRateScheduler):  0.0001
390/390 [==============================] - 23s 58ms/step - loss: 0.4453 - acc: 0.8514 - val_loss: 0.5150 - val_acc: 0.8324
Model took 1166.30 seconds to train
```

![img](data:image/png;base64,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%0A9sYAAAAAAElFTkSuQmCC)

```
Accuracy on test data is: 83.24
```

Epoch 34/50

Learning rate (from LearningRateScheduler): 0.0005

390/390 [==============================] - 23s 59ms/step - loss: 0.4564 - acc: 0.8453 - val_loss: 0.5147 - **val_acc: 0.8345**