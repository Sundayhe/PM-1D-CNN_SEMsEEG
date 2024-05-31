import numpy as np
import random
import keras
from keras.optimizers import SGD
from keras.models import Sequential, Model
from keras.layers import Activation, Dense, Dropout, Flatten, Input, Merge, Convolution2D, MaxPooling1D,Conv2D,Conv1D
import scipy.io as sio
from keras.layers.normalization import BatchNormalization
from keras.layers import Dense, Dropout, Flatten, Reshape, GlobalAveragePooling1D
from keras.optimizers import Adam
from keras.regularizers import l1,l2
from keras.callbacks import ReduceLROnPlateau, EarlyStopping, ModelCheckpoint
from keras.models import Model
from sklearn.metrics import confusion_matrix
from sklearn.metrics import confusion_matrix, accuracy_score, recall_score, precision_score
from keras.layers import Multiply
# Generate dummy data


# ---------------------------测试集---------------------------
Testalphafile1=r"D:\5min_lianxu\wx\pos_292_AS.mat"
TestAlpha1=sio.loadmat(Testalphafile1)
TestdataAlpha1=TestAlpha1['combinedData'] #[488,1500]
TestAlpha1 = TestdataAlpha1[:, :, 0]
TestSEM1 = TestdataAlpha1[:, :, 1]

Testalphafile0=r"D:\5min_lianxu\wx\neg_1683_AS.mat"
TestAlpha0=sio.loadmat(Testalphafile0)
TestdataAlpha0=TestAlpha0['combinedData'] #[488,1500]
TestAlpha0 = TestdataAlpha0[:, :, 0]
TestSEM0 = TestdataAlpha0[:, :, 1]


TestAlpha = np.concatenate((TestAlpha1, TestAlpha0), axis=0)
TestAlpha = np.expand_dims(TestAlpha, axis=2)
TestSEM = np.concatenate((TestSEM1, TestSEM0), axis=0)
TestSEM = np.expand_dims(TestSEM, axis=2)
# ----------------------标签-----------------------------------------
y_test = np.concatenate((np.ones(TestAlpha1.shape[0], ), np.zeros(TestAlpha0.shape[0], )), axis=0)  # [12000,]

# ---------------------------训练集------------------------------
TrainAlphafile1=r"D:\8V1\wx\pos_10965.mat"
TrainAlpha1=sio.loadmat(TrainAlphafile1)
x_train1=TrainAlpha1['result'] #(1174, 500)


TrainAlphafile0=r"D:\8V1\wx\neg_112375.mat"
TrainAlpha0=sio.loadmat(TrainAlphafile0)
x_train0=TrainAlpha0['result'] #(1428, 500)

if len(x_train1) < len(x_train0):
    lenx0 = len(x_train0)
    Idseq = np.linspace(1 - 1, lenx0 - 1, lenx0)
    random.shuffle(Idseq)
    Idint = Idseq.astype(int)
    lenx1 = len(x_train1)
    x_train00 = x_train0[Idint[0:lenx1], :]

TrainAlpha1 = x_train1[:, :, 0]
TrainSEM1 = x_train1[:, :, 1]
TrainAlpha0 = x_train00[:, :, 0]
TrainSEM0 = x_train00[:, :, 1]

TrainAlpha = np.concatenate((TrainAlpha1, TrainAlpha0), axis=0)
TrainAlpha = np.expand_dims(TrainAlpha, axis=2)
TrainSEM = np.concatenate((TrainSEM1, TrainSEM0), axis=0)
TrainSEM = np.expand_dims(TrainSEM, axis=2)
# ----------------------标签-----------------------------------------
y_train = np.concatenate((np.ones(TrainAlpha1.shape[0], ), np.zeros(TrainAlpha0.shape[0], )), axis=0)  # [12000,]


#parallel ip for different sections of image
inp1 = Input(shape=TrainAlpha.shape[1:])
inp2 = Input(shape=TrainSEM.shape[1:])



# Alpha波处理
conv1 = Conv1D(150, 50, activation='relu')(inp1)
Nor1 = BatchNormalization()(conv1)
dro1 = Dropout(0.2)(Nor1)
maxp1 = MaxPooling1D(2)(dro1)
conv11 = Conv1D(100, 30, activation='relu')(maxp1)
Nor11 = BatchNormalization()(conv11)
go1 = GlobalAveragePooling1D()(Nor11)

# 眼电信号处理
conv2 = Conv1D(150, 50, activation='relu')(inp2)
Nor2 = BatchNormalization()(conv2)
dro2 = Dropout(0.2)(Nor2)
maxp2 = MaxPooling1D(2)(dro2)
conv22 = Conv1D(100, 30, activation='relu')(maxp2)
Nor22 = BatchNormalization()(conv22)
go2 = GlobalAveragePooling1D()(Nor22)

# 合并并行处理后的结果
mrg = Merge(mode='concat')([go1, go2])

# 增加正则化
dense = Dense(128, activation='relu', kernel_regularizer=l2(0.001))(mrg)
op = Dense(50, activation='relu', kernel_regularizer=l2(0.001))(dense)
out = Dense(2, activation='softmax')(op)


model = Model(input=[inp1, inp2], output=out)

# Print the model summary to see the network architecture
model.summary()

model.compile(
    optimizer=Adam(lr=0.001),  # You can adjust the learning rate
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"],
)

# Set the SGD optimizer with a specific learning rate and momentum
# sgd_optimizer = SGD(lr=0.001, momentum=0.9)
#
# model.compile(
#     optimizer=sgd_optimizer,
#     loss="sparse_categorical_crossentropy",
#     metrics=["accuracy"],
# )


reduce_lr = ReduceLROnPlateau(
    monitor='val_loss',  # The metric to monitor for learning rate reduction
    factor=0.5,          # Reduce learning rate by half when the monitored metric plateaus  当监控指标稳定时将学习率降低一半
    patience=3,          # Number of epochs with no improvement after which learning rate will be reduced  没有改善的时期数，之后学习率将降低
    min_lr=1e-6          # Lower bound on the learning rate   学习率下限
)

# Initialize variables to track consecutive accuracy decreases
consecutive_decreases = 0
best_accuracy = 0.0

# Train the model with validation data and add the learning rate scheduler and early stopping callbacks
for epoch in range(200):  # Set the maximum number of epochs (you can adjust this as needed)
    history = model.fit(
        [TrainAlpha, TrainSEM],
        y_train,
        epochs=1,  # Train for one epoch at a time
        batch_size=150,
        validation_split=0.2
    )

    # Evaluate the model on the validation set
    val_loss = history.history['val_loss'][0]
    val_accuracy = history.history['val_acc'][0]

    # Check if the accuracy has improved
    if val_accuracy > best_accuracy:
        best_accuracy = val_accuracy
        consecutive_decreases = 0
        # Save the model if the accuracy has improved
        model.save_weights('best_model_weights.h5')
    else:
        consecutive_decreases += 1

    # Check if consecutive_decreases is 5, then stop training
    if consecutive_decreases >= 10:
        print(f"Validation accuracy has not improved for 10 consecutive epochs. Stopping training.")
        break

# Load the best weights saved by ModelCheckpoint
model.load_weights('best_model_weights.h5')

# Evaluate the model on the test set
loss, accuracy = model.evaluate([TestAlpha, TestSEM], y_test)
loss = round(loss, 3)
accuracy = round(accuracy, 3)# Evaluate the model on the test set

print("Test Loss:", loss)
print("Test Accuracy:", accuracy)

# 对测试集进行预测
y_pred = model.predict([TestAlpha, TestSEM])
y_pred = np.argmax(y_pred, axis=1)

# 计算混淆矩阵
conf_matrix = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:")
print(conf_matrix)

# 计算准确率
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", round(accuracy, 3))

# 计算召回率
recall = recall_score(y_test, y_pred)
print("Recall:", round(recall, 3))

# 计算精确率
precision = precision_score(y_test, y_pred)
print("Precision:", round(precision, 3))