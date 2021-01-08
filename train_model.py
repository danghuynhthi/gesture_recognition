import os
import warnings
import cv2
import keras
import matplotlib.pyplot as plt
import matplotlib.style as style
import numpy as np
import pandas as pd
from PIL import Image
from keras import models, layers, optimizers
from keras.applications import VGG16
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.layers import Dense, Dropout, Flatten
from keras.models import Model
from keras.preprocessing import image as image_utils
from keras.preprocessing.image import ImageDataGenerator
from keras.utils import to_categorical
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True


# Dinh nghia cac bien

gestures = {'L_': 'L',
           'fi': 'E',
           'ok': 'F',
           'pe': 'V',
           'pa': 'B'
            }

gestures_map = {'E': 0,
                'L': 1,
                'F': 2,
                'V': 3,
                'B': 4
                }


gesture_names = {0: 'E',
                 1: 'L',
                 2: 'F',
                 3: 'V',
                 4: 'B'}


image_path = 'data'
models_path = 'models/saved_model.hdf5'
rgb = False
imageSize = 224


# Ham xu ly anh resize ve 224x224 va chuyen ve numpy array
def process_image(path):
    img = Image.open(path)
    img = img.resize((imageSize, imageSize))
    img = np.array(img)
    return img

# Xu ly du lieu dau vao
#  nếu là ảnh đen trắng sẽ *3 lên để được 3 kênh RGB
def process_data(X_data, y_data):
    X_data = np.array(X_data, dtype = 'float32')
    if rgb:
        pass
    else:
        X_data = np.stack((X_data,)*3, axis=-1)
    X_data /= 255
    y_data = np.array(y_data)#tạo mảng 
    y_data = to_categorical(y_data)#Chuyển đổi một vectơ lớp (số nguyên) thành ma trận lớp nhị phân
    return X_data, y_data

# Ham duuyet thu muc anh dung de train
def walk_file_tree(image_path):
    X_data = []
    y_data = []
    for directory, subdirectories, files in os.walk(image_path):
        for file in files:
            if not file.startswith('.'):
                path = os.path.join(directory, file)
                gesture_name = gestures[file[0:2]]
                print(gesture_name)
                print(gestures_map[gesture_name])
                y_data.append(gestures_map[gesture_name])
                X_data.append(process_image(path))

            else:
                continue

    X_data, y_data = process_data(X_data, y_data)
    return X_data, y_data




# Load du lieu vao X va Y.
X_data, y_data = walk_file_tree(image_path)

# Phan chia du lieu train va test theo ty le 80/20.
X_train, X_test, y_train, y_test = train_test_split(X_data, y_data, test_size = 0.2, random_state=12, stratify=y_data)#80/20
X_train, X_vali, y_train, y_vali = train_test_split(X_train, y_train, test_size = 0.2, random_state=12)#80/20
# Dat cac checkpoint de luu lai model tot nhat.
model_checkpoint = ModelCheckpoint(filepath=models_path, save_best_only=True) # chọn ra các thông số tốt nhất lưu lại 
early_stopping = EarlyStopping(monitor='val_acc',  # ngừng lại khi chỉ số ngừng cải thiện
                               min_delta=0,
                               patience=10,
                               verbose=1,
                               mode='auto',
                               restore_best_weights=True)

# Khoi tao model.
# weights='imagenet' : sử dụng thông số được đào tạo trước trên image net.
# include_top :cho phep tùy chỉnh kích thước dữ liệu đầu vào (False).
# input_shape : tùy chọn kích thước.
model1 = VGG16(weights='imagenet', include_top=False, input_shape=(imageSize, imageSize, 3))
optimizer1 = optimizers.Adam() #tối ưu trọng số
base_model = model1

# Them cac lop ben tren.
x = base_model.output
# đưa về mang 1 chiều.
x = Flatten()(x) 
x = Dense(128, activation='relu', name='fc1')(x)
x = Dense(128, activation='relu', name='fc2')(x)
x = Dense(128, activation='relu', name='fc2a')(x)
x = Dense(128, activation='relu', name='fc3')(x)
x = Dropout(0.5)(x) # fix overfiting
x = Dense(64, activation='relu', name='fc4')(x)
# đây là bài toán multi classifier nên sử dụng activation='softmax'.
#output 5 ngõ ra .
predictions = Dense(5, activation='softmax')(x) 
# nhóm các lớp thành một đối tượng.
model = Model(inputs=base_model.input, outputs=predictions) 

# Dong bang cac lop duoi, chi train lop ben tren minh them vao.
# không train lại Vgg16.
for layer in base_model.layers:
    layer.trainable = False
#Định cấu hình mô hình để đào tạo.
#Adam là thuật toán tối ưu hóa tốc độ học tập thích ứng được thiết kế đặc biệt để đào tạo các mạng deep learning.
#loss Giá trị tổn thất theo kiểu sử dụng entropy chéo vì đây là bài toán nhiều nhãn.
#metrics Danh sách các thước đo được đánh giá bởi mô hình trong quá trình đào tạo và thử nghiệm.
model.compile(optimizer='Adam', loss='categorical_crossentropy', metrics=['accuracy'])
#model.fit(X_train, y_train, epochs=50, batch_size=64, validation_data=(X_test, y_test), verbose=1,
#đào tạo mô hình
 #thay đỏi các thông số #epoch:số lần duyệt qua tất cả phần tử trong tập
 # batch_size là sl phần tử dc duyet mỗi lần
 # validation_data :dữ liêu dánh giá
#  Một Epoch được tính là khi chúng ta đưa tất cả dữ liệu vào mạng neural network 1 lần.

# Khi dữ liệu quá lớn, chúng ta không thể đưa hết mỗi lần tất cả tập dữ liệu vào để huấn luyện được.
#  Buộc lòng chúng ta phải chia nhỏ tập dữ liệu ra thành các batch (size nhỏ hơn).
model.fit(X_train, y_train, epochs=20, batch_size=64, validation_data=(X_vali, y_vali), verbose=1,  
          callbacks=[early_stopping, model_checkpoint])

# Luu model da train ra file
model.save('models/mymodel.h5')


