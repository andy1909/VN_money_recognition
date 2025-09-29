import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, MaxPooling2D, Flatten, Dropout
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt

DATASET_PATH = 'Image' 

IMAGE_WIDTH = 64
IMAGE_HEIGHT = 64
IMAGE_SIZE = (IMAGE_WIDTH, IMAGE_HEIGHT)
CHANNELS = 1 

TARGET_CURRENCIES = ['1000', '2000', '10000', '20000', '100000']

def load_data_from_folders(dataset_path, image_size, class_list):
    images = []
    labels = []
    class_map = {name: i for i, name in enumerate(class_list)}

    print("Bắt đầu quá trình tải dữ liệu...")
    print(f"Các lớp sẽ được tải: {class_list}")

    for class_name in class_list:
        class_path = os.path.join(dataset_path, class_name)
        if not os.path.isdir(class_path):
            print(f"Cảnh báo: Thư mục cho lớp '{class_name}' không tồn tại. Bỏ qua.")
            continue
        
        print(f"Đang đọc ảnh từ thư mục: {class_name}")
        for image_name in os.listdir(class_path):
            image_path = os.path.join(class_path, image_name)
            img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE) 
            
            if img is not None:
                img = cv2.resize(img, image_size)
                images.append(img)
                labels.append(class_map[class_name])
            else:
                print(f"Cảnh báo: Không thể đọc ảnh {image_path}. Bỏ qua.")

    return np.array(images), np.array(labels), class_list

images, labels, class_names = load_data_from_folders(DATASET_PATH, IMAGE_SIZE, TARGET_CURRENCIES)

if len(images) == 0:
    print("LỖI: Không có ảnh nào được tải. Vui lòng kiểm tra lại đường dẫn DATASET_PATH.")
else:
    print(f"\nĐã tải thành công {len(images)} ảnh.")
    
    images = images.astype('float32') / 255.0
    images = np.expand_dims(images, axis=-1)

    print(f"Kích thước của tập ảnh sau khi thêm kênh: {images.shape}")
    print(f"Kích thước của tập nhãn: {labels.shape}")


    x_train, x_test, y_train, y_test = train_test_split(
        images, labels, test_size=0.2, random_state=42, stratify=labels
    )

    num_classes = len(class_names)
    y_train = to_categorical(y_train, num_classes)
    y_test = to_categorical(y_test, num_classes)

    print(f"Kích thước x_train: {x_train.shape}")
    print(f"Kích thước y_train: {y_train.shape}")


    model = Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=(IMAGE_HEIGHT, IMAGE_WIDTH, CHANNELS)),
        MaxPooling2D((2, 2)),
        Dropout(0.25),

        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        Dropout(0.25),

        Conv2D(128, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        Dropout(0.25),

        Flatten(),
        Dense(512, activation='relu'),
        Dropout(0.5),
        Dense(num_classes, activation='softmax')
    ])

    model.summary()

    # Biên dịch mô hình
    model.compile(optimizer=Adam(learning_rate=0.001), 
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    # Huấn luyện mô hình
    print("\nBắt đầu huấn luyện mô hình...")
    history = model.fit(x_train, y_train, epochs=50, batch_size=32, validation_data=(x_test, y_test))
    print("Huấn luyện hoàn tất.")

 
    test_loss, test_acc = model.evaluate(x_test, y_test, verbose=2)
    print(f"\nĐộ chính xác trên tập kiểm tra: {test_acc:.4f}")


    model.save('vietnamese_currency_model_cnn.h5') 
    print("Đã lưu mô hình vào file 'vietnamese_currency_model_cnn.h5'")
 
    np.save('vietnamese_currency_class_names.npy', class_names)
    print("Đã lưu danh sách tên các lớp vào file 'vietnamese_currency_class_names.npy'")

    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Training Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title('Training and Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.tight_layout()
    plt.show()