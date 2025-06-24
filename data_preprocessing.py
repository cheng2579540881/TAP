import os
import cv2
import numpy as np
import pandas as pd
from glob import glob
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.utils import to_categorical
import tensorflow as tf

class DataProcessor:
    def __init__(self, data_dir, clahe_clip=2.0, clahe_tile=(8,8)):
        self.data_dir = data_dir
        self.clahe = cv2.createCLAHE(clipLimit=clahe_clip, tileGridSize=clahe_tile)
        
    def load_data(self, img_size=(256,256)):
        images = []
        masks = []
        labels = []
        
        # 加载图像和标注
        img_paths = glob(os.path.join(self.data_dir, 'images', '*.png'))
        for img_path in img_paths:
            base_name = os.path.basename(img_path)
            mask_path = os.path.join(self.data_dir, 'masks', base_name)
            label_path = os.path.join(self.data_dir, 'labels', base_name.replace('.png', '.csv'))
            
            # 读取图像
            img = cv2.imread(img_path)
            img = cv2.resize(img, img_size)
            img = self._preprocess_image(img)
            images.append(img)
            
            # 读取掩码
            mask = cv2.imread(mask_path, 0)
            mask = cv2.resize(mask, img_size)
            mask = (mask > 127).astype(np.float32)
            masks.append(mask)
            
            # 读取标签
            label_df = pd.read_csv(label_path)
            malignancy = label_df['malignancy'].values[0]
            tirads = label_df['tirads'].values[0]
            labels.append([malignancy, tirads])
        
        return np.array(images), np.array(masks), np.array(labels)
    
    def _preprocess_image(self, img):
        # 去除探头标记（简化处理）
        img = self._remove_probe_marker(img)
        
        # 归一化
        img = img / 255.0
        
        # CLAHE增强
        img_yuv = cv2.cvtColor(img, cv2.COLOR_BGR2YUV)
        img_yuv[:,:,0] = self.clahe.apply(np.uint8(img_yuv[:,:,0]*255))/255.0
        img = cv2.cvtColor(img_yuv, cv2.COLOR_YUV2BGR)
        
        return img
    
    def _remove_probe_marker(self, img):
        # 简化处理，实际应用中需要更复杂的算法
        return img
    
    def create_generators(self, images, masks, labels, batch_size=16, seed=42):
        # 划分训练集和测试集
        X_train, X_test, y_mask_train, y_mask_test, y_label_train, y_label_test = train_test_split(
            images, masks, labels, test_size=0.2, random_state=seed
        )
        
        # 创建数据增强生成器
        data_gen_args = dict(
            rotation_range=20,
            width_shift_range=0.1,
            height_shift_range=0.1,
            shear_range=0.2,
            zoom_range=0.2,
            horizontal_flip=True,
            fill_mode='nearest'
        )
        
        image_datagen = ImageDataGenerator(**data_gen_args)
        mask_datagen = ImageDataGenerator(**data_gen_args)
        
        # 为GAN过采样准备恶性样本
        malignant_indices = np.where(y_label_train[:,0] == 1)[0]
        X_malignant = X_train[malignant_indices]
        y_mask_malignant = y_mask_train[malignant_indices]
        y_label_malignant = y_label_train[malignant_indices]
        
        # 假设我们通过GAN生成了额外的恶性样本
        X_gan, y_mask_gan, y_label_gan = self._generate_gan_samples(
            X_malignant, y_mask_malignant, y_label_malignant, num_samples=500
        )
        
        # 将生成的样本添加到训练集中
        X_train = np.concatenate([X_train, X_gan])
        y_mask_train = np.concatenate([y_mask_train, y_mask_gan])
        y_label_train = np.concatenate([y_label_train, y_label_gan])
        
        # 编译生成器
        image_generator = image_datagen.flow(
            X_train, seed=seed, batch_size=batch_size
        )
        
        mask_generator = mask_datagen.flow(
            y_mask_train.reshape(-1, y_mask_train.shape[1], y_mask_train.shape[2], 1), 
            seed=seed, batch_size=batch_size
        )
        
        # 创建组合生成器
        train_generator = zip(image_generator, mask_generator, 
                             to_categorical(y_label_train[:,0], num_classes=2), 
                             to_categorical(y_label_train[:,1]-1, num_classes=5))
        
        test_data = (X_test, 
                    y_mask_test.reshape(-1, y_mask_test.shape[1], y_mask_test.shape[2], 1), 
                    to_categorical(y_label_test[:,0], num_classes=2), 
                    to_categorical(y_label_test[:,1]-1, num_classes=5))
        
        return train_generator, test_data
    
    def _generate_gan_samples(self, X, y_mask, y_label, num_samples=200):
        # 模拟GAN生成的样本
        indices = np.random.choice(len(X), num_samples)
        X_gan = np.array([self._augment_for_gan(X[i]) for i in indices])
        y_mask_gan = y_mask[indices]
        y_label_gan = y_label[indices]
        return X_gan, y_mask_gan, y_label_gan
    
    def _augment_for_gan(self, img):
        # 模拟GAN增强效果
        augmented = img.copy()
        # 添加随机噪声
        noise = np.random.normal(0, 0.05, img.shape)
        augmented = np.clip(augmented + noise, 0, 1)
        return augmented    