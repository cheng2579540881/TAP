import tensorflow as tf
from tensorflow.keras import layers, Model
from tensorflow.keras.applications import MobileNetV3Small, ResNet50
import tensorflow.keras.backend as K

def dice_coef(y_true, y_pred, smooth=1):
    intersection = K.sum(y_true * y_pred, axis=[1,2,3])
    union = K.sum(y_true, axis=[1,2,3]) + K.sum(y_pred, axis=[1,2,3])
    return K.mean((2. * intersection + smooth) / (union + smooth), axis=0)

def dice_loss(y_true, y_pred):
    return 1 - dice_coef(y_true, y_pred)

def bce_dice_loss(y_true, y_pred):
    return tf.keras.losses.binary_crossentropy(y_true, y_pred) + dice_loss(y_true, y_pred)

def focal_loss(alpha=0.25, gamma=2.0):
    def loss(y_true, y_pred):
        epsilon = K.epsilon()
        y_pred = K.clip(y_pred, epsilon, 1.0 - epsilon)
        loss_pos = -alpha * K.pow(1 - y_pred, gamma) * K.log(y_pred)
        loss_neg = -(1 - alpha) * K.pow(y_pred, gamma) * K.log(1 - y_pred)
        loss = y_true * loss_pos + (1 - y_true) * loss_neg
        return K.mean(loss)
    return loss

def create_unetpp_classifier(input_shape=(256,256,3), num_classes=5):
    # 使用MobileNetV3Small作为编码器
    base_model = MobileNetV3Small(
        input_shape=input_shape,
        include_top=False,
        weights='imagenet'
    )
    
    # 提取特征层
    skip_connections = [
        base_model.get_layer('input_1').output,  # 256x256
        base_model.get_layer('re_lu_2').output,  # 128x128
        base_model.get_layer('re_lu_7').output,  # 64x64
        base_model.get_layer('re_lu_15').output, # 32x32
    ]
    
    # 编码器输出
    encoder_output = base_model.get_layer('re_lu_29').output  # 16x16
    
    # UNet++ 解码器
    def conv_block(x, filters, name=None):
        x = layers.Conv2D(filters, 3, padding='same', activation='relu', name=name+'_conv1')(x)
        x = layers.BatchNormalization(name=name+'_bn1')(x)
        x = layers.Conv2D(filters, 3, padding='same', activation='relu', name=name+'_conv2')(x)
        x = layers.BatchNormalization(name=name+'_bn2')(x)
        return x
    
    # 第一级
    d0_0 = conv_block(skip_connections[0], 64, 'd0_0')
    d1_0 = conv_block(skip_connections[1], 96, 'd1_0')
    d2_0 = conv_block(skip_connections[2], 128, 'd2_0')
    d3_0 = conv_block(skip_connections[3], 192, 'd3_0')
    
    # 第二级
    u1_0 = layers.UpSampling2D(2, interpolation='bilinear')(d1_0)
    d0_1 = conv_block(layers.Concatenate()([d0_0, u1_0]), 64, 'd0_1')
    
    u2_0 = layers.UpSampling2D(2, interpolation='bilinear')(d2_0)
    d1_1 = conv_block(layers.Concatenate()([d1_0, u2_0]), 96, 'd1_1')
    
    u3_0 = layers.UpSampling2D(2, interpolation='bilinear')(d3_0)
    d2_1 = conv_block(layers.Concatenate()([d2_0, u3_0]), 128, 'd2_1')
    
    # 第三级
    u1_1 = layers.UpSampling2D(2, interpolation='bilinear')(d1_1)
    d0_2 = conv_block(layers.Concatenate()([d0_0, d0_1, u1_1]), 64, 'd0_2')
    
    u2_1 = layers.UpSampling2D(2, interpolation='bilinear')(d2_1)
    d1_2 = conv_block(layers.Concatenate()([d1_0, d1_1, u2_1]), 96, 'd1_2')
    
    # 第四级
    u1_2 = layers.UpSampling2D(2, interpolation='bilinear')(d1_2)
    d0_3 = conv_block(layers.Concatenate()([d0_0, d0_1, d0_2, u1_2]), 64, 'd0_3')
    
    # 分割头
    segmentation_head = layers.Conv2D(1, 1, activation='sigmoid', name='segmentation')(d0_3)
    
    # 分类头
    global_pool = layers.GlobalAveragePooling2D()(encoder_output)
    fc1 = layers.Dense(128, activation='relu')(global_pool)
    fc1 = layers.Dropout(0.5)(fc1)
    
    # 恶性分类
    malignancy_output = layers.Dense(2, activation='softmax', name='malignancy')(fc1)
    
    # TI-RADS分级
    tirads_output = layers.Dense(num_classes, activation='softmax', name='tirads')(fc1)
    
    # 创建模型
    model = Model(
        inputs=base_model.input,
        outputs=[segmentation_head, malignancy_output, tirads_output]
    )
    
    return model

def create_deeplabv3_classifier(input_shape=(256,256,3), num_classes=5):
    # 使用ResNet50作为骨干网络
    base_model = ResNet50(
        input_shape=input_shape,
        include_top=False,
        weights='imagenet'
    )
    
    # 提取特征层
    backbone_output = base_model.get_layer('conv5_block3_out').output  # 16x16
    low_level_features = base_model.get_layer('conv2_block3_out').output  # 64x64
    
    # ASPP模块
    def aspp_block(x, atrous_rates):
        dims = x.shape
        
        # 全局平均池化
        x_pool = layers.AveragePooling2D(pool_size=(dims[1], dims[2]))(x)
        x_pool = layers.Conv2D(256, 1, padding='same', use_bias=False)(x_pool)
        x_pool = layers.BatchNormalization()(x_pool)
        x_pool = layers.Activation('relu')(x_pool)
        x_pool = layers.UpSampling2D(size=(dims[1], dims[2]), interpolation='bilinear')(x_pool)
        
        # 空洞卷积
        x1 = layers.Conv2D(256, 1, dilation_rate=1, padding='same', use_bias=False)(x)
        x1 = layers.BatchNormalization()(x1)
        x1 = layers.Activation('relu')(x1)
        
        x2 = layers.Conv2D(256, 3, dilation_rate=atrous_rates[0], padding='same', use_bias=False)(x)
        x2 = layers.BatchNormalization()(x2)
        x2 = layers.Activation('relu')(x2)
        
        x3 = layers.Conv2D(256, 3, dilation_rate=atrous_rates[1], padding='same', use_bias=False)(x)
        x3 = layers.BatchNormalization()(x3)
        x3 = layers.Activation('relu')(x3)
        
        x4 = layers.Conv2D(256, 3, dilation_rate=atrous_rates[2], padding='same', use_bias=False)(x)
        x4 = layers.BatchNormalization()(x4)
        x4 = layers.Activation('relu')(x4)
        
        # 合并ASPP结果
        x = layers.Concatenate()([x_pool, x1, x2, x3, x4])
        
        # 投影层
        x = layers.Conv2D(256, 1, padding='same', use_bias=False)(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation('relu')(x)
        x = layers.Dropout(0.1)(x)
        
        return x
    
    # 构建ASPP
    x = aspp_block(backbone_output, [6, 12, 18])
    
    # 上采样
    x = layers.UpSampling2D(size=(4, 4), interpolation='bilinear')(x)
    
    # 低级别特征处理
    low_level_features = layers.Conv2D(48, 1, padding='same', use_bias=False)(low_level_features)
    low_level_features = layers.BatchNormalization()(low_level_features)
    low_level_features = layers.Activation('relu')(low_level_features)
    
    # 合并特征
    x = layers.Concatenate()([x, low_level_features])
    
    # 再处理
    x = layers.Conv2D(256, 3, padding='same', use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.Dropout(0.1)(x)
    
    x = layers.Conv2D(256, 3, padding='same', use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.Dropout(0.1)(x)
    
    # 分割头
    segmentation_head = layers.Conv2D(1, 1, activation='sigmoid', name='segmentation')(x)
    
    # 分类头
    global_pool = layers.GlobalAveragePooling2D()(backbone_output)
    fc1 = layers.Dense(256, activation='relu')(global_pool)
    fc1 = layers.Dropout(0.5)(fc1)
    
    # 恶性分类
    malignancy_output = layers.Dense(2, activation='softmax', name='malignancy')(fc1)
    
    # TI-RADS分级
    tirads_output = layers.Dense(num_classes, activation='softmax', name='tirads')(fc1)
    
    # 创建模型
    model = Model(
        inputs=base_model.input,
        outputs=[segmentation_head, malignancy_output, tirads_output]
    )
    
    return model    