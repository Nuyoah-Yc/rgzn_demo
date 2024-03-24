import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
import os


# 数据预处理
train_datagen = ImageDataGenerator(rescale=1./255,validation_split=0.2)
train_generator = train_datagen.flow_from_directory(
    'demo\\data\\train',
    target_size=(224, 224),
    batch_size=32,
    # 每个图像都有一个二进制标签（通常是0或1）
    class_mode='binary',
    # 指定加载的是训练数据还是验证数据
    subset='training'
)

validation_generator = train_datagen.flow_from_directory(
    'demo\\data\\train',
    target_size=(224, 224),
    batch_size=32,
    class_mode='binary',
    # 验证数据
    subset='validation'
)

# 构建模型
model = Sequential()
# 卷积层 Conv2D，它是用于图像处理的层
model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(224, 224, 3)))
#池化层 MaxPooling2D 减小图像的空间维度，从而简化网络结构并减少计算量
model.add(MaxPooling2D(pool_size=(2, 2)))
#  Flatten 层将多维的特征图展平为一维，以便可以连接到全连接层（或称为密集层）
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

# 编译模型
# 损失函数为 'binary_crossentropy'（对于二分类问题），优化器为 'adam'，并且要求模型在训练过程中跟踪准确率
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])


# 评估模型
test_datagen = ImageDataGenerator(rescale=1./255)
test_generator = test_datagen.flow_from_directory(
    # 测试图像的目录路径
    'demo\\data\\test',
    target_size=(224, 224),
    # 每个批次应包含的图像数量
    batch_size=32,
    # 每个图像都有一个二进制标签（通常是0或1）
    class_mode='binary',
    # 不要对图像进行随机排序,以相同的顺序评估所有图像
    shuffle=False
)

loss, accuracy = model.evaluate(test_generator, steps=test_generator.samples // test_generator.batch_size)
print(f'Test Loss: {loss}')
print(f'Test Accuracy: {accuracy}')


# 训练模型并保存训练历史
history = model.fit(
    # 训练数据生成器
    train_generator,
    # 每个epoch中应该执行的步骤数。这通常等于训练样本数除以每个batch的大小
    steps_per_epoch=train_generator.samples // train_generator.batch_size,
    # 训练应该进行的轮数
    epochs=20,
    # 验证数据生成器，用于在每个epoch结束时评估模型的性能。
    validation_data=validation_generator,
    #  在验证期间应该执行的步骤数，通常等于验证样本数除以每个batch的大小
    validation_steps=validation_generator.samples // validation_generator.batch_size
)



# 确保目录存在，如果不存在则创建
if not os.path.exists('demo'):
    os.makedirs('demo')

# 现在可以安全地保存模型了
model.save('demo/my_model.keras')
print("训练完成...")
# 保存模型
# model.save('demo\mask_model.h5')


# 绘制损失和准确率图表
plt.figure(figsize=(12, 6))

# 绘制损失曲线
plt.subplot(1, 2, 1)
plt.plot(history.epoch, history.history['loss'], label='Train Loss')
plt.plot(history.epoch, history.history['val_loss'], label='Validation Loss')
plt.title('Loss Progression')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(loc='upper right')

# 绘制准确率曲线
plt.subplot(1, 2, 2)
plt.plot(history.epoch, history.history['accuracy'], label='Train Accuracy')
plt.plot(history.epoch, history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Accuracy Progression')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(loc='lower right')

# 调整子图布局，防止标签重叠
plt.tight_layout()
plt.savefig("demo\\mask.jpg")
# 显示图表
plt.show()
