# CIFAR-10

CIFAR是一个影响力很大的图像分类数据集，分为了CIFAR-10 和CIFAR-100 两个问题，其中的图片是由Alex Krizhevsky, Vinod Nair和Geoffrey Hinton收集的。

CIFAR-10包含了来自10个不同种类，总共60,000张32x32的彩色图片，每一类含有6000张图片。进一步地，其中50,000张为训练图片，剩余的10,000张为测试图片。

下面是数据集中的样例图片：

<img width="500" height="400" src="https://github.com/TIFOSI528/CIFAR-10/raw/master/raw/2017-05-19.png"/>

使用图片生成器ImageDataGenerator对数据进行提升：

	train_datagen = ImageDataGenerator(
    	featurewise_center=True,              # Set input mean to 0 over the dataset, feature-wise.
    	featurewise_std_normalization=True,         # Divide inputs by std of the dataset, feature-wise.
    	rotation_range=20,          # Degree range for random rotations.
    	shear_range=0.2,            # Shear Intensity (Shear angle in counter-clockwise direction as radians)
    	zoom_range=0.2,             # Range for random zoom.
    	fill_mode='nearest',        # Points outside the boundaries of the input are filled according to the given mode.
    	horizontal_flip=True,		# Randomly flip inputs horizontally.
    	)

使用CNN模型，添加BN和正则化，可以实现～91%的准确率。

下面是模型训练过程中accuracy和loss的变化情况：
	
<img width="400" height="300" src="https://github.com/TIFOSI528/CIFAR-10/raw/master/raw/accuracy.png"/>

<img width="400" height="300" src="https://github.com/TIFOSI528/CIFAR-10/raw/master/raw/loss.png"/>

可以看到，模型在训练后期出现了过拟合，还需要进一步改进。


















Reference

* Learning Multiple Layers of Features from Tiny Images, Alex Krizhevsky, 2009.

