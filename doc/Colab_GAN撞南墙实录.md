# Colab_GAN撞南墙实录

作者: Unknown

原文链接: https://zhuanlan.zhihu.com/p/479544866

---

2022年3月11日,在这个并不特殊的日子,我沉痛的缅怀在colab上意外终端的GAN训练模型。

为了防止意外再次发生,我觉得有必要做一系列踩坑手册。

首先,从这浪费的5小时中,我们可以知道:

## Colab 的限制

- 使用了**容器技术**,**实例释放后所有实例中的数据会清空**。这会影响数据集和模型存储的方式。
- 运行 24 小时之后系统会自动回收实例。这会考验代码是否支持无缝衔接(resumption)。

参考了chenglu的[Google Colab 的正确使用姿势](https://zhuanlan.zhihu.com/p/54389036)和Avinash的[How to save our model to Google Drive and reuse it](https://medium.com/@ml_kid/how-to-save-our-model-to-google-drive-and-reuse-it-2c1028058cb2)的建议,我将总结如何避免在Colab上做'无效'训练。

## 第一、梳理工作流

**目的**: 及时记录模型训练过程,保存关键参数,以便之后调用。

**方法**: 挂载google drive,在每个 epoch 将模型检查点保存到 Google Drive,并在下次启动时重新加载。

### 如何挂载google?

需要进行**身份验证**,向 Colab 授予权限,以便它可以访问它并安装驱动器:

```python
from google.colab import drive
drive.mount('/content/drive')
```

## 第二、如何在gdrive中保存模型呢?

要保存我们的模型检查点(或任何文件),我们需要将其保存在驱动器的安装路径中。

例如:

```python
model_save_name = 'classifier.pt'
path = F"/content/gdrive/My Drive/{model_save_name}"
torch.save(model.state_dict(), path)
```

## 第三、如何再次调用我们保存的模型?

记住我们保存的路径:

```python
model_save_name = 'classifier.pt'
path = F"/content/gdrive/My Drive/{model_save_name}"
model.load_state_dict(torch.load(path))
```

## 更多参考

1. [TensorFlow Keras 保存和加载模型](https://colab.research.google.com/github/tensorflow/docs-l10n/blob/master/site/zh-cn/tutorials/keras/save_and_load.ipynb?hl=zh-cn)
2. [How to save our model to Google Drive and reuse it](https://medium.com/@ml_kid/how-to-save-our-model-to-google-drive-and-reuse-it-2c1028058cb2)
3. [Google Colab 的正确使用姿势](https://zhuanlan.zhihu.com/p/54389036)
