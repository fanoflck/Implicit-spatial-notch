from scipy.ndimage.filters import gaussian_filter
import numpy as np
from PIL import Image
import torch
import os
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
import torchvision
from adversarial_noise import adv_guided_noise


def gaussian_blur(img, sigma):
    gaussian_filter(img[:, :, 0], output=img[:, :, 0], sigma=sigma)
    gaussian_filter(img[:, :, 1], output=img[:, :, 1], sigma=sigma)
    gaussian_filter(img[:, :, 2], output=img[:, :, 2], sigma=sigma)


def gaussian_noise(img, mean, sigma, adv_guided_map=None):
    img = np.asarray(img, dtype=np.float32)
    noise = np.random.normal(mean, sigma, img.shape)

    if not adv_guided_map is None:
        noise = noise * adv_guided_map

    out = img + noise
    out = np.clip(out, 0, 255)
    out = np.uint8(out)
    return out


def uniform_noise(img, low, high, adv_guided_map=None):
    img = np.asarray(img, dtype=np.float32)
    noise = np.random.uniform(low=low, high=high, size=img.shape)

    if not adv_guided_map is None:
        noise = noise * adv_guided_map

    out = img + noise
    out = np.clip(out, 0, 255)
    out = np.uint8(out)
    return out


def denorm(img):
    return (img + 1) / 2


def saving_image(image, dir):
    # from tensor -> numpy
    image = denorm(image)
    image = image.squeeze()
    if not len(image.shape) == 3:
        raise AssertionError("shape should be [c,h,w]")
    image = image.mul(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).to("cpu", torch.uint8).numpy()
    image = Image.fromarray(image)
    image.save(dir)


def concludescore(y_pred, y_true):
    score_result = []

    #计算准确率
    Accuracy = accuracy_score(y_true, y_pred)

    #计算精确率
    Precision = precision_score(y_true, y_pred, average='macro')
    #precision_score(y_true, y_pred, average='micro')
    # precision_score(y_true, y_pred, average='weighted')
    # precision_score(y_true, y_pred, average=None)

    #计算召回率
    Recall = recall_score(y_true, y_pred, average='macro')
    # recall_score(y_true, y_pred, average='micro')
    # recall_score(y_true, y_pred, average='weighted')
    # recall_score(y_true, y_pred, average=None)

    score_result.append(Accuracy)
    score_result.append(Precision)
    score_result.append(Recall)
    return score_result


def drawAUC_TwoClass(y_true, y_score):
    fpr, tpr, thresholds = roc_curve(y_true, y_score)
    roc_auc = auc(fpr, tpr)  #auc为Roc曲线下的面积
    #开始画ROC曲线
    plt.plot(fpr, tpr, 'b', label='AUC = %0.2f' % roc_auc)
    plt.legend(loc='lower right')  #设定图例的位置，右下角
    plt.plot([0, 1], [0, 1], 'r--')
    plt.xlim([-0.1, 1.1])
    plt.ylim([-0.1, 1.1])
    plt.xlabel('False Positive Rate')  #横坐标是fpr
    plt.ylabel('True Positive Rate')  #纵坐标是tpr
    plt.title('Receiver operating characteristic example')
    if os.path.exists('./resultphoto') == False:
        os.makedirs('./resultphoto')
    plt.savefig('AUC_TwoClass.png', format='png')


if __name__ == "__main__":
    dir = 'Stargan/test/fake/0000.png'
    img = Image.open(dir).convert('RGB')
    out_image = gaussian_noise(img, mean=0, sigma=5)
    out_image.save('good.png')
