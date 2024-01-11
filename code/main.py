import glob
import os
import pandas as pd
import numpy as np
import torch.cuda
import torchvision
from PIL import Image
from torch import nn, optim
from torch.utils import data
from torchvision import transforms
from torchvision.models import DenseNet161_Weights, DenseNet121_Weights, DenseNet201_Weights
import torchvision.transforms.functional as TF
import new_dense_net


def get_picture_path(dir_path):  # 获取所有鸟类图片的路径

    bird_dir_list = os.listdir(dir_path)
    bird_species = dict()
    picture_paths = []
    each_bird_species = []  # 储存每一个鸟类对应的种类
    for each_bird in bird_dir_list:
        bird_name = each_bird.split('.')[1]
        this_bird_picture_path = glob.glob(dir_path + '/' + each_bird + '/*.jpg')
        for i in range(len(this_bird_picture_path)):
            each_bird_species.append(bird_name)
        picture_paths.append(this_bird_picture_path)
        bird_species[bird_name] = len(this_bird_picture_path)
    bird_statistics = pd.DataFrame(list(bird_species.items()), columns=['species', 'picture_number'])  # 统计每一个鸟的种类的图片数量

    picture_paths = list(np.concatenate(picture_paths))
    return picture_paths, each_bird_species, bird_statistics


def digitize_bird_tags(each_bird_species, bird_statistics):  # 将鸟的种类名称数字化
    digital_bird_species = []
    for bird in each_bird_species:
        digital_bird_species.append(bird_statistics[bird_statistics.species == bird].index.tolist()[0])
    return digital_bird_species


def change_list_random(a1, a2):  # 将列表随机打乱
    np.random.seed(117)
    random_list_number = np.random.permutation(len(a1))
    a1 = np.array(a1)[random_list_number]
    a2 = np.array(a2)[random_list_number]
    return a1, a2


def pad(img, fill=0, size_max=500):
    pad_height = max(0, size_max - img.height)
    pad_width = max(0, size_max - img.width)

    pad_top = pad_height // 2
    pad_bottom = pad_height - pad_top
    pad_left = pad_width // 2
    pad_right = pad_width - pad_left

    return TF.pad(img, (pad_left, pad_top, pad_right, pad_bottom), fill=fill)


fill = tuple(map(lambda x: int(round(x * 256)), (0.485, 0.456, 0.406)))
max_padding = transforms.Lambda(lambda x: pad(x, fill=fill))

transforms_train = transforms.Compose([
    # max_padding,
    max_padding,
    transforms.RandomOrder([
        transforms.RandomCrop((375, 375)),
        transforms.RandomVerticalFlip(),
        transforms.RandomHorizontalFlip()
    ]),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])
transforms_eval = transforms.Compose([
    # max_padding,
    transforms.CenterCrop((375, 375)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])


# 构建鸟类的DataSet
class bird_dataset(data.Dataset):
    def __init__(self, picture, transform, species):
        self.picture = picture
        self.species = species
        self.transform = transform

    def __len__(self):
        return len(self.picture)

    def __getitem__(self, index):

        np_picture = self.picture[index]
        number = self.species[index]

        if len(np_picture.shape) == 2:  # 处理输入时的黑色图片 将它转换成三通道图片
            picture_data = np.repeat(np_picture[:, :, np.newaxis], 3, axis=2)
            picture = Image.fromarray(picture_data)
        else:
            picture = Image.fromarray(np_picture)
        if self.transform == "train":
            picture_tensor = transforms_train(picture)
        else:
            picture_tensor = transforms_eval(picture)
        return picture_tensor, number


if __name__ == '__main__':
    # 获取所有图片的路径、每个图片对应的鸟类种类、鸟类统计数据
    all_picture_paths, English_bird_species, bird_stat = get_picture_path("./images")

    # 将每个图片对应的鸟类种类转换成数字
    digital_bids_species = digitize_bird_tags(English_bird_species, bird_stat)

    # 将路径和数字鸟类种类分类打乱
    all_picture_paths, digital_bids_species = change_list_random(all_picture_paths, digital_bids_species)

    # 将训练集和测试集划分成9:1
    first_part_number = int(len(digital_bids_species) * 0.9)
    train_data_path = all_picture_paths[:first_part_number]
    test_data_path = all_picture_paths[first_part_number:]

    train_data_species = digital_bids_species[:first_part_number]
    test_data_species = digital_bids_species[first_part_number:]

    train_pic = list()
    i = 0
    new_train_data_species = list()
    for data_path in train_data_path:
        pic = Image.open(data_path)
        new_train_data_species.append(train_data_species[i])
        transforms1 = transforms.Compose([
            transforms.RandomRotation(45)
        ])
        pic2 = transforms1(pic)
        new_train_data_species.append(train_data_species[i])
        np_picture = np.asarray(pic, dtype=np.uint8)
        np_picture2 = np.asarray(pic2, dtype=np.uint8)
        i += 1

        train_pic.append(np_picture)
        train_pic.append(np_picture2)
        pic.close()

    test_pic = list()
    for data_path in test_data_path:
        pic = Image.open(data_path)
        np_picture = np.asarray(pic, dtype=np.uint8)
        test_pic.append(np_picture)
        pic.close()

    # 测试集和训练集实例化
    train_bird = bird_dataset(train_pic, "train", new_train_data_species)
    test_bird = bird_dataset(test_pic, "eval", test_data_species)

    # 一次训练所抓取的数据样本数量
    BATCH_SIZE = 50
    train_dl = data.DataLoader(
        train_bird,
        batch_size=BATCH_SIZE,
    )
    test_dl = data.DataLoader(
        test_bird,
        batch_size=BATCH_SIZE,
    )

    # bird_densenet = torchvision.models.densenet201(weights=("pretrained", DenseNet201_Weights.IMAGENET1K_V1))
    bird_densenet = new_dense_net.densenet201_se(weights=("pretrained", DenseNet201_Weights.IMAGENET1K_V1))

    DEVICE = torch.device("cuda")
    bird_densenet.classifier = nn.Linear(bird_densenet.classifier.in_features, 200)
    bird_densenet.to(DEVICE)

    loss_function = nn.CrossEntropyLoss()  # Cross entropy loss function
    optimizer = optim.Adam(bird_densenet.parameters(), lr=0.0001)

    train_total_number = len(train_pic)
    test_total_number = len(test_pic)
    max_accuracy = 0.0
    for epoch in range(1, 36):
        output = open('result_201_se_rotation.txt', 'a', encoding='utf-8')
        bird_densenet.train()
        running_loss = 0.0
        train_acc = 0.0
        for step, data in enumerate(train_dl, start=0):
            images, labels = data
            optimizer.zero_grad()

            train_y = bird_densenet(images.to(DEVICE))
            loss = loss_function(train_y, labels.long().to(DEVICE))

            train_predict_y = torch.max(train_y, dim=1)[1]
            train_acc += (train_predict_y == labels.long().to(DEVICE)).sum().item()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()  # calculate the loss
            rate = (step + 1) / len(train_dl)
            a = "*" * int(rate * 50)
            b = "." * int((1 - rate) * 50)
            print("\rtrain loss: {:^3.0f}%[{}->{}]{:.3f}".format(int(rate * 100), a, b, loss), end="")
        train_accurate = train_acc / train_total_number

        bird_densenet.eval()
        test_acc = 0.0
        test_loss = 0.0
        with torch.no_grad():
            for val_data in test_dl:
                val_images, val_labels = val_data
                outputs = bird_densenet(val_images.to(DEVICE))
                predict_y = torch.max(outputs, dim=1)[1]
                test_acc += (predict_y == val_labels.long().to(DEVICE)).sum().item()
                loss = loss_function(outputs, val_labels.long().to(DEVICE))
                test_loss += loss.item()
            val_accurate = test_acc / test_total_number
            if val_accurate > max_accuracy:
                max_accuracy = val_accurate
            output.write('[epoch %d] train_loss: %.3f test_loss: %.3f train_accuracy: %.3f test_accuracy: %.3f\n' %
                         (epoch, running_loss, test_loss, train_accurate, val_accurate))
            print('[epoch %d] train_loss: %.3f test_loss: %.3f train_accuracy: %.3f test_accuracy: %.3f' %
                  (epoch, running_loss, test_loss, train_accurate, val_accurate))
        output.close()
    print('Finished Training.Best accuracy is %.3f' % max_accuracy)
