import os
import json
import argparse
import time
import torch
import numpy as np
import torch.nn as nn
import matplotlib.pyplot as plt
from prettytable import PrettyTable
from tqdm import tqdm
from torchvision import transforms
from models import Efficientnet_B3_CBAM_CA_Opacity_Segmask_Text_Trm
from torch.utils.data import Dataset
from PIL import Image


class ConfusionMatrix(object):
    def __init__(self, args, num_classes: int, labels: list):
        self.matrix = np.zeros((num_classes, num_classes))
        self.num_classes = num_classes
        self.labels = labels
        self.Precision_sum = 0.
        self.Recall_sum = 0.
        self.Specificity_sum = 0.
        self.Accuracy_sum = 0.
        self.F1_score_sum = 0.
        self.args = args

    def update(self, preds, labels):
        for p, t in zip(preds, labels):
            self.matrix[p, t] += 1

    def summary(self):
        # calculate accuracy
        sum_TP = 0
        for i in range(self.num_classes):
            sum_TP += self.matrix[i, i]
        # precision, recall, specificity, accuracy
        table = PrettyTable()
        table.field_names = ["", "Precision", "Recall", "Specificity", "Accuracy", "F1_score"]

        for i in range(self.num_classes):
            TP = self.matrix[i, i]
            FP = np.sum(self.matrix[i, :]) - TP
            FN = np.sum(self.matrix[:, i]) - TP
            TN = np.sum(self.matrix) - TP - FP - FN
            all_samples = TP + TN + FP + FN

            Precision = round(TP / (TP + FP), 3) if TP + FP != 0 else 0.
            self.Precision_sum += Precision

            Recall = round(TP / (TP + FN), 3) if TP + FN != 0 else 0.
            self.Recall_sum += Recall

            Specificity = round(TN / (TN + FP), 3) if TN + FP != 0 else 0.
            self.Specificity_sum += Specificity

            accuracy = round((TP + TN) / all_samples, 3) if all_samples != 0 else 0.  # OuXue Add
            self.Accuracy_sum += accuracy

            F1_score = round(2 * Precision * Recall / (Precision + Recall), 3)
            print(f"F1-score: {F1_score}")
            self.F1_score_sum += F1_score

            table.add_row([self.labels[i], Precision, Recall, Specificity, accuracy, F1_score])

        print(table)
        print(f"Precision:{round(self.Precision_sum / 4, 3)} | "
              f"Recall:{round(self.Recall_sum / 4, 3)} | "
              f"Specificity:{round(self.Specificity_sum / 4, 3)} | "
              f"Accuracy:{round(self.Accuracy_sum / 4, 3)} | "
              f"F1_score:{round(self.F1_score_sum / 4, 3)}")
        table_path = f'./table/{self.args.model_name}'
        if not os.path.exists(table_path):
            os.makedirs(table_path)
        with open(f'{table_path}/{self.args.model_name}.txt', 'w') as f:
            f.write(f"\n {self.args.model_name}: \n {table} \n "
                    f"Precision:{round(self.Precision_sum / 4, 3)} | "
                    f"Recall:{round(self.Recall_sum / 4, 3)} | "
                    f"Specificity:{round(self.Specificity_sum / 4, 3)} | "
                    f"Accuracy:{round(self.Accuracy_sum / 4, 3)} | "
                    f"F1_score:{round(self.F1_score_sum / 4, 3)}")

    def plot(self):
        matrix = self.matrix
        plt.imshow(matrix, cmap=plt.cm.Blues)

        # 设置x轴坐标label
        plt.xticks(range(self.num_classes), self.labels, rotation=45, fontsize=14)
        # 设置y轴坐标label
        plt.yticks(range(self.num_classes), self.labels, fontsize=14)
        # 显示colorbar
        plt.colorbar()
        plt.xlabel('True Labels', fontsize=14)
        plt.ylabel('Predicted Labels', fontsize=14)
        plt.title('Confusion matrix')

        # 在图中标注数量/概率信息
        thresh = matrix.max() / 2
        for x in range(self.num_classes):
            for y in range(self.num_classes):
                # 注意这里的matrix[y, x]不是matrix[x, y]
                info = int(matrix[y, x])
                plt.text(x, y, info,
                         verticalalignment='center',
                         horizontalalignment='center',
                         color="white" if info > thresh else "black")
        plt.tight_layout()
        plt.show(block=False)
        save_path = './figures'
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        plt.savefig(f'{save_path}/{self.args.model_name}.png')
        plt.pause(3)
        plt.close()


class FocalLoss(nn.Module):
    def __init__(self, args, alpha=None, gamma=5.0):
        # gamma一般取值在1-5之间，较小的gamma值会使难易样本的权重调整较小，
        # 更加关注于难以分类的样本；较大的gamma值则会增加对难易样本的权重调整，对于易分类的样本产生较小的损失
        super(FocalLoss, self).__init__()
        self.args = args
        self.gamma = gamma
        self.alpha = alpha
        self.cross_entropy_loss = nn.CrossEntropyLoss(reduction='none')

    def forward(self, inputs, targets):
        bce = self.cross_entropy_loss(inputs, targets)

        # -------------------- 计算权重调整系数 --------------------- #
        if self.alpha is not None:
            alpha_weights = torch.tensor([self.alpha[i] for i in targets]).to(self.args.device)
            bce = torch.mul(bce, alpha_weights)

        # ---------------- 计算FocalLoss --------------- #
        pt = torch.exp(-bce)
        focal_loss = ((1 - pt) ** self.gamma) * bce
        return focal_loss.mean()


class MyDataSet(Dataset):

    def __init__(self, images_path: list, transform=None):
        self.images_path = images_path
        self.transform = transform

    def __len__(self):
        return len(self.images_path)

    def __getitem__(self, item):  # 这里应该是一个一个遍历
        file = self.images_path[item]
        img_path, opacity, segmask, hpv_age, label = file.split('\t')
        img_save = Image.open(img_path)  # 这里是单个文件
        image = img_save
        label = int(label)
        hpv = [int(c) for c in hpv_age]

        opacity = Image.open(opacity)
        opacity = np.array(opacity)
        opacity = transforms.ToPILImage()(opacity)
        opacity = opacity.convert('RGB')

        segmask = Image.open(segmask)
        segmask = np.array(segmask)
        segmask = transforms.ToPILImage()(segmask)
        segmask = segmask.convert('RGB')

        # RGB为彩色图片，L为灰度图片
        if image.mode != 'RGB':
            raise ValueError("image: {} isn't RGB mode.".format(self.images_path[item]))
        if self.transform is not None:
            image = self.transform(image)
            opacity = self.transform(opacity)
            segmask = self.transform(segmask)

        opacity = np.array(opacity)
        opacity = opacity[0, :, :]

        segmask = np.array(segmask)
        segmask = segmask[0, :, :]

        return image, img_save, img_path, opacity, segmask, hpv, label

    @staticmethod
    def collate_fn(batch):
        images, img_save, img_name, opacity, segmask, hpv, labels = tuple(zip(*batch))
        images = torch.stack(images, dim=0)
        opacity = torch.as_tensor(np.array(opacity))
        segmask = torch.as_tensor(np.array(segmask))
        hpv = torch.as_tensor(hpv)
        labels = torch.as_tensor(labels)
        return images, img_save, img_name, opacity, segmask, hpv, labels


def read_split_data(test_path):
    test_images_path = []  # 存储验证集的所有图片路径

    # 打开txt文件路径并且读取每一行
    with open(test_path, 'r') as f:
        lines = [line.rstrip() for line in f.readlines()]
        # 遍历每一行
        for line in lines:
            test_images_path.append(line)

    print("{} images for test.".format(len(test_images_path)))

    return test_images_path


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6


def main(args):
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(device)

    test_path = read_split_data(args.data_path)

    data_transform = {
        "test": transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])}

    test_data_set = MyDataSet(images_path=test_path,
                              transform=data_transform["test"])

    batch_size = args.batch_size
    # 计算使用num_workers的数量
    nw = min([os.cpu_count(), batch_size if batch_size > 1 else 0, 8])  # number of workers
    print('Using {} dataloader workers every process'.format(nw))
    test_loader = torch.utils.data.DataLoader(test_data_set,
                                              batch_size=batch_size,
                                              shuffle=False,
                                              pin_memory=True,
                                              num_workers=nw,
                                              collate_fn=test_data_set.collate_fn)

    model = Efficientnet_B3_CBAM_CA_Opacity_Segmask_Text_Trm(args)
    num_params = count_parameters(model)
    print(f'model size: {num_params}M')
    model.to(device)

    if args.weights:
        weights_dict = torch.load(args.weights, map_location=device)
        load_weights_dict = weights_dict['model']
        model.load_state_dict(load_weights_dict, strict=True)
        print('load weights successfully!!!')

    # read class_indict
    json_label_path = r'D:\OUXUE\Cervical\class_indices.json'
    json_file = open(json_label_path, 'r')
    class_indict = json.load(json_file)
    labels = [label for _, label in class_indict.items()]
    confusion = ConfusionMatrix(args, num_classes=args.num_classes, labels=labels)

    with torch.no_grad():
        model.eval()
        accu_num = torch.zeros(1).to(device)  # 累计预测正确的样本数
        accu_loss = torch.zeros(1).to(device)  # 累计损失

        sample_num = 0
        data_loader = tqdm(test_loader)
        t1 = time.time()
        for step, data in enumerate(data_loader):
            images, img_save, img_name, opacity, segmask, hpv, labels = data
            images = images.to(device)
            opacity = opacity.to(device)
            segmask = segmask.to(device)
            hpv = hpv.to(device)
            labels = labels.to(device)

            sample_num += images.shape[0]

            pred = model(images, opacity, segmask, hpv)
            pred = torch.softmax(pred, dim=1)

            pred_classes = torch.max(pred, dim=1)[1]
            accu_num += torch.eq(pred_classes, labels).sum()

            # 创建一个保存病例信息的字典
            cls_dict = {0: 'CA', 1: 'HSIL', 2: 'LSIL', 3: 'Normal'}
            for via, via_name, pred_cls, label in zip(img_save, img_name, pred_classes, labels):
                pred_cls = pred_cls.cpu().item()
                label = label.cpu().item()

                save_path = os.path.join(
                    fr'.\{args.model_name}\{cls_dict[label]}-{cls_dict[pred_cls]}')
                if not os.path.exists(save_path):
                    os.makedirs(save_path)
                via.save(os.path.join(save_path, os.path.basename(via_name)))

            data_loader.desc = "[test ] loss: {:.3f}, acc: {:.3f}".format(accu_loss.item() / (step + 1),
                                                                          accu_num.item() / sample_num)

            confusion.update(pred_classes.to("cpu").numpy(), labels.to("cpu").numpy())
    print(f'平均时间： {(time.time()-t1)/sample_num}')
    print("loss:{}".format(accu_loss.item() / (step + 1)))
    print("acc:{}".format(accu_num.item() / sample_num))
    # 输出混淆矩阵
    confusion.summary()
    confusion.plot()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # ---------------- Transformer超参数 ---------------- #
    parser.add_argument('--ch_in', default=1014, type=int)
    parser.add_argument('--num_layers', default=1, type=int)
    parser.add_argument('--hidden_size', default=768, type=int)
    parser.add_argument('--mlp_dim', default=256, type=int)
    parser.add_argument('--dropout_rate', default=0.2, type=int)
    parser.add_argument('--num_heads', default=2, type=int)
    parser.add_argument('--attention_dropout_rate', default=0.0, type=int)
    # parser.add_argument('--num_classes', default=4, type=int)

    parser.add_argument('--num_classes', type=int, default=4)
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--data_path', type=str, default='./test.txt')
    # load weight
    parser.add_argument('--weights', type=str,
                        default='your weight path',
                        help='initial weights path')
    parser.add_argument('--device', default='cuda:1', help='device id (i.e. 0 or 0,1 or cpu)')
    parser.add_argument('--model_name', type=str, default='efficientnet_b3_cbam_ca_opacity_segmask_text_trm')
    opt = parser.parse_args()

    main(opt)
