import os
import argparse
import time
import datetime
import torch
import numpy as np
import sys
import json
import matplotlib.pyplot as plt
from prettytable import PrettyTable
from models import Efficientnet_B3_CBAM_CA_Opacity_Segmask_Text_Trm
from torch.utils.data import Dataset
import torch.nn as nn
from PIL import Image
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
from torchvision.transforms import transforms

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'


class ConfusionMatrix(object):
    def __init__(self, args, num_classes: int, labels: list, epochs: int):
        self.matrix = np.zeros((num_classes, num_classes))
        self.num_classes = num_classes
        self.labels = labels
        self.Precision_sum = 0.
        self.Recall_sum = 0.
        self.Specificity_sum = 0.
        self.Accuracy_sum = 0.
        self.F1_score_sum = 0.
        self.epochs = epochs
        self.args = args

    def update(self, preds, labels):
        for p, t in zip(preds, labels):
            self.matrix[p, t] += 1

    def summary(self):
        # calculate accuracy
        sum_TP = 0
        for i in range(self.num_classes):
            sum_TP += self.matrix[i, i]
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

            accuracy = round((TP + TN) / all_samples, 3) if all_samples != 0 else 0.
            self.Accuracy_sum += accuracy

            F1_score = round(2 * Precision * Recall / (Precision + Recall), 3)
            print(f"F1-score: {F1_score}")
            self.F1_score_sum += F1_score

            table.add_row([self.labels[i], Precision, Recall, Specificity, accuracy, F1_score])

        print(table)
        print(f"Precision:{round(self.Precision_sum/4, 3)} | "
              f"Recall:{round(self.Recall_sum/4, 3)} | "
              f"Specificity:{round(self.Specificity_sum/4, 3)} | "
              f"Accuracy:{round(self.Accuracy_sum/4, 3)} | "
              f"F1_score:{round(self.F1_score_sum/4, 3)}")
        table_path = f'./table/{self.args.model_name}'
        if not os.path.exists(table_path):
            os.makedirs(table_path)
        with open(f'{table_path}/{self.epochs}.txt', 'w') as f:
            f.write(f"\n {self.epochs}: \n {table} \n "
                    f"Precision:{round(self.Precision_sum / 4, 3)} | "
                    f"Recall:{round(self.Recall_sum / 4, 3)} | "
                    f"Specificity:{round(self.Specificity_sum / 4, 3)} | "
                    f"Accuracy:{round(self.Accuracy_sum / 4, 3)} | "
                    f"F1_score:{round(self.F1_score_sum / 4, 3)}")

    def plot(self):
        matrix = self.matrix
        plt.imshow(matrix, cmap=plt.cm.Blues)

        plt.xticks(range(self.num_classes), self.labels, rotation=45)
        plt.yticks(range(self.num_classes), self.labels)
        plt.colorbar()
        plt.xlabel('True Labels')
        plt.ylabel('Predicted Labels')
        plt.title('Confusion matrix')

        thresh = matrix.max() / 2
        for x in range(self.num_classes):
            for y in range(self.num_classes):
                info = int(matrix[y, x])
                plt.text(x, y, info,
                         verticalalignment='center',
                         horizontalalignment='center',
                         color="white" if info > thresh else "black")
        plt.tight_layout()
        plt.show(block=False)
        save_path = os.path.join(f'./figures_{self.args.model_name}')
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        plt.savefig(os.path.join(save_path, f'ConfusionMatrix-{self.epochs}.png'))
        plt.pause(3)
        plt.close()


class FocalLoss(nn.Module):
    def __init__(self, args, alpha=None, gamma=5.0):
        super(FocalLoss, self).__init__()
        self.args = args
        self.gamma = gamma
        self.alpha = alpha
        self.cross_entropy_loss = nn.CrossEntropyLoss(reduction='none')

    def forward(self, inputs, targets):
        bce = self.cross_entropy_loss(inputs, targets)

        if self.alpha is not None:
            alpha_weights = torch.tensor([self.alpha[i] for i in targets]).to(self.args.device)
            bce = torch.mul(bce, alpha_weights)

        pt = torch.exp(-bce)
        focal_loss = ((1 - pt) ** self.gamma) * bce
        return focal_loss.mean()


def adjust_learning_rate(config, optimizer, epoch):  #
    lr0 = config.lr
    lr = lr0 * (1 - epoch / config.epochs) ** 0.9
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return lr


def train_one_epoch(model, optimizer, data_loader, device, epoch, criteria):
    t0 = time.time()
    model.train()
    mean_loss = torch.zeros(1).to(device)

    sum_num = torch.zeros(1).to(device)
    num_samples = len(data_loader.dataset)

    data_loader = tqdm(data_loader, desc='train')
    for step, data in enumerate(data_loader):
        via, opacity, segmask, text, labels = data
        via = via.to(device)
        opacity = opacity.to(device)
        segmask = segmask.to(device)
        text = text.to(device)
        labels = labels.to(device)
        pred = model(via, opacity, segmask, text)
        loss = criteria(pred, labels.to(device))
        pred = torch.max(pred, dim=1)[1]
        sum_num += torch.eq(pred, labels.to(device)).sum()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        mean_loss = (mean_loss * step + loss.detach()) / (step + 1)  # update mean losses
        data_loader.desc = "[epoch {}] mean loss {}".format(epoch, round(mean_loss.item(), 3))

        if not torch.isfinite(loss):
            print('WARNING: non-finite loss, ending training ', loss)
            sys.exit(1)

    acc = sum_num.item() / num_samples
    print(f"time：{time.time() - t0}")
    return acc, mean_loss.item()


@torch.no_grad()
def evaluate(args, model, data_loader, device, criteria, epochs):
    # read class_indict
    json_label_path = './class_indices.json'
    assert os.path.exists(json_label_path), f"cannot find {json_label_path} file"
    json_file = open(json_label_path, 'r')
    class_indict = json.load(json_file)
    labels = [label for _, label in class_indict.items()]

    confusion = ConfusionMatrix(args, num_classes=4, labels=labels, epochs=epochs)

    model.eval()

    sum_num = torch.zeros(1).to(device)
    num_samples = len(data_loader.dataset)
    num = 0
    loss = 0
    with torch.no_grad():
        data_loader = tqdm(data_loader, desc="valid")
        for step, data in enumerate(data_loader):
            via, opacity, segmask, text, labels = data
            via = via.to(device)
            opacity = opacity.to(device)
            segmask = segmask.to(device)
            text = text.to(device)
            labels = labels.to(device)
            pred = model(via, opacity, segmask, text)
            num += via.shape[0]
            loss += criteria(pred, labels)
            pred = torch.max(pred, dim=1)[1]
            sum_num += torch.eq(pred, labels).sum()
            confusion.update(pred.to("cpu").numpy(), labels.to("cpu").numpy())
    loss_val = loss / num
    acc = sum_num.item() / num_samples

    return acc, confusion, loss_val


def read_split_data(train_path, val_path):
    train_images_path = []
    val_images_path = []

    with open(train_path, 'r') as f:
        lines = [line.rstrip() for line in f.readlines()]
        for line in lines:
            train_images_path.append(line)

    with open(val_path, 'r') as f:
        lines = [line.rstrip() for line in f.readlines()]
        for line in lines:
            val_images_path.append(line)

    print("{} images for training.".format(len(train_images_path)))
    print("{} images for validation.".format(len(val_images_path)))

    return train_images_path, val_images_path


class MyDataSet(Dataset):

    def __init__(self, images_path: list, transform=None):
        self.images_path = images_path
        self.transform = transform

    def __len__(self):
        return len(self.images_path)

    def __getitem__(self, item):
        file = self.images_path[item]
        via_path, opacity_path, segmask_path, text_info, label = file.split('\t')
        via = Image.open(via_path)
        opacity = Image.open(opacity_path)
        opacity = np.array(opacity)
        opacity = transforms.ToPILImage()(opacity)
        opacity = opacity.convert('RGB')

        segmask = Image.open(segmask_path)
        segmask = np.array(segmask)
        segmask = transforms.ToPILImage()(segmask)
        segmask = segmask.convert('RGB')

        text_info = [int(c) for c in text_info]

        label = int(label)

        if via.mode != 'RGB':
            raise ValueError("image: {} isn't RGB mode.".format(self.images_path[item]))
        # if self.transform is not None and label == 3:
        if self.transform is not None:
            via = self.transform(via)
            opacity = self.transform(opacity)
            segmask = self.transform(segmask)

        opacity = np.array(opacity)
        opacity = opacity[0, :, :]

        segmask = np.array(segmask)
        segmask = segmask[0, :, :]

        return via, opacity, segmask, text_info, label

    @staticmethod
    def collate_fn(batch):
        via, opacity, segmask, text_info, label = tuple(zip(*batch))
        via = torch.stack(via, dim=0)
        opacity = torch.as_tensor(np.array(opacity))
        segmask = torch.as_tensor(np.array(segmask))
        text_info = torch.as_tensor(text_info)
        label = torch.as_tensor(label)
        return via, opacity, segmask, text_info, label


# 自定义的回调函数
class AddDataAugmentation:
    def __init__(self, start_epoch, augmentation_function):
        self.start_epoch = start_epoch
        self.augmentation_function = augmentation_function

    def __call__(self, epoch, dataset):
        # if epoch >= self.start_epoch:
        dataset.transform = self.augmentation_function
        print("Data augmentation is applied.")


def main(args):
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    tb_writer_train = SummaryWriter(args.log_path + f"./{args.model_name}/train")
    tb_writer_val = SummaryWriter(args.log_path + f"./{args.model_name}/val")

    if os.path.exists(args.save_path) is False:
        os.makedirs(args.save_path)

    train_images_path, val_images_path = read_split_data(args.train_path, args.val_path)

    data_transform = {
        "train": transforms.Compose([  # transforms.RandomResizedCrop((400, 400)),
            # transforms.CenterCrop((384, 384)),
            # transforms.RandomVerticalFlip(),
            # TrivialAugmentWide(),
            # transforms.RandomHorizontalFlip(),
            # transforms.RandomRotation(degrees=10),
            # transforms.RandomErasing(),
            transforms.ToTensor(),
            # transforms.ConvertImageDtype(torch.float),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]),

        "val": transforms.Compose([
            # transforms.Resize((512, 512)),
            # transforms.CenterCrop((384, 384)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])
    }

    without_augmentation_function = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    augmentation_function = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomRotation(degrees=[-10, 10]),
        transforms.ToTensor(),
        # transforms.RandomErasing(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    train_data_set = MyDataSet(images_path=train_images_path,
                               transform=data_transform["train"])

    val_data_set = MyDataSet(images_path=val_images_path,
                             transform=data_transform["val"])

    batch_size = args.batch_size
    nw = min([os.cpu_count(), batch_size if batch_size > 1 else 0, 8])  # number of workers
    print('Using {} dataloader workers every process'.format(nw))
    train_loader = torch.utils.data.DataLoader(train_data_set,
                                               batch_size=batch_size,
                                               shuffle=True,
                                               pin_memory=True,
                                               num_workers=nw,
                                               collate_fn=train_data_set.collate_fn)

    val_loader = torch.utils.data.DataLoader(val_data_set,
                                             batch_size=batch_size,
                                             shuffle=True,
                                             pin_memory=True,
                                             num_workers=nw,
                                             collate_fn=val_data_set.collate_fn)

    # 实例化模型
    model = Efficientnet_B3_CBAM_CA_Opacity_Segmask_Text_Trm(args)

    # if args.weights:
    #     state_dict = torch.load(args.weights)
    #     # load_weights_dict = {k: v for k, v in state_dict.items()}
    #     unload = ['classifier.1.weight', 'classifier.1.bias', 'fc2.weight', 'fc2.bias']
    #     state_dict = {k: v for k, v in state_dict.items() if k in model.state_dict() and k not in unload}
    #     model.load_state_dict(state_dict, strict=False)
    #     print("load pre trained successfully!!!!")

    model.to(device)

    # optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=args.weight_decay)

    if args.resume:
        checkpoint = torch.load(args.resume)
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        args.start_epoch = checkpoint['epoch'] + 1
        print('start epoch {}'.format(args.start_epoch))
        for param_group in optimizer.param_groups:
            param_group['lr'] = args.lr

    start_time = time.time()
    best_acc = 0.0
    best_recall = 0.0

    # 创建回调函数实例
    without_data_augmentation_callback = AddDataAugmentation(start_epoch=10,
                                                             augmentation_function=without_augmentation_function)
    data_augmentation_callback = AddDataAugmentation(start_epoch=20, augmentation_function=augmentation_function)

    criteria = FocalLoss(args=args, alpha=[0.9, 1.0, 0.8, 0.9], gamma=5.0)

    recall = {}
    for epoch in range(args.start_epoch, args.epochs):

        if epoch % 2 == 1:
            without_data_augmentation_callback(epoch, train_data_set)  # 如果epoch是奇数，就不要增强
        if epoch % 2 == 0:
            data_augmentation_callback(epoch, train_data_set)  # 如果epoch是偶数，就增强

        # train
        current_lr = adjust_learning_rate(args, optimizer, epoch)  # 调节学习率
        train_acc, train_loss = train_one_epoch(model=model,
                                                optimizer=optimizer,
                                                data_loader=train_loader,
                                                device=device,
                                                epoch=epoch,
                                                criteria=criteria)
        # validate
        val_acc, confusion, val_loss = evaluate(args, model=model, data_loader=val_loader, device=device,
                                                criteria=criteria, epochs=epoch)
        confusion.summary()

        current_recall = round(confusion.Recall_sum / 4, 3)
        recall[epoch] = current_recall

        confusion.plot()

        print(f"[epoch {epoch}] accuracy: {round(val_acc, 3)}")

        module = 'train_paper'
        tb_writer_train.add_scalar(f'{module}/acc', train_acc, epoch)
        tb_writer_train.add_scalar(f'{module}/loss', train_loss, epoch)
        tb_writer_val.add_scalar(f'{module}/acc', val_acc, epoch)
        tb_writer_val.add_scalar(f'{module}/loss', val_loss, epoch)
        tb_writer_val.add_scalar(f'{module}/recall', current_recall, epoch)
        # tb_writer_val.add_scalar(f'{module}/lr', optimizer.param_groups[0]["lr"], epoch)
        tb_writer_val.add_scalar(f'{module}/lr', current_lr, epoch)

        save_file = {"model": model.state_dict(),
                     "optimizer": optimizer.state_dict(),
                     "epoch": epoch,
                     "args": args}
        torch.save(save_file, os.path.join(args.save_path, f'{args.model_name}_current.pth'))

        if best_acc < val_acc:
            best_acc = val_acc
            torch.save(save_file, os.path.join(args.save_path, f"{args.model_name}_best_acc.pth"))

        if best_recall < current_recall:
            best_recall = current_recall
            torch.save(save_file, os.path.join(args.save_path, f"{args.model_name}_best_recall.pth"))

        max_key = max(recall, key=recall.get)
        max_value = recall[max_key]
        print(f"\ncurrent best recall: epoch: {max_key} \t recall: {max_value}\n")

    final_max_key = max(recall, key=recall.get)
    final_max_value = recall[final_max_key]
    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print(f"best recall: epoch: {final_max_key} \t recall: {final_max_value}")
    print(f"training time {total_time_str}")


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

    # ----------------- 超参数 ------------------- #
    parser.add_argument('--num_classes', type=int, default=4)
    parser.add_argument('--epochs', type=int, default=500)
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--weight_decay', type=float, default=1e-4)
    parser.add_argument('--start-epoch', default=1, type=int, metavar='N', help='start epoch')
    parser.add_argument('--model_name', type=str,
                        default='efficientnet_b3_cbam_ca_opacity_segmask_text_trm-real')

    # -------------- 数据集所在根目录 ---------------- #
    parser.add_argument('--train_path', type=str, default=r"./train.txt")
    parser.add_argument('--val_path', type=str, default=r"./val.txt")

    # -------------- 训练日志存放路径 --------------- #
    parser.add_argument('--log_path', type=str,
                        default=fr'./runs/{datetime.date.today()}-{datetime.datetime.now().hour}')

    # --------------- 模型保存路径 ---------------- #
    parser.add_argument('--save_path', type=str, default=fr"./save-weights-{datetime.date.today()}")

    # ---------------- 预训练权重 ----------------- #
    # parser.add_argument('--weights', default=pre_trained_dir, type=str, help='initial weights path')
    parser.add_argument('--weights', default=None, type=str, help='initial weights path')

    # --------------- 选择显卡号 ------------------ #
    parser.add_argument('--device', default='cuda:1', help='device id (i.e. 0 or 0,1 or cpu)')

    # --------------- 中断继续训练加载模型权重路径 ----------------- #
    parser.add_argument('--resume', default=None, help='resume from checkpoint')
    # parser.add_argument('--resume', default=resume_root_dir, help='resume from checkpoint')

    opt = parser.parse_args()

    main(opt)
