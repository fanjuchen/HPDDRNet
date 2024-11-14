from mmengine.evaluator import BaseMetric
import torch
from mmseg.registry import METRICS


@METRICS.register_module()
class CustomAAccMetric(BaseMetric):
    def __init__(self, num_classes, collect_device: str = 'cpu', prefix: str = None, **kwargs):
        super().__init__(collect_device=collect_device, prefix=prefix)

        self.num_classes = num_classes

        self.total_correct_per_class = torch.zeros(num_classes)

        self.total_label_per_class = torch.zeros(num_classes)

        self.iou_sum = torch.zeros(num_classes)

        self.class_count = torch.zeros(num_classes)

        self.results = []  # 初始化 results 用于存储中间结果

    def process(self, data_batch, data_samples):

        for sample in data_samples:

            pred = sample['pred_sem_seg']['data'].squeeze().argmax(dim=0)  # 获取预测分割结果

            gt = sample['gt_sem_seg']['data'].squeeze(0)  # 获取真实分割标签

            for i in range(self.num_classes):

                pred_i = (pred == i)

                gt_i = (gt == i)

                intersection = (pred_i & gt_i).sum().item()

                union = (pred_i | gt_i).sum().item()

                if union > 0:
                    self.iou_sum[i] += intersection / union

                    self.class_count[i] += 1

                # 计算每个类的aAcc

                self.total_correct_per_class[i] += intersection

                self.total_label_per_class[i] += gt_i.sum().item()

            # 将每个样本的结果添加到 self.results 中

            self.results.append({'pred': pred, 'gt': gt})

    def compute_metrics(self, results=None):

        # 计算每个类的aAcc

        aAcc_per_class = self.total_correct_per_class / torch.clamp(self.total_label_per_class, min=1)

        aAcc_per_class = aAcc_per_class.tolist()  # 转换为列表以便打印或记录

        # 计算平均IoU

        iou_per_class = self.iou_sum / torch.clamp(self.class_count, min=1)

        iou_per_class = iou_per_class.tolist()  # 转换为列表以便打印或记录

        # 计算整体平均aAcc和mIoU

        overall_aAcc = self.total_correct_per_class.sum().item() / self.total_label_per_class.sum().item()

        overall_mIoU = sum(iou_per_class) / len(iou_per_class)

        # 生成每个类别的aAcc和IoU的字典

        class_metrics = {f'class_{i}_aAcc': aAcc for i, aAcc in enumerate(aAcc_per_class)}

        class_metrics.update({f'class_{i}_IoU': iou for i, iou in enumerate(iou_per_class)})

        # 返回整体平均aAcc、mIoU和每个类别的aAcc和IoU

        return {

            'overall_aAcc': overall_aAcc,

            'overall_mIoU': overall_mIoU,

            **class_metrics

        }