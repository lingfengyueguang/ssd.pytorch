#codiing=utf-8
import torch
from torch.autograd import Function
from ..box_utils import decode, nms
from data import voc as cfg


class Detect(Function):
    """At test time, Detect is the final layer of SSD.  Decode location preds,
    apply non-maximum suppression to location predictions based on conf
    scores and threshold to a top_k number of output predictions for both
    confidence score and locations.
    """
    def __init__(self, num_classes, bkg_label, top_k, conf_thresh, nms_thresh):
        self.num_classes = num_classes
        self.background_label = bkg_label
        self.top_k = top_k
        # Parameters used in nms.
        self.nms_thresh = nms_thresh
        if nms_thresh <= 0:
            raise ValueError('nms_threshold must be non negative.')
        self.conf_thresh = conf_thresh
        self.variance = cfg['variance']

    def forward(self, loc_data, conf_data, prior_data):
        """
        Args:
            loc_data: (tensor) Loc preds from loc layers           预测框的偏移量
                Shape: [batch,num_priors*4]
            conf_data: (tensor) Shape: Conf preds from conf layers   每个预测框中各类别的得分
                Shape: [batch*num_priors,num_classes]
            prior_data: (tensor) Prior boxes and variances from priorbox layers       默认框
                Shape: [1,num_priors,4]

        该函数根据默认框和预测框计算出最终的检测框
        """
        num = loc_data.size(0)  # batch size
        num_priors = prior_data.size(0)  #默认框的数目
        output = torch.zeros(num, self.num_classes, self.top_k, 5)
        conf_preds = conf_data.view(num, num_priors,
                                    self.num_classes).transpose(2, 1)

        # Decode predictions into bboxes.
        for i in range(num):
            """
            由预测的偏移值和默认框生成最终的预测框
            偏移值和默认框为[x,y,w,h]形式，解码的预测框为[xmin,ymin,xmax,ymax]形式
            decode_boxes的shape为[num_priors,4]
            """
            decoded_boxes = decode(loc_data[i], prior_data, self.variance)
            # For each class, perform nms
            #一张图片中所有的预测框得分，shape为[num_classes,num_priors]
            conf_scores = conf_preds[i].clone()
            #类内做nms
            for cl in range(1, self.num_classes):
                c_mask = conf_scores[cl].gt(self.conf_thresh)  #选择大于设置的阈值的得分掩码
                scores = conf_scores[cl][c_mask]               #筛选大于阈值的得分
                if scores.dim() == 0:                          #如果当前类没有符合条件的预测框，继续下一个类的循环
                    continue
                """
                decoded_boxes[l_mask]其实是一维的,排列方式形如[x1min,y1min,x1max,y1max,x2min,y2min,x2max,y2max...]
                因此decoded_boxes[l_mask].view(-1, 4) 才会转变为[num_priors,4]的形状，使得每一行对应一个bbox
                个人感觉大可不必使用l_mask,可将下面代码直接替代为
                boxes = decode_boxes[c_mask]
            
                """
                l_mask = c_mask.unsqueeze(1).expand_as(decoded_boxes)   #c_mask是一维的，将其扩展为[num_priors,4]
                boxes = decoded_boxes[l_mask].view(-1, 4)               
                # idx of highest scoring and non-overlapping boxes per class
                ids, count = nms(boxes, scores, self.nms_thresh, self.top_k)
                """
                构造输出，最后一个output为(batchsize, num_classes, self.top_k, 5)形状
                最后一纬的5个数为[score,xmin,ymin,xmax,ymax]
                """
                output[i, cl, :count] = \
                    torch.cat((scores[ids[:count]].unsqueeze(1),
                               boxes[ids[:count]]), 1)
        flt = output.contiguous().view(num, -1, 5)
        _, idx = flt[:, :, 0].sort(1, descending=True)
        _, rank = idx.sort(1)
        flt[(rank < self.top_k).unsqueeze(-1).expand_as(flt)].fill_(0)
        return output
