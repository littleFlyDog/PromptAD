from .model import *

# def get_model_from_args(**kwargs)->WinClipAD:
#     model = WinClipAD(**kwargs)
#     return model

class TripletLoss(nn.Module):
    def __init__(self, margin=1.0):
        super(TripletLoss, self).__init__()
        self.margin = margin

    def forward(self, anchor, positive, negative):
#seg阶段，2，225，640， positive、[1,640]
        pos_distance = torch.sum((anchor - positive).pow(2), dim=1)

        neg_distance = torch.sum((anchor - negative).pow(2), dim=1)

        loss = torch.relu(pos_distance - neg_distance + self.margin)
#强行要求距离 seg阶段[2,640],返回的是平均值
        return torch.mean(loss)