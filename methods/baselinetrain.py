import backbone
import tqdm

import torch.nn as nn
from torch.autograd import Variable


class BaselineTrain(nn.Module):
    def __init__(self, model_func, num_class, loss_type="softmax"):
        super(BaselineTrain, self).__init__()
        self.feature = model_func()
        if loss_type == "softmax":
            self.classifier = nn.Linear(self.feature.final_feat_dim, num_class)
            self.classifier.bias.data.fill_(0)
        elif loss_type == "dist":  # Baseline ++
            self.classifier = backbone.distLinear(
                self.feature.final_feat_dim, num_class
            )
        self.loss_type = loss_type  #'softmax' #'dist'
        self.num_class = num_class
        self.loss_fn = nn.CrossEntropyLoss()

    def forward(self, x):
        x = Variable(x.cuda())
        out = self.feature.forward(x)
        scores = self.classifier.forward(out)
        return scores

    def forward_loss(self, x, y):
        scores = self.forward(x)
        y = Variable(y.cuda())
        return self.loss_fn(scores, y)

    def train_loop(self, epoch, train_loader, optimizer):
        avg_loss = 0

        pbar = tqdm.tqdm(train_loader)
        i = 0
        for (x, y) in pbar:
            i += 1
            optimizer.zero_grad()
            loss = self.forward_loss(x, y)
            loss.backward()
            optimizer.step()

            avg_loss = avg_loss + loss.data.item()

            pbar.set_description(
                "Epoch {:d} | Batch {:d}/{:d} | Loss {:f}".format(
                    epoch, i, len(train_loader), avg_loss / float(i)
                )
            )

        return avg_loss / float(i)

    def test_loop(self, val_loader):
        return -1  # no validation, just save model during iteration
