import torch
import torch.nn as nn


class MDFLoss(nn.Module):
    '''https://github.com/gfxdisp/mdf'''
    def __init__(self, 
                saved_ds_path ='./mdf_weights/Ds_SISR.pth',
                cuda_available = True):
        super(MDFLoss, self).__init__()

        if cuda_available:
            self.Ds = torch.load(saved_ds_path)
        else:
            self.Ds = torch.load(saved_ds_path, map_location=torch.device('cpu'))

        self.num_discs = len(self.Ds)

    def forward(self, x, y, num_scales=8, is_ascending=1):
        ''' x is target, y is predicted image'''
        # Get batch_size
        batch_size = x.shape[0]
        
        # Initialize loss vector
        loss = torch.zeros([batch_size]).to(x.device)
        # For every scale
        for scale_idx in range(num_scales):
            # Reverse if required
            if is_ascending:
                scale = scale_idx
            else:
                scale = self.num_discs - 1 - scale_idx

            # Choose discriminator
            D = self.Ds[scale]
            
            # Get discriminator activations
            pxs = D(x, is_loss=True)
            pys = D(y, is_loss=True)

            # For every layer in the output
            for idx in range(len(pxs)):
                # Compute L2 between representations
                l2 = (pxs[idx] - pys[idx])**2
                l2 = torch.mean(l2, dim=(1, 2, 3))

                # Add current difference to the loss
                loss += l2

        # Mean loss
        loss = torch.mean(loss)

        return loss


# torch.log  and math.log is e based
class AdaptiveWingLoss(nn.Module):
    def __init__(self, omega=14, theta=0.5, epsilon=1, alpha=2.1):
        super(AdaptiveWingLoss, self).__init__()
        self.omega = omega
        self.theta = theta
        self.epsilon = epsilon
        self.alpha = alpha

    def forward(self, pred, target):
        '''
        :param pred: BxNxHxH
        :param target: BxNxHxH
        :return:
        '''

        y = target
        y_hat = pred
        delta_y = (y - y_hat).abs()
        delta_y1 = delta_y[delta_y < self.theta]
        delta_y2 = delta_y[delta_y >= self.theta]
        y1 = y[delta_y < self.theta]
        y2 = y[delta_y >= self.theta]
        loss1 = self.omega * torch.log(1 + torch.pow(delta_y1 / self.omega, self.alpha - y1))
        A = self.omega * (1 / (1 + torch.pow(self.theta / self.epsilon, self.alpha - y2))) * (self.alpha - y2) * (
            torch.pow(self.theta / self.epsilon, self.alpha - y2 - 1)) * (1 / self.epsilon)
        C = self.theta * A - self.omega * torch.log(1 + torch.pow(self.theta / self.epsilon, self.alpha - y2))
        loss2 = A * delta_y2 - C
        return (loss1.sum() + loss2.sum()) / (len(loss1) + len(loss2))


def cosine_loss(a, v, y):
    # print("a : ", a.shape)
    # print("v : ", v.shape)
    logloss = nn.BCELoss()

    d = nn.functional.cosine_similarity(a, v)
    loss = logloss(d.unsqueeze(1), y)

    return loss