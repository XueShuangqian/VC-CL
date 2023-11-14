import torch
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np


def kl_div_with_logit(q_logit, p_logit):

    q_logit = q_logit.float()
    p_logit = p_logit.float()

    q_logit = q_logit.view(q_logit.size(0), q_logit.size(1), -1).mean(dim=2)
    p_logit = p_logit.view(p_logit.size(0), p_logit.size(1), -1).mean(dim=2)

    q = F.softmax(q_logit, dim=1)
    logq = F.log_softmax(q_logit, dim=1)
    logp = F.log_softmax(p_logit, dim=1)

    qlogq = ( q *logq).sum(dim=1).mean(dim=0)
    qlogp = ( q *logp).sum(dim=1).mean(dim=0)

    return qlogq - qlogp


def _l2_normalize(d):

    d = d.cpu().numpy()
    d /= (np.sqrt(np.sum(d ** 2, axis=(2, 3, 4))).reshape((-1, 1, 1, 1, 1)) + 1e-16)
    return torch.from_numpy(d)


def vat(model1, model2, ul_x, ul_y, xi=1e-6, eps=2.5, num_iters=10):

    # find r_adv

    d = torch.clamp(torch.randn_like(ul_x) * 0.1, -0.2, 0.2) # b,c,d,w,h
    for i in range(num_iters):

        d = xi *_l2_normalize(d)
        d = d.cuda().requires_grad_()

        y_hat1 = model1(ul_x + d)
        y_hat1_soft = torch.softmax(y_hat1, dim=1)
        y_hat2 = model2(ul_x + d)
        y_hat2_soft = torch.softmax(y_hat2, dim=1)

        output_soft = torch.stack([y_hat1_soft, y_hat2_soft], dim=0)
        output_mean_soft = torch.mean(output_soft, dim=0)


        delta_kl = kl_div_with_logit(ul_y.detach(), output_mean_soft)
        delta_kl = delta_kl.unsqueeze(0)
        delta_kl.backward()

        d = d.grad.data.clone().cpu()

        model1.zero_grad()
        model2.zero_grad()

    d = _l2_normalize(d)
    #d = Variable(d.cuda())
    d = d.cuda()
    r_vat = eps *d
    # compute lds
    y_hat1 = model1(ul_x + r_vat.detach())
    y_hat1_soft = torch.softmax(y_hat1, dim=1)
    y_hat2 = model2(ul_x + r_vat.detach())
    y_hat2_soft = torch.softmax(y_hat2, dim=1)
    output_soft = torch.stack([y_hat1_soft, y_hat2_soft], dim=0)
    output_mean_soft = torch.mean(output_soft, dim=0)
    output = torch.argmax(output_mean_soft, dim=1)

    delta_kl = kl_div_with_logit(ul_y.detach(), output_mean_soft)
    return r_vat, delta_kl


def entropy_loss(ul_y):
    p = F.softmax(ul_y, dim=1)
    return -(p*F.log_softmax(ul_y, dim=1)).sum(dim=1).mean(dim=0)
