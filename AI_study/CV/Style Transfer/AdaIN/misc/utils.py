import torch


def get_mean_std(feat, eps = 1e-8):

    size = feat.size()
    B, C = size[:2]

    feat_var  = feat.view(B, C, -1).var(dim = 2) + eps
    feat_std  = feat_var.sqrt().view(B, C, 1, 1)
    feat_mean = feat.view(B, C, -1).mean(dim = 2).view(B, C, 1, 1) 

    return feat_mean, feat_std


def AdaIN(content_feat, style_feat):

    size          = content_feat.size()
    c_mean, c_std = get_mean_std(content_feat)
    s_mean, s_std = get_mean_std(style_feat)

    normalized_feat = (content_feat - c_mean.expand(size)) / c_std.expand(size)
    return s_std.expand(size) * normalized_feat + s_mean.expand(size)


def _get_flatten_mean_std(feat):

    flatten = feat.view(3, -1)
    mean    = flatten.mean(dim = -1, keepdim = True)
    std     =  flatten.std(dim = -1, keepdim = True)

    return flatten, mean, std


def _mat_sqrt(x):

    U, D, V = torch.svd(x)
    return torch.mm(torch.mm(U, D.pow(0.5).diag()), V.t())


def coral(src, dst):

    src_f, src_mean, src_std = _get_flatten_mean_std(src)
    dst_f, dst_mean, dst_std = _get_flatten_mean_std(dst)

    src_norm = (src_f - src_mean.expand_as(src_f)) / src_std.expand_as(src_f)
    dst_norm = (dst_f - dst_mean.expand_as(dst_f)) / dst_std.expand_as(dst_f)

    src_cov_eye = torch.mm(src_norm, src_norm.t()) + torch.eye(3)
    dst_cov_eye = torch.mm(dst_norm, dst_norm.t()) + torch.eye(3)

    src_norm_transfer = torch.mm(_mat_sqrt(dst_cov_eye), 
                                torch.mm(torch.inverse(_mat_sqrt(src_cov_eye)), src_norm))

    src_transfer      = src_norm_transfer * \
                        dst_std.expand_as(src_norm) + dst_mean.expand_as(src_norm)

    return src_transfer.view(src.size())