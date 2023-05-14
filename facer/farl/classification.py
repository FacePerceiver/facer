from torch import nn
import torch
import torch.utils.checkpoint as checkpoint
from .model import VisualTransformer


class VITClassificationHeadV0(nn.Module):
    def __init__(
        self,
        num_features: int,
        channel: int,
        num_labels: int,
        norm=False,
        dropout=0.0,
        ret_feat=False,
    ):
        super().__init__()
        self.weights = nn.Parameter(
            torch.ones(1, num_features * 3, 1, dtype=torch.float32)
        )
        self.final_fc = nn.Linear(channel, num_labels)
        self.norm = norm
        if self.norm:
            for i in range(num_features * 3):
                setattr(self, f"norm_{i}", nn.LayerNorm(channel))
        self.dropout = nn.Dropout(p=dropout)
        self.ret_feat = ret_feat

    def forward(self, features, cls_tokens):
        xs = []
        for feature, cls_token in zip(features, cls_tokens):
            # feature: b x c x s x s
            # cls_token: b x c
            xs.append(feature.mean([2, 3]))
            xs.append(feature.max(-1).values.max(-1).values)
            xs.append(cls_token)
        if self.norm:
            xs = [getattr(self, f"norm_{i}")(x) for i, x in enumerate(xs)]
        xs = torch.stack(xs, dim=1)  # b x 3N x c
        feat = (xs * self.weights.softmax(dim=1)).sum(1)  # b x c
        x = self.dropout(feat)
        x = self.final_fc(x)  # b x num_labels
        if self.ret_feat:
            return x, feat
        else:
            return x


class FACTransformer(nn.Module):
    """A face attribute classification transformer leveraging multiple cls_tokens.
    Args:
        image (torch.Tensor): Float32 tensor with shape [b, 3, h, w], normalized to [0, 1].
    Returns:
        logits (torch.Tensor): Float32 tensor with shape [b, n_classes].
        aux_outputs:
    """

    def __init__(self, backbone: nn.Module, head: nn.Module):
        super().__init__()
        self.backbone = backbone
        self.head = head
        self.cuda().float()

    def forward(self, image):
        logits = self.head(*self.backbone(image))
        return logits


def add_method(obj, name, method):
    import types

    setattr(obj, name, types.MethodType(method, obj))


def get_clip_encode_func(layers):
    def func(self, x):
        x = self.conv1(x)  # shape = [*, width, grid, grid]
        # shape = [*, width, grid ** 2]
        x = x.reshape(x.shape[0], x.shape[1], -1)
        extra_tokens = getattr(self, "extra_tokens", [])
        x = x.permute(0, 2, 1)  # shape = [*, grid ** 2, width]
        class_token = self.class_embedding.to(x.dtype) + torch.zeros(
            x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device
        )
        special_tokens = [
            getattr(self, name).to(x.dtype)
            + torch.zeros(x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device)
            for name in extra_tokens
        ]
        x = torch.cat(
            [class_token, *special_tokens, x], dim=1
        )  # shape = [*, grid ** 2 + 1, width]
        x = x + self.positional_embedding.to(x.dtype)
        x = self.ln_pre(x)
        x = x.permute(1, 0, 2)  # NLD -> LND
        outs = []
        max_layer = max(layers)
        use_checkpoint = self.transformer.use_checkpoint
        for layer_i, blk in enumerate(self.transformer.resblocks):
            if layer_i > max_layer:
                break
            if self.training and use_checkpoint:
                x = checkpoint.checkpoint(blk, x)
            else:
                x = blk(x)
            outs.append(x)

        outs = torch.stack(outs).permute(0, 2, 1, 3)
        cls_tokens = outs[layers, :, 0, :]

        extra_token_feats = {}
        for i, name in enumerate(extra_tokens):
            extra_token_feats[name] = outs[layers, :, i + 1, :]
        L, B, N, C = outs.shape
        import math

        W = int(math.sqrt(N - 1 - len(extra_tokens)))
        features = (
            outs[layers, :, 1 + len(extra_tokens) :, :]
            .reshape(len(layers), B, W, W, C)
            .permute(0, 1, 4, 2, 3)
        )
        if getattr(self, "ret_special", False):
            return features, cls_tokens, extra_token_feats
        else:
            return features, cls_tokens

    return func


def farl_classification(num_classes=2, layers=list(range(12))):
    model = VisualTransformer(
        input_resolution=224,
        patch_size=16,
        width=768,
        layers=12,
        heads=12,
        output_dim=512,
    )
    channel = 768
    model = model.cuda()
    del model.proj
    del model.ln_post
    add_method(model, "forward", get_clip_encode_func(layers))
    head = VITClassificationHeadV0(
        num_features=len(layers), channel=channel, num_labels=num_classes, norm=True
    )
    model = FACTransformer(model, head)
    return model
