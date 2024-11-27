import warnings
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter

from transformers.modeling_outputs import SemanticSegmenterOutput
from .FCNHead import FCNHead
from .SegformerHead import SegformerHeadv2

class _LoRA_qkv(nn.Module):
    """In Dinov2 it is implemented as
    self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
    B, N, C = x.shape
    qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
    q, k, v = qkv.unbind(0)
    """

    def __init__(
            self,
            qkv: nn.Module,
            linear_a_q: nn.Module,
            linear_b_q: nn.Module,
            linear_a_v: nn.Module,
            linear_b_v: nn.Module,
    ):
        super().__init__()
        self.qkv = qkv
        self.linear_a_q = linear_a_q
        self.linear_b_q = linear_b_q
        self.linear_a_v = linear_a_v
        self.linear_b_v = linear_b_v
        self.dim = qkv.in_features
        self.w_identity = torch.eye(qkv.in_features)

    def forward(self, x):
        qkv = self.qkv(x)  # B,N,3*org_C
        new_q = self.linear_b_q(self.linear_a_q(x))
        new_v = self.linear_b_v(self.linear_a_v(x))
        
        qkv[:, :, : self.dim] += new_q
        qkv[:, :, -self.dim:] += new_v
        return qkv
    
class RCMNet(nn.Module):
    """Applies low-rank adaptation to Dinov2 model's image encoder.

    Args:
        backbone_size: the pretrained size of dinov2 model
        r: rank of LoRA
        image_shape: input image shape
        decode_type: the decode type of decode head, "linear" or ""

    """

    def __init__(self, backbone_size = "base", r=4, image_shape=(532,532), \
                decode_type = 'linear4', confidence_head='base', lora_layer=None):
        super(RCMNet, self).__init__()

        assert r > 0
        self.backbone_size = backbone_size
        self.backbone_archs = {
            "small": "vits14",
            "base": "vitb14",
            "large": "vitl14",
            "giant": "vitg14",
        }
        self.intermediate_layers = {
            "small": [2, 5, 8, 11],
            "base": [2, 5, 8, 11],
            "large": [4, 11, 17, 23],
            "giant": [9, 19, 29, 39],
        }
        self.embedding_dims = {
            "small": 384,
            "base": 768,
            "large": 1024,
            "giant": 1536,
        }
        self.backbone_arch = self.backbone_archs[self.backbone_size]
        self.n = self.intermediate_layers[self.backbone_size] if decode_type == 'linear4' else 1 
        self.embedding_dim = self.embedding_dims[self.backbone_size]
        
        self.backbone_name = f"dinov2_{self.backbone_arch}"
        dinov2 = torch.hub.load(repo_or_dir="facebookresearch/dinov2", model=self.backbone_name)
        self.image_shape = image_shape
        
        if lora_layer:
            self.lora_layer = lora_layer
        else:
            self.lora_layer = list(
                range(len(dinov2.blocks)))  # Only apply lora to the image encoder by default
        self.decode_type = decode_type
        # create for storage, then we can init them or load weights
        self.w_As = []  # These are linear layers
        self.w_Bs = []
        # freeze first
        for param in dinov2.parameters():
            param.requires_grad = False

        # Here, we do the surgery
        for t_layer_i, blk in enumerate(dinov2.blocks):
            # If we only want few lora layer instead of all
            if t_layer_i not in self.lora_layer:
                continue
            w_qkv_linear = blk.attn.qkv
            self.dim = w_qkv_linear.in_features
            w_a_linear_q = nn.Linear(self.dim, r, bias=False)
            w_b_linear_q = nn.Linear(r, self.dim, bias=False)
            w_a_linear_v = nn.Linear(self.dim, r, bias=False)
            w_b_linear_v = nn.Linear(r, self.dim, bias=False)
            self.w_As.append(w_a_linear_q)
            self.w_Bs.append(w_b_linear_q)
            self.w_As.append(w_a_linear_v)
            self.w_Bs.append(w_b_linear_v)
            blk.attn.qkv = _LoRA_qkv(
                w_qkv_linear,
                w_a_linear_q,
                w_b_linear_q,
                w_a_linear_v,
                w_b_linear_v,
            )
        self.reset_parameters()
        self.dinov2 = dinov2
        # The decode depth estimation head
        
        if self.decode_type == 'linear':
            self.inchannels = [self.embedding_dim]
            self.channels = self.embedding_dim*2
            self.in_index = (0)
            self.input_transform="resize"
        elif self.decode_type == 'linear4':
            self.inchannels = [self.embedding_dim,self.embedding_dim,self.embedding_dim,self.embedding_dim]
            self.channels = self.embedding_dim*8
            self.in_index = (0, 1, 2, 3)
            self.input_transform="resize_concat"
            
        if confidence_head == 'segformer':
            self.confidence_head = SegformerHeadv2(depth_channel=0, num_classes=1, in_channels=[channel * 2 for channel in self.inchannels], \
                                               channels=self.inchannels[0] // 2, image_shape=image_shape)
        
        elif confidence_head == 'FCNHead':
            self.confidence_head = FCNHead(in_channels=[1536,1536,1536,1536],
                                            channels=self.inchannels[0] // 2,
                                            in_index=(0, 1, 2, 3),
                                            image_shape=image_shape,
                                            input_transform='resize_concat',
                                            num_classes=1)
        else:
            raise ValueError(f'no such kind of confidence head: {confidence_head}')
        
        print(f'Using {confidence_head} as confidence head')
       
        self.mse_loss = nn.MSELoss()
        
    def save_parameters(self, filename: str) -> None:
        r"""Only safetensors is supported now.

        pip install safetensor if you do not have one installed yet.

        save both lora and fc parameters.
        """

        assert filename.endswith(".pt") or filename.endswith('.pth')

        num_layer = len(self.w_As)  # actually, it is half
        a_tensors = {f"w_a_{i:03d}": self.w_As[i].weight for i in range(num_layer)}
        b_tensors = {f"w_b_{i:03d}": self.w_Bs[i].weight for i in range(num_layer)}
        confidence_head_tensors = {}

        # save confidence head
        if self.confidence_head is not None:
            if isinstance(self.confidence_head, torch.nn.DataParallel) or isinstance(self.confidence_head, torch.nn.parallel.DistributedDataParallel):
                state_dict = self.confidence_head.module.state_dict()
            else:
                state_dict = self.confidence_head.state_dict()
            for key, value in state_dict.items():
                confidence_head_tensors[key] = value

        merged_dict = {**a_tensors, **b_tensors, **confidence_head_tensors}

        torch.save(merged_dict, filename)

        print('saved lora parameters to %s.' % filename)

    def load_parameters(self, filename: str, device: str) -> None:
        r"""Only safetensors is supported now.

        pip install safetensor if you do not have one installed yet.\

        load both lora and fc parameters.
        """

        assert filename.endswith(".pt") or filename.endswith('.pth')

        state_dict = torch.load(filename, map_location=device)

        for i, w_A_linear in enumerate(self.w_As):
            saved_key = f"w_a_{i:03d}"
            saved_tensor = state_dict[saved_key]
            w_A_linear.weight = Parameter(saved_tensor)

        for i, w_B_linear in enumerate(self.w_Bs):
            saved_key = f"w_b_{i:03d}"
            saved_tensor = state_dict[saved_key]
            w_B_linear.weight = Parameter(saved_tensor)

        if self.confidence_head is not None:
            confidence_head_dict = self.confidence_head.state_dict()
            confidence_head_keys = confidence_head_dict.keys()

            confidence_head_keys = [k for k in confidence_head_keys]
            confidence_head_values = [state_dict[k] for k in confidence_head_keys]
            confidence_head_new_state_dict = {k: v for k, v in zip(confidence_head_keys, confidence_head_values)}
            confidence_head_dict.update(confidence_head_new_state_dict)

            self.confidence_head.load_state_dict(confidence_head_dict)

        print('loaded lora parameters from %s.' % filename)

    def reset_parameters(self) -> None:
        for w_A in self.w_As:
            nn.init.kaiming_uniform_(w_A.weight, a=math.sqrt(5))
        for w_B in self.w_Bs:
            nn.init.zeros_(w_B.weight)

    def forward(self, pixel_values, confidence_gt):
        feature = self.dinov2.get_intermediate_layers(pixel_values, n=self.n, reshape = True, return_class_token = True, norm=False)

        pred_confidence = None
       
        pred_confidence = self.confidence_head(feature, None)

        loss = None
        loss_confidence = 0.0
        if confidence_gt is not None:   
            if len(confidence_gt.shape) == 3:
                confidence_gt = confidence_gt.unsqueeze(1)

            # alpha = 10; beta = 1
            # weights = torch.where(confidence_gt == 0, alpha, beta)

            loss_confidence = self.mse_loss(pred_confidence, confidence_gt) # * weights
           
        else:
            raise ValueError('confidence GT must be given!')
                    
        loss = loss_confidence
        return SemanticSegmenterOutput(
            loss=loss,
            logits=pred_confidence
        )
    
    def interface(self, pixel_values):
        feature = self.dinov2.get_intermediate_layers(pixel_values, n=self.n, reshape = True, return_class_token = True, norm=False)
        pred_confidence = self.confidence_head(feature, None)
        
        return SemanticSegmenterOutput(
            loss=0.0,
            logits=pred_confidence
        ) 
    
if __name__ == "__main__":
    device = torch.device('cuda:0')
    model_size = 'base'
    img_size = 532
    model = RCMNet(backbone_size=model_size, r=4, lora_layer=None, image_shape=(img_size,img_size), \
                        decode_type = 'linear4', confidence_head='segformer').to(device)
    
    input = torch.rand((2, 3, img_size, img_size)).to(device)
    confidence_gt =  torch.rand((2, img_size, img_size)).to(device)
    output = model(input, confidence_gt)
    
    print(f'loss: {output.loss}')
    print(f'output shape: {output.logits.shape}')

