PATH = '../Vision Transformer/vit.pth'

import torch

"""# Vision Transformer"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchsummary


class PatchEmbedding(nn.Module): # Done
    """
    img_size: 1d size of each image (32 for CIFAR-10)
    patch_size: 1d size of each patch (img_size/num_patch_1d, 4 in this experiment)
    in_chans: input channel (3 for RGB images)
    emb_dim: flattened length for each token (or patch)
    """
    def __init__(self, img_size:int, patch_size:int, in_chans:int=3, emb_dim:int=48):
        super(PatchEmbedding, self).__init__()
        self.img_size = img_size
        self.patch_size = patch_size

        self.proj = nn.Conv2d(
            in_chans,
            emb_dim,
            kernel_size = patch_size,
            stride = patch_size
        )

    def forward(self, x):
        with torch.no_grad():
            # x: [batch, in_chans, img_size, img_size]
            x = self.proj(x) # [batch, embed_dim, # of patches in a row, # of patches in a col], [batch, 48, 8, 8] in this experiment
            x = x.flatten(2) # [batch, embed_dim, total # of patches], [batch, 48, 64] in this experiment
            x = x.transpose(1, 2) # [batch, total # of patches, emb_dim] => Transformer encoder requires this dimensions [batch, number of words, word_emb_dim]
        return x


class TransformerEncoder(nn.Module): # Done
    def __init__(self, input_dim:int, mlp_hidden_dim:int, num_head:int=8, dropout:float=0.):
        # input_dim and head for Multi-Head Attention
        super(TransformerEncoder, self).__init__()
        self.norm1 = nn.LayerNorm(input_dim) # LayerNorm is BatchNorm for NLP
        self.msa = MultiHeadSelfAttention(input_dim, n_heads=num_head)
        self.norm2 = nn.LayerNorm(input_dim)
        # Position-wise Feed-Forward Networks with GELU activation functions
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, mlp_hidden_dim),
            nn.GELU(),
            nn.Linear(mlp_hidden_dim, input_dim),
            nn.GELU(),
        )

    def forward(self, x):
        out = self.msa(self.norm1(x)) + x # add residual connection
        out = self.mlp(self.norm2(out)) + out # add another residual connection
        return out


class MultiHeadSelfAttention(nn.Module):
    """
    dim: dimension of input and out per token features (emb dim for tokens)
    n_heads: number of heads
    qkv_bias: whether to have bias in qkv linear layers
    attn_p: dropout probability for attention
    proj_p: droupout probability last linear layer
    scale: scaling factor for attention (1/sqrt(dk))
    qkv: initial linear layer for the query, key, and value
    proj: last linear layer
    attn_drop, proj_drop: dropout layers for attn and proj
    """
    def __init__(self, dim:int, n_heads:int=8, qkv_bias:bool=True, attn_p:float=0.01, proj_p:float=0.01):
        super(MultiHeadSelfAttention, self).__init__()
        self.n_heads = n_heads
        self.dim = dim # embedding dimension for input
        self.head_dim = dim // n_heads # d_q, d_k, d_v in the paper (int div needed to preserve input dim = output dim)
        self.scale = self.head_dim ** -0.5 # 1/sqrt(d_k)

        self.qkv = nn.Linear(dim, dim*3, bias=qkv_bias) # lower linear layers in Figure 2 of the paper
        self.attn_drop = nn.Dropout(attn_p)
        self.proj = nn.Linear(dim, dim) # upper linear layers in Figure 2 of the paper
        self.proj_drop = nn.Dropout(proj_p)

    def forward(self, x):
        """
        Input and Output shape: [batch_size, n_patches + 1, dim]
        """
        batch_size, n_tokens, x_dim = x.shape # n_tokens = n_patches + 1 (1 is cls_token), x_dim is input dim

        # Sanity Check
        if x_dim != self.dim: # make sure input dim is same as concatnated dim (output dim)
            raise ValueError
        if self.dim != self.head_dim*self.n_heads: # make sure dim is divisible by n_heads
            raise ValueError(f"Input & Output dim should be divisible by Number of Heads")

        # Linear Layers for Query, Key, Value
        qkv = self.qkv(x) # (batch_size, n_patches+1, 3*dim)
        qkv = qkv.reshape(batch_size, n_tokens, 3, self.n_heads, self.head_dim) # (batch_size, n_patches+1, 3, n_heads, head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4) # (3, batch_size, n_heads, n_patches+1, head_dim)
        q, k, v = qkv[0], qkv[1], qkv[2] # (batch_size, n_heads, n_patches+1, head_dim)

        # Scaled Dot-Product Attention
        k_t = k.transpose(-2, -1) # K Transpose: (batch_size, n_heads, head_dim, n_patches+1)
        dot_product = (q @ k_t)*self.scale # Query, Key Dot Product with Scale Factor: (batch_size, n_heads, n_patches+1, n_patches+1)
        attn = dot_product.softmax(dim=-1) # Softmax: (batch_size, n_heads, n_patches+1, n_patches+1)
        attn = self.attn_drop(attn) # Attention Dropout: (batch_size, n_heads, n_patches+1, n_patches+1)
        weighted_avg = attn @ v # (batch_size, n_heads, n_patches+1, head_dim)
        weighted_avg = weighted_avg.transpose(1, 2) # (batch_size, n_patches+1, n_heads, head_dim)

        # Concat and Last Linear Layer
        weighted_avg = weighted_avg.flatten(2) # Concat: (batch_size, n_patches+1, dim)
        x = self.proj(weighted_avg) # Last Linear Layer: (batch_size, n_patches+1, dim)
        x = self.proj_drop(x) # Last Linear Layer Dropout: (batch_size, n_patches+1, dim)

        return x

import torch
import torch.nn as nn
import torchsummary

class ViT(nn.Module): # Done
    def __init__(self, in_c:int=3, num_classes:int=10, img_size:int=32, num_patch_1d:int=8, dropout:float=0., num_enc_layers:int=7, hidden_dim:int=384, mlp_hidden_dim:int=384*4, num_head:int=8, is_cls_token:bool=True):
        super(ViT, self).__init__()
        """
        is_cls_token: are we using class token?
        num_patch_1d: number of patches in one row (or col), 3 in Figure 1 of the paper, 8 in this experiment
        patch_size: # 1d size (size of row or col) of each patch, 16 for ImageNet in the paper, 4 in this experiment
        flattened_patch_dim: Flattened vec length for each patch (4 x 4 x 3, each side is 4 and 3 color scheme), 48 in this experiment
        num_tokens: number of total patches + 1 (class token), 10 in Figure 1 of the paper, 65 in this experiment
        """
        self.is_cls_token = is_cls_token
        self.num_patch_1d = num_patch_1d
        self.patch_size = img_size//self.num_patch_1d
        flattened_patch_dim = (img_size//self.num_patch_1d)**2*3
        num_tokens = (self.num_patch_1d**2)+1 if self.is_cls_token else (self.num_patch_1d**2)

        # Divide each image into patches
        self.images_to_patches = PatchEmbedding(
                                    img_size=img_size,
                                    patch_size=img_size//num_patch_1d
                                )

        # Linear Projection of Flattened Patches
        self.lpfp = nn.Linear(flattened_patch_dim, hidden_dim) # 48 x 384 (384 is the latent vector size D in the paper)

        # Patch + Position Embedding (Learnable)
        self.cls_token = nn.Parameter(torch.randn(1, 1, hidden_dim)) if is_cls_token else None # learnable classification token with dim [1, 1, 384]. 1 in 2nd dim because there is only one class per each image not each patch
        self.pos_emb = nn.Parameter(torch.randn(1, num_tokens, hidden_dim)) # learnable positional embedding with dim [1, 65, 384]

        # Transformer Encoder
        enc_list = [TransformerEncoder(hidden_dim, mlp_hidden_dim=mlp_hidden_dim, dropout=dropout, num_head=num_head) for _ in range(num_enc_layers)] # num_enc_layers is L in Transformer Encoder at Figure 1
        self.enc = nn.Sequential(*enc_list) # * should be adeed if given regular python list to nn.Sequential

        # MLP Head (Standard Classifier)
        self.mlp_head = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, num_classes)
        )

    def forward(self, x): # x: [batch, 3, 32, 32]
        # Images into Patches (including flattening)
        out = self.images_to_patches(x) # [batch, 64, 48]

        # Linear Projection on Flattened Patches
        out = self.lpfp(out) # [batch, 64, 384]

        # Add Class Token and Positional Embedding
        if self.is_cls_token:
            out = torch.cat([self.cls_token.repeat(out.size(0),1,1), out], dim=1) # [batch, 65, 384], added as extra learnable embedding
        out = out + self.pos_emb # [batch, 65, 384]

        # Transformer Encoder
        out = self.enc(out) # [batch, 65, 384]
        if self.is_cls_token:
            out = out[:,0] # [batch, 384]
        else:
            out = out.mean(1)

        # MLP Head
        out = self.mlp_head(out) # [batch, 10]
        return out

"""# Load state dict using .pth file"""

state_dict = torch.load(PATH, map_location=torch.device('cpu'))

for key in list(state_dict.keys()):
    state_dict[key.replace('model.', '')] = state_dict.pop(key)

model = ViT(
          3,
          10,
          32,
          8,
          0.0,
          7,
          384,
          384,
          12,
          True
          )
model.load_state_dict(state_dict)
model.eval()

"""# Get the Test Loader"""

import torchvision
from torchvision import transforms
from torch.utils.data.dataloader import DataLoader

transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
        ])

batch_size = 1024

testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                       download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                         shuffle=False, num_workers=0)

classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

"""# Test the model"""

def evaluate(model, val_loader):
    outputs = [model.validation_step(batch) for batch in val_loader]
    return model.validation_epoch_end(outputs)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Move model to the appropriate device
model.to(device)

correct = 0
total = 0

# Since we're not training, we don't need to calculate the gradients for our outputs
with torch.no_grad():
    for data in testloader:
        images, labels = data
        # Move tensors to the appropriate device
        images, labels = images.to(device), labels.to(device)
        # Calculate outputs by running images through the network
        outputs = model(images)
        # The class with the highest energy is what we choose as prediction
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print(f'Accuracy of the network on the 10000 test images: {round(100 * correct / total, 2)} %')

# Prepare to count predictions for each class
correct_pred = {classname: 0 for classname in classes}
total_pred = {classname: 0 for classname in classes}

# Again, no gradients needed
with torch.no_grad():
    for data in testloader:
        images, labels = data
        # Move tensors to the appropriate device
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        _, predictions = torch.max(outputs, 1)
        # Collect the correct predictions for each class
        for label, prediction in zip(labels, predictions):
            if label == prediction:
                correct_pred[classes[label.item()]] += 1
            total_pred[classes[label.item()]] += 1

# Print accuracy for each class
for classname, correct_count in correct_pred.items():
    accuracy = 100 * float(correct_count) / total_pred[classname]
    print(f'Accuracy for class: {classname:5s} is {accuracy:.1f} %')







