import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable


class STNKd(nn.Module):
    # T-Net a.k.a. Spatial Transformer Network
    def __init__(self, k: int):
        super().__init__()
        self.k = k
        self.conv1 = nn.Sequential(nn.Conv1d(k, 64, 1), nn.BatchNorm1d(64))
        self.conv2 = nn.Sequential(nn.Conv1d(64, 128, 1), nn.BatchNorm1d(128))
        self.conv3 = nn.Sequential(nn.Conv1d(128, 1024, 1), nn.BatchNorm1d(1024))

        self.fc = nn.Sequential(
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Linear(256, k * k),
        )

    def forward(self, x):
        """
        Input: [B,k,N]
        Output: [B,k,k]
        """
        B = x.shape[0]
        device = x.device
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = torch.max(x, 2)[0]

        x = self.fc(x)
        
        # Followed the original implementation to initialize a matrix as I.
        identity = (
            Variable(torch.eye(self.k, dtype=torch.float))
            .reshape(1, self.k * self.k)
            .expand(B, -1)
            .to(device)
        )
        x = x + identity
        x = x.reshape(-1, self.k, self.k)
        return x 
        # return 되는 x가 입력으로 부터 정렬되서 나오는 출력이 아니라, 정렬 행렬 그 자체이다.
        # x를 입력할시 네트워크들은 그 x로부터 어떻게 정렬할지를 출력한다. 그 정렬 행렬은 추후에 입력과 곱해진다.


class PointNetFeat(nn.Module):
    """
    Corresponds to the part that extracts max-pooled features.
    """
    def __init__(
        self,
        input_transform: bool = False,
        feature_transform: bool = False,
    ):
        super().__init__()
        self.input_transform = input_transform
        self.feature_transform = feature_transform

        if self.input_transform:
            self.stn3 = STNKd(k=3)
        if self.feature_transform:
            self.stn64 = STNKd(k=64)

        # point-wise mlp
        # TODO : Implement point-wise mlp model based on PointNet Architecture.

        self.conv1 = nn.Sequential(
            nn.Conv1d(3, 64, 1),
            nn.BatchNorm1d(64),
            nn.ReLU()
        )
        self.conv2 = nn.Sequential(
            nn.Conv1d(64, 128, 1),
            nn.BatchNorm1d(128),
            nn.ReLU()
        )
        self.conv3 = nn.Sequential(
            nn.Conv1d(128, 1024, 1),
            nn.BatchNorm1d(1024),
            nn.ReLU()
        )

    def forward(self, pointcloud):
        """
        Input:
            - pointcloud: [B,N,3]
        Output:
            - Global feature: [B,1024]
            - ...
        """

        # TODO : Implement forward function.
        B, N, _ = pointcloud.shape
        x = pointcloud.transpose(1, 2) # [B, 3, N]. bmm 계산식에 맞게 전치해준다.

        if self.input_transform:
            T = self.stn3(x)           # [B, 3, 3]
            x = torch.bmm(T, x)        # [B, 3, N]
        else:
            T = None

        x = self.conv1(x)

        if self.feature_transform:
            T_feat = self.stn64(x)
            x = torch.bmm(T_feat, x)
            pointwise_feat = x
        else:
            pointwise_feat = x

        x = self.conv2(x)              # [B, 128, N]
        x = self.conv3(x)              # [B, 1024, N]

        x = torch.max(x, 2)[0]         # [B, 1024] max-pooling

        return x, pointwise_feat
        


class PointNetCls(nn.Module):
    def __init__(self, num_classes, input_transform, feature_transform):
        super().__init__()
        self.num_classes = num_classes
        
        # extracts max-pooled features
        self.pointnet_feat = PointNetFeat(input_transform, feature_transform)
        
        # returns the final logits from the max-pooled features.
        # TODO : Implement MLP that takes global feature as an input and return logits.
        self.fc_layers = nn.Sequential(
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),

            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),

            nn.Linear(256, num_classes)
        )

    def forward(self, pointcloud):
        """
        Input:
            - pointcloud [B,N,3]
        Output:
            - logits [B,num_classes]
            - ...
        """
        # TODO : Implement forward function.
        global_feat, T64 = self.pointnet_feat(pointcloud)  # global_feat : [B, 1024]

        logits = self.fc_layers(global_feat)   # [B, num_classes]

        return logits


class PointNetPartSeg(nn.Module):
    def __init__(self, m=50):
        super().__init__()

        # returns the logits for m part labels each point (m = # of parts = 50).
        # TODO: Implement part segmentation model based on PointNet Architecture.
        self.pointnet_feat = PointNetFeat(input_transform=True, feature_transform=True)

        self.conv1 = nn.Sequential(
            nn.Conv1d(1088, 512, 1),  # 64 (pointwise) + 1024 (global) = 1088
            nn.BatchNorm1d(512),
            nn.ReLU()
        )
        self.conv2 = nn.Sequential(
            nn.Conv1d(512, 256, 1),
            nn.BatchNorm1d(256),
            nn.ReLU()
        )
        self.conv3 = nn.Sequential(
            nn.Conv1d(256, 128, 1),
            nn.BatchNorm1d(128),
            nn.ReLU()
        )
        self.conv4 = nn.Conv1d(128, m, 1)  # Output logits [B, m, N]

    def forward(self, pointcloud):
        """
        Input:
            - pointcloud: [B,N,3]
        Output:
            - logits: [B,50,N] | 50: # of point labels
            - ...
        """
        # TODO: Implement forward function.
        B, N, _ = pointcloud.shape
        global_feat, pointwise_feat = self.pointnet_feat(pointcloud)  

        global_feat_expand = global_feat.unsqueeze(-1).repeat(1, 1, N)  # [B, 1024, N]
        concat_feat = torch.cat([pointwise_feat, global_feat_expand], dim=1)  # [B, 1088, N]

        x = self.conv1(concat_feat)
        x = self.conv2(x)
        x = self.conv3(x)
        logits = self.conv4(x)  # [B, m, N]

        return logits


class PointNetAutoEncoder(nn.Module):
    def __init__(self, num_points):
        super().__init__()
        self.pointnet_feat = PointNetFeat()

        # Decoder is just a simple MLP that outputs N x 3 (x,y,z) coordinates.
        # TODO : Implement decoder.

        self.num_points = num_points

        self.decoder = nn.Sequential(
            nn.Linear(1024, num_points // 4),
            nn.BatchNorm1d(num_points // 4),
            nn.ReLU(),

            nn.Linear(num_points // 4, num_points // 2),
            nn.BatchNorm1d(num_points // 2),
            nn.ReLU(),

            nn.Linear(num_points // 2, num_points),
            nn.BatchNorm1d(num_points),
            nn.Dropout(p=0.3),
            nn.ReLU(),

            nn.Linear(num_points, num_points * 3),
        )

    def forward(self, pointcloud):
        """
        Input:
            - pointcloud [B,N,3]
        Output:
            - pointcloud [B,N,3]
            - ...
        """
        # TODO : Implement forward function.

        global_feat, _ = self.pointnet_feat(pointcloud)

        x = self.decoder(global_feat)
        x = x.view(-1, self.num_points, 3)

        return x


def get_orthogonal_loss(feat_trans, reg_weight=1e-3):
    """
    a regularization loss that enforces a transformation matrix to be a rotation matrix.
    Property of rotation matrix A: A*A^T = I
    """
    if feat_trans is None:
        return 0

    B, K = feat_trans.shape[:2]
    device = feat_trans.device

    identity = torch.eye(K).to(device)[None].expand(B, -1, -1)
    mat_square = torch.bmm(feat_trans, feat_trans.transpose(1, 2))

    mat_diff = (identity - mat_square).reshape(B, -1)

    return reg_weight * mat_diff.norm(dim=1).mean()
