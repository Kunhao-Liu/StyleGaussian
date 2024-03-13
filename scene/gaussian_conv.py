import torch
import torch_scatter
from sklearn.neighbors import NearestNeighbors
from fast_pytorch_kmeans import KMeans

class GaussianConv(torch.nn.Module):
    def __init__(self, xyz, input_channel=256, layers_channel=[256, 128, 64, 32, 3], downsample_layer=[], upsample_layer=[], K=8):
        super(GaussianConv, self, ).__init__()
        assert len(downsample_layer) == len(upsample_layer) == 0 or \
            (len(downsample_layer) == len(upsample_layer) and max(downsample_layer) < min(upsample_layer)) ,\
            'downsample_layer and upsample_layer must be the same length and satisfy max(downsample_layer) < min(upsample_layer) or both are empty lists'
        
        self.K = K
        self.N = xyz.shape[0]
        self.downsample_layer = downsample_layer
        self.upsample_layer = upsample_layer

        self.init_kmeans_knn(xyz, len(downsample_layer))
        self.init_conv_params(input_channel, layers_channel)

    @torch.no_grad()
    def init_kmeans_knn(self, xyz, len_sample_layer):
        self.knn_indices = []
        self.kmeans_labels = []

        # get original knn_indices
        xyz_numpy = xyz.cpu().numpy()
        nn = NearestNeighbors(n_neighbors=self.K, algorithm='auto')
        nn.fit(xyz_numpy)
        _, knn_indices = nn.kneighbors(xyz_numpy) # [N, K]
        self.knn_indices.append(knn_indices) 

        last_N = self.N
        last_xyz = xyz

        for i in range(len_sample_layer):
            print('Using KMeans to cluster point clouds in level', i)
            kmeans = KMeans(n_clusters=last_N//self.K, mode='euclidean', verbose=1)
            self.kmeans_labels.append(kmeans.fit_predict(last_xyz)) # [N]
            down_centroids = torch_scatter.scatter(last_xyz, self.kmeans_labels[-1], dim=0, reduce='mean') # [cluster_num=N//5, D]

            # get knn_indices for downsampled point clouds
            nn = NearestNeighbors(n_neighbors=self.K, algorithm='auto')
            nn.fit(down_centroids.cpu().numpy())
            _, knn_indices = nn.kneighbors(down_centroids.cpu().numpy())
            self.knn_indices.append(knn_indices)

            last_N = down_centroids.shape[0]
            last_xyz = down_centroids

    def init_conv_params(self, input_channel, layers_channel):
        self.kernels = []
        self.bias = []
        for out_channel in layers_channel:
            self.kernels.append(torch.randn(out_channel, self.K*input_channel)*0.1)  # [out_channel, K*input_channel]
            self.bias.append(torch.zeros(1, out_channel))  # [1, out_channel]
            input_channel = out_channel

        self.kernels = torch.nn.ParameterList(self.kernels)
        self.bias = torch.nn.ParameterList(self.bias)

    def forward(self, features):
        '''
        Args:
            features: [N, D]
            D: input_channel
            S: output_channel
        '''
        sample_level = 0
        for i in range(len(self.kernels)):
            if i in self.downsample_layer:
                sample_level += 1
                features = torch_scatter.scatter(features, self.kmeans_labels[sample_level-1], dim=0, reduce='mean')
            elif i in self.upsample_layer:
                sample_level -= 1
                features = features[self.kmeans_labels[sample_level]]

            knn_indices = self.knn_indices[sample_level]

            knn_features = features[knn_indices] # [N, K, D]
            knn_features = knn_features.reshape(knn_features.size(0), -1) # [N, K*D]
            features = knn_features @ self.kernels[i].T + self.bias[i] # [N, S]
            features = torch.sigmoid(features) if i != len(self.kernels)-1 else features

        return features # [N, S]