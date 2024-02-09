import math

import torch
import torch.nn as nn
import torch.nn.functional as F


# Define the NB class first, not mixture version

class GaussNorm_A(nn.Module):
    def __init__(self, c_in, c_out):
        super(GaussNorm_A, self).__init__()
        self.c_in = c_in
        self.c_out = c_out
        self.n_conv = nn.Conv2d(in_channels=64, out_channels=1, kernel_size=(1, 1), bias=True).to(device="cuda")
        self.p_conv = nn.Conv2d(in_channels=64, out_channels=1, kernel_size=(1, 1), bias=True).to(device="cuda")

        self.fully = nn.Linear(c_in, c_out, )
        self.out_dim = c_out  # output horizon

    # def forward(self, x):
    #     x = x.permute(0, 2, 1, 3)
    #     (B, _, N, _) = x.shape  # B: batch_size; N: input nodes
    #     loc = self.n_conv(x).squeeze_(-1)
    #     # The location (loc) keyword specifies the mean. The scale (scale) keyword specifies the standard deviation.
    #     scale = self.p_conv(x).squeeze_(-1)
    #
    #     # Reshape
    #     loc = loc.view([B, self.out_dim, N])
    #     scale = scale.view([B, self.out_dim, N])
    #
    #     # Ensure n is positive and p between 0 and 1
    #     loc = F.softplus(loc)  # Some parameters can be tuned here, count data are always positive
    #     scale = F.sigmoid(scale)
    #
    #     return loc.permute([0, 2, 1]), scale.permute([0, 2, 1])
    def forward(self, x):
        # x = x.permute(0, 2, 1, 3)
        # (B, _, N, _) = x.shape  # B: batch_size; N: input nodes
        # print(x.shape)
        x1 = x.reshape((x.shape[0], x.shape[1], -1))
        x1 = self.fully(x1).unsqueeze(2)

        # x2 = x.permute(0, 2, 1, 3)
        # x2=self.n_conv(x2).permute(0,2,1,3)
        # x2=F.softplus(x2)

        # x3 = x.permute(0, 2, 1, 3)
        # x3=self.p_conv(x3).permute(0,2,1,3)
        # x3=F.sigmoid(x3)

        # x2=x.reshape((x.shape[0], x.shape[1], -1))
        # x2=self.n_conv(x2).unsqueeze(2)

        return x1


# Define the Gaussian
class GaussNorm_B(nn.Module):
    def __init__(self, c_in, c_out):
        super(GaussNorm_B, self).__init__()
        self.c_in = c_in
        self.c_out = c_out
        self.n_conv = nn.Conv2d(in_channels=64, out_channels=1, kernel_size=(1, 1), bias=True).to(device="cuda")

        self.p_conv = nn.Conv2d(in_channels=64, out_channels=1, kernel_size=(1, 1), bias=True).to(device="cuda")

        self.out_dim = c_out  # output horizon

    def forward(self, x):
        x = x.permute(0, 2, 1, 3)
        # (B, _, N, _) = x.shape  # B: batch_size; N: input nodes
        # print(x.shape)
        loc = self.n_conv(x).permute(0, 2, 1, 3)  # .squeeze_(-1)
        scale = self.p_conv(x).permute(0, 2, 1, 3)  # .squeeze_(-1)
        loc = F.softplus(loc)
        scale = F.sigmoid(scale)

        # x=x.reshape((x.shape[0], x.shape[1], -1))
        # x=self.fully(x).unsqueeze(2)
        return loc, scale


class MDGCN(nn.Module):
    """
    Neural network block that applies a diffusion graph convolution to sampled location
    """

    def __init__(self, in_channels, out_channels, orders, activation="relu"):
        """
        :param in_channels: Number of time step.
        :param out_channels: Desired number of output features at each node in
        each time step.
        :param order: The diffusion steps.
        """
        super(MDGCN, self).__init__()
        self.orders = orders
        self.activation = activation
        self.num_matrices = 2 * self.orders + 1
        self.Theta1 = nn.Parameter(torch.FloatTensor(in_channels * self.num_matrices, out_channels))
        self.bias = nn.Parameter(torch.FloatTensor(out_channels))
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1.0 / math.sqrt(self.Theta1.shape[1])
        self.Theta1.data.uniform_(-stdv, stdv)
        stdv1 = 1.0 / math.sqrt(self.bias.shape[0])
        self.bias.data.uniform_(-stdv1, stdv1)

    def _concat(self, x, x_):
        x_ = x_.unsqueeze(0)
        return torch.cat([x, x_], dim=0)

    def forward(self, X, A_q, A_h):
        """
        :param X: Input data of shape (batch_size, num_nodes, num_timesteps)
        :A_q: The forward random walk matrix (num_nodes, num_nodes)
        :A_h: The backward random walk matrix (num_nodes, num_nodes)
        :return: Output data of shape (batch_size, num_nodes, num_features)
        """

        # X.shape 0batch_size, 1node, 2timestep, 3feature
        batch_size = X.shape[0]  # batch_size
        num_node = X.shape[1]
        input_size = X.size(2)  # time_length
        feature = X.shape[3]

        supports = [A_q, A_h]
        x0 = X.permute(3, 1, 2, 0)  # (num_nodes, num_times, batch_size, feature)

        x0 = torch.reshape(x0, shape=[num_node, feature * input_size * batch_size])
        x = torch.unsqueeze(x0, 0)

        for support in supports:
            # x1 = torch.mm(support, x0)
            x1 = torch.matmul(support, x0)
            x = self._concat(x, x1)
            for k in range(2, self.orders + 1):
                x2 = 2 * torch.matmul(support, x1) - x0
                x = self._concat(x, x2)
                x1, x0 = x2, x1

        x = torch.reshape(x, shape=[self.num_matrices, feature, num_node, input_size, batch_size])
        # x = torch.reshape(x, shape=[self.num_matrices, num_node, input_size, batch_size*feature])

        # x = x.permute(3, 1, 2, 0)  # (batch_size, num_nodes, input_size, order)

        x = x.permute(4, 2, 1, 3, 0)  # batch_size, num_nodes, feature, input_size, order

        x = torch.reshape(x, shape=[batch_size, num_node, feature, input_size * self.num_matrices])

        # x = x.permute(0,2,1,3)

        # print("bing",x.shape,self.Theta1.shape)
        # x = torch.matmul(x, self.Theta1)  # (batch_size * self._num_nodes, output_size)
        # print(x.shape)
        x = torch.matmul(x, self.Theta1)

        x += self.bias

        if self.activation == "relu":
            x = F.relu(x)
        elif self.activation == "selu":
            x = F.selu(x)

        # batch, node, feature, out_channels
        x = x.permute(0, 1, 3, 2)
        # print(x.shape)
        return x


class IATCN(nn.Module):
    def __init__(self, in_channels, spatial_channels, out_channels, node=491):
        super(IATCN, self).__init__()

        self.temporal1 = TCNN(in_channels=in_channels, out_channels=out_channels)

        self.Theta1 = nn.Parameter(torch.FloatTensor(out_channels, spatial_channels))

        self.temporal2 = TCNN(in_channels=spatial_channels, out_channels=out_channels)

        self.batch_norm = nn.BatchNorm2d(node)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.Theta1.shape[1])
        self.Theta1.data.uniform_(-stdv, stdv)

    def forward(self, X, A_hat):
        """
        :param X: Input data of shape (batch_size, num_nodes, num_timesteps,
        num_features=in_channels).
        :param A_hat: Normalized adjacency matrix.
        :return: Output data of shape (batch_size, num_nodes,
        num_timesteps_out, num_features=out_channels).
        """

        t = self.temporal1(X)
        lfs = torch.einsum("ij,jklm->kilm", [A_hat, t.permute(1, 0, 2, 3)])
        t2 = F.relu(torch.matmul(lfs, self.Theta1))
        t3 = self.temporal2(t2)
        x = self.batch_norm(t3)

        return x


class TCNN(nn.Module):
    """
    Neural network block that applies a bidirectional temporal convolution to each node of
    a graph.
    """

    def __init__(self, in_channels, out_channels, kernel_size=3, activation="relu", device="cuda", node=491):
        """
        :param in_channels: Number of nodes in the graph.
        :param out_channels: Desired number of output features.
        :param kernel_size: Size of the 1D temporal kernel.
        """

        super(TCNN, self).__init__()
        # forward dirction temporal convolution
        self.kernel_size = kernel_size
        self.out_channels = out_channels
        self.activation = activation
        self.device = device
        self.node = node

        self.conv1 = nn.Conv2d(in_channels, out_channels, (1, kernel_size))
        self.conv2 = nn.Conv2d(in_channels, out_channels, (1, kernel_size))
        self.conv3 = nn.Conv2d(in_channels, out_channels, (1, kernel_size))

        self.conv1b = nn.Conv2d(in_channels, out_channels, (1, kernel_size))
        self.conv2b = nn.Conv2d(in_channels, out_channels, (1, kernel_size))
        self.conv3b = nn.Conv2d(in_channels, out_channels, (1, kernel_size))

    def forward(self, X):
        """
        :param X: Input data of shape (batch_size,f num_timesteps, num_nodes)
        :return: Output data of shape (batch_size,f num_timesteps, num_features)
        """

        batch_size = X.shape[0]
        seq_len = X.shape[2]

        X = X.permute(0, 3, 1, 2)

        # Xf = X.unsqueeze(1)  # (batch_size, 5, num_timesteps, num_nodes)
        Xf = X

        inv_idx = (torch.arange(Xf.size(3) - 1, -1, -1).long().to(device=self.device))
        Xb = Xf.index_select(3, inv_idx)  # inverse the direction of time

        # Xf = Xf.permute(0, 3, 1, 2)
        # Xb = Xb.permute(0, 3, 1, 2)  # (batch_size, num_nodes, 1, num_timesteps)

        # Xf = Xf.permute(0, 1, 3, 2)
        # Xb = Xb.permute(0, 1, 3, 2)

        tempf = self.conv1(Xf) * torch.sigmoid(self.conv2(Xf))  # +
        outf = F.relu(tempf + self.conv3(Xf))

        tempb = self.conv1b(Xb) * torch.sigmoid(self.conv2b(Xb))  # +
        outb = tempb + self.conv3b(Xb)
        rec = torch.zeros([batch_size, self.out_channels, self.node, self.kernel_size - 1]).to(device=self.device)
        outf = torch.cat((outf, rec), dim=3)
        outb = torch.cat((outb, rec), dim=3)  # (batch_size, num_timesteps, out_features)
        inv_idx = (torch.arange(outb.size(3) - 1, -1, -1).long().to(device=self.device))
        outb = outb.index_select(3, inv_idx)
        out = outf + outb
        # print(out.shape)
        if self.activation == "relu":
            out = F.relu(outf) + F.relu(outb)
        elif self.activation == "sigmoid":
            out = F.sigmoid(outf) + F.sigmoid(outb)

        out = F.relu(out + self.conv3(X))
        out = out.permute(0, 2, 3, 1)

        return out


class UAHGNN(nn.Module):

    def __init__(self, SC1, SC2, SC3, TC1, TC2, TC3, SNB, TNB, node_dim, features, args, A_wave=None):
        super(UAHGNN, self).__init__()
        self.TC1 = TC1
        self.TC2 = TC2
        self.TC3 = TC3
        self.TNB = TNB
        #
        self.SC1 = SC1
        self.SC2 = SC2
        self.SC3 = SC3
        self.SNB = SNB
        self.A_wave = A_wave

        self.block1 = IATCN(in_channels=features, out_channels=64, spatial_channels=16)
        self.block2 = IATCN(in_channels=64, out_channels=64, spatial_channels=16)
        self.last_tcn = TCNN(in_channels=64, out_channels=64)

        # self.block1 = STGCNBlock(in_channels=features, out_channels=64,spatial_channels=16, num_nodes=node_dim)
        # self.block2 = STGCNBlock(in_channels=64, out_channels=64,spatial_channels=16, num_nodes=node_dim)
        # self.last_tcn = TimeBlock(in_channels=64, out_channels=64)

        self.fully = nn.Linear((args.num_timesteps_input - 2 * 5) * 64, features)

    def forward(self, X, A_q, A_h):
        """
        :param X: Input data of shape (batch_size, num_timesteps, num_nodes)
        :A_hat: The Laplacian matrix (num_nodes, num_nodes)
        :return: Reconstructed X of shape (batch_size, num_timesteps, num_nodes)
        """
        X = X.to(device='cuda')  # Dummy dimension deleted
        xs = None
        xt = None

        out1 = self.block1(X, self.A_wave)
        out2 = self.block2(out1, self.A_wave)
        out3 = self.last_tcn(out2)
        out4 = self.fully(out3.reshape((out3.shape[0], out3.shape[1], -1)))
        xt = out4.unsqueeze(2)

        # X_T = X.permute(0, 1, 2, 3)
        # X_t1 = self.TC1(X_T)
        # lfs = torch.einsum("ij,jklm->kilm", [self.A_wave, X_t1.permute(1, 0, 2, 3)])
        # t2 = F.relu(torch.matmul(lfs, self.Theta1))
        # t3 = self.TC2(t2)
        # xt=self.batch_norm(t3)
        # # X_t2 = self.TC2(X_t1)
        # # X_t3 = self.TC3(X_t2)
        # # xt = self.TNB(X_t3)
        # g=xt.reshape((xt.shape[0], xt.shape[1], -1))
        # xt=self.fully(g)
        # xt=xt.unsqueeze(2)

        # x=xt.permute(0, 2, 1, 3)
        # x=xt + xs
        # _b, _n, _hs, _feature = X_s3.shape
        # n_s_nb, p_s_nb, pi_s_nb = self.SNB(X_s3.view(_b, _n, _hs, _feature))
        # n_s_nb, p_s_nb, pi_s_nb = self.SNB(X_s3)
        # n_res = n_t_nb.permute(0, 2, 1,3) * n_s_nb
        # p_res = p_t_nb.permute(0, 2, 1,3) * p_s_nb
        # pi_res = pi_t_nb.permute(0, 2, 1,3) * pi_s_nb

        # n_res = n_s_nb
        # p_res = p_s_nb
        # pi_res = pi_s_nb
        # n_res = n_t_nb.permute(0, 2, 1,3)
        # p_res = p_t_nb.permute(0, 2, 1,3)
        # pi_res = pi_t_nb.permute(0, 2, 1,3)
        # return n_res, p_res, pi_res

        X_s1 = self.SC1(X, A_q, A_h)
        X_s2 = self.SC2(X_s1, A_q, A_h)
        X_s3 = self.SC3(X_s2, A_q, A_h)
        xs = self.SNB(X_s3)

        x = xs + xt
        return x
