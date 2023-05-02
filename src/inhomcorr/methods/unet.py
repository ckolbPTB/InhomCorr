"""Implementation of a UNet."""

from dataclasses import dataclass

import torch
import torch.nn as nn

from inhomcorr.interfaces.bias_estimator_interface import HyperParameters


@dataclass
class UNetHyperParameters(HyperParameters):
    """Hyperparameters for the UNet Estimator."""

    dim: torch.int = 2
    nChIn: torch.int = 2
    nChOut: torch.int = 2
    ActivationFun: torch.nn = torch.nn.LeakyReLU()
    KernelSize: torch.int = 3
    nEncStages: torch.int = 3
    nConvsPerStage: torch.int = 2
    nFilters: torch.int = 16
    ConvBlockResConnection: torch.bool = True
    Bias: torch.bool = True


class ConvBlock(nn.Module):
    """A block of convolutional layers (1D, 2D or 3D)."""

    def __init__(
            self,
            dim,
            n_ch_in,
            n_ch_out,
            n_convs,
            activate_fun,
            conv_block_res_connection,
            kernel_size=3,
            bias=True):
        super().__init__()

        self.conv_block_res_connection = conv_block_res_connection

        if dim == 1:
            conv_op = nn.Conv1d
        if dim == 2:
            conv_op = nn.Conv2d
        elif dim == 3:
            conv_op = nn.Conv3d

        padding = 'same'

        conv_block_list = []
        conv_block_list.extend([conv_op(n_ch_in,
                                        n_ch_out,
                                        kernel_size,
                                        padding=padding,
                                        bias=bias),
                                activate_fun])

        for i in range(n_convs - 1):
            conv_block_list.extend([conv_op(
                n_ch_out, n_ch_out, kernel_size, padding=padding, bias=bias),
                activate_fun])

        self.conv_block = nn.Sequential(*conv_block_list)

        if conv_block_res_connection:
            self.reslayer = conv_op(
                n_ch_in, n_ch_out, kernel_size, padding=padding, bias=bias)

    def forward(self, x):
        """Forward pass of the convolutional block.

        Parameters
        ----------
        x
            _description_

        Returns
        -------
            _description_
        """
        if self.conv_block_res_connection:
            return self.conv_block(x) + self.reslayer(x)
        else:
            return self.conv_block(x)


class Encoder(nn.Module):
    """Encoder of the UNet."""

    def __init__(
            self,
            dim,
            n_ch_in,
            n_enc_stages,
            n_convs_per_stage,
            n_filters,
            activation_fun,
            conv_block_res_connection,
            kernel_size=3,
            bias=True):
        super().__init__()

        n_ch_list = [n_ch_in]
        for ne in range(n_enc_stages):
            n_ch_list.append(int(n_filters) * 2**ne)

        self.enc_blocks = nn.ModuleList([ConvBlock(dim,
                                                   n_ch_list[i],
                                                   n_ch_list[i + 1],
                                                   n_convs_per_stage,
                                                   activation_fun,
                                                   conv_block_res_connection,
                                                   kernel_size=kernel_size)
                                         for i in range(len(n_ch_list) - 1)])

        if dim == 1:
            pool_op = nn.MaxPool1d(2)
        elif dim == 2:
            pool_op = nn.MaxPool2d(2)
        elif dim == 3:
            pool_op = nn.MaxPool3d(2)

        self.pool = pool_op

    def forward(self, x):
        """Forward pass of the encoder.

        Parameters
        ----------
        x
            _description_

        Returns
        -------
            _description_
        """
        features = []
        for block in self.enc_blocks:
            x = block(x)
            features.append(x)
            x = self.pool(x)
        return features


class Decoder(nn.Module):
    """Decoder of the UNet."""

    def __init__(
            self,
            dim,
            n_ch_in,
            n_dec_stages,
            n_convs_per_stage,
            n_filters,
            activation_fun,
            conv_block_res_connection,
            kernel_size=3,
            bias=False):
        super(Decoder, self).__init__()

        n_ch_list = []
        for ne in range(n_dec_stages):
            n_ch_list.append(int(n_ch_in * (1 / 2)**ne))

        if dim == 1:
            interp_mode = 'linear'
            conv_op = nn.Conv1d
        elif dim == 2:
            conv_op = nn.Conv2d
            interp_mode = 'bilinear'
        elif dim == 3:
            interp_mode = 'trilinear'
            conv_op = nn.Conv3d

        self.interp_mode = interp_mode

        padding = 'same'
        self.upconvs = nn.ModuleList([conv_op(n_ch_list[i],
                                              n_ch_list[i + 1],
                                              kernel_size=kernel_size,
                                              padding=padding,
                                              bias=bias)
                                      for i in range(len(n_ch_list) - 1)])
        self.dec_blocks = nn.ModuleList([ConvBlock(dim,
                                                   n_ch_list[i],
                                                   n_ch_list[i + 1],
                                                   n_convs_per_stage,
                                                   activation_fun,
                                                   conv_block_res_connection,
                                                   kernel_size=kernel_size,
                                                   bias=bias)
                                         for i in range(len(n_ch_list) - 1)])

    def forward(self, x, encoder_features):
        """Forward pass of the decoder."""
        for i in range(len(self.dec_blocks)):
            # x        = self.upconvs[i](x)
            enc_features = encoder_features[i]
            enc_features_shape = enc_features.shape
            x = nn.functional.interpolate(
                x, enc_features_shape[2:], mode=self.interp_mode)
            x = self.upconvs[i](x)
            x = torch.cat([x, enc_features], dim=1)
            x = self.dec_blocks[i](x)
        return x


class UNet(nn.Module):
    """UNet model."""

    def __init__(self, hparams: UNetHyperParameters):
        super(UNet, self).__init__()

        self.encoder = Encoder(hparams.dim,
                               hparams.nChIn,
                               hparams.nEncStages,
                               hparams.nConvsPerStage,
                               hparams.nFilters,
                               hparams.ActivationFun,
                               hparams.ConvBlockResConnection,
                               hparams.KernelSize,
                               hparams.Bias)
        self.decoder = Decoder(hparams.dim,
                               hparams.nFilters *
                               (2**(hparams.nEncStages - 1)),
                               hparams.nEncStages,
                               hparams.nConvsPerStage,
                               hparams.nFilters * (hparams.nEncStages * 2),
                               hparams.ActivationFun,
                               hparams.ConvBlockResConnection,
                               kernel_size=hparams.KernelSize,
                               bias=hparams.Bias)

        if hparams.dim == 1:
            conv_op = nn.Conv1d
        elif hparams.dim == 2:
            conv_op = nn.Conv2d
        elif hparams.dim == 3:
            conv_op = nn.Conv3d

        self.c1x1 = conv_op(hparams.nFilters,
                            hparams.nChOut,
                            kernel_size=1, padding=0, bias=hparams.Bias)

    def forward(self, x):
        """Forward pass of the UNet.

        Parameters
        ----------
        x
            _description_

        Returns
        -------
            _description_
        """
        enc_features = self.encoder(x)
        x = self.decoder(enc_features[-1], enc_features[::-1][1:])
        x = self.c1x1(x)
        return x
