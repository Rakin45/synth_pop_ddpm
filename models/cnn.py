from typing import Union

from torch import Tensor, nn
import numpy as np


class Conv(nn.Module):
    def __init__(self, in_shape=(48, 8), hidden_layers=(64, 64), dropout=0.1):
        """Convolution based encoder and decoder with encoder embedding layer."""
        super().__init__()

        self.encoder = Encoder(
            in_shape=in_shape,
            hidden_layers=hidden_layers,
            dropout=dropout,
            kernel_size=3,
            stride=2,
            padding=1,
        )
        self.decoder = Decoder(
            target_shapes=self.encoder.target_shapes,
            hidden_layers=hidden_layers,
            kernel_size=3,
            stride=2,
            padding=1,
        )

    def forward(self, x: Tensor, target=None, **kwargs):
        """Forward pass, also return latent parameterization.

        Args:
            x (tensor): Input sequences [N, L, Cin].

        Returns:
            list[tensor]: [Log probs, Probs [N, L, Cout], Input [N, L, Cin], mu [N, latent], var [N, latent]].
        """
        z = self.encoder(x)
        y = self.decoder(z)
        return y


class Encoder(nn.Module):
    def __init__(
        self,
        in_shape: tuple,
        hidden_layers: list,
        dropout: float = 0.1,
        kernel_size: Union[tuple[int, int], int] = 3,
        stride: Union[tuple[int, int], int] = 2,
        padding: Union[tuple[int, int], int] = 1,
    ):
        """2d Convolutions Encoder.

        Args:
            in_shape (tuple[int, int, int]): [C, time_step, activity_encoding].
            hidden_layers (list, optional): _description_. Defaults to None.
            dropout (float): dropout. Defaults to 0.1.
            kernel_size (Union[tuple[int, int], int], optional): _description_. Defaults to 3.
            stride (Union[tuple[int, int], int], optional): _description_. Defaults to 2.
            padding (Union[tuple[int, int], int], optional): _description_. Defaults to 1.
        """
        super(Encoder, self).__init__()
        h, w = in_shape
        channels = 1

        modules = []
        self.target_shapes = [(channels, h, w)]

        for hidden_channels in hidden_layers:
            modules.append(
                nn.Sequential(
                    nn.Conv2d(
                        in_channels=channels,
                        out_channels=hidden_channels,
                        kernel_size=kernel_size,
                        stride=stride,
                        padding=padding,
                        bias=True,
                    ),
                    nn.BatchNorm2d(hidden_channels),
                    nn.LeakyReLU(),
                )
            )
            h, w = conv_size(
                (h, w), kernel_size=kernel_size, padding=padding, stride=stride
            )
            self.target_shapes.append((hidden_channels, h, w))
            channels = hidden_channels

        self.dropout = nn.Dropout(dropout)

        self.z_shape = (-1, channels, h, w)
        self.encoder = nn.Sequential(*modules)

    def forward(self, x):
        y = x.unsqueeze(1)  # add channel dim for Conv
        y = self.dropout(self.encoder(y))
        return y


class Decoder(nn.Module):
    def __init__(
        self,
        target_shapes,
        hidden_layers: list,
        kernel_size: Union[tuple[int, int], int] = 3,
        stride: Union[tuple[int, int], int] = 2,
        padding: Union[tuple[int, int], int] = 1,
    ):
        """2d Conv Decoder.

        Args:
            target_shapes (list): list of target shapes from encoder.
            hidden_layers (list, optional): _description_. Defaults to None.
            kernel_size (Union[tuple[int, int], int], optional): _description_. Defaults to 3.
            stride (Union[tuple[int, int], int], optional): _description_. Defaults to 2.
            padding (Union[tuple[int, int], int], optional): _description_. Defaults to 1.
        """
        super(Decoder, self).__init__()
        modules = []
        target_shapes.reverse()

        for i in range(len(hidden_layers) - 1):
            modules.append(
                nn.Sequential(
                    nn.ConvTranspose2d(
                        in_channels=target_shapes[i][0],
                        out_channels=target_shapes[i + 1][0],
                        kernel_size=kernel_size,
                        stride=stride,
                        padding=padding,
                        output_padding=calc_output_padding(target_shapes[i + 1]),
                        bias=True,
                    ),
                    nn.BatchNorm2d(target_shapes[i + 1][0]),
                    nn.LeakyReLU(),
                )
            )

        # Final layer with Tanh activation
        modules.append(
            nn.Sequential(
                nn.ConvTranspose2d(
                    in_channels=target_shapes[-2][0],
                    out_channels=target_shapes[-1][0],
                    kernel_size=kernel_size,
                    stride=stride,
                    padding=padding,
                    output_padding=calc_output_padding(target_shapes[-1]),
                ),
                nn.BatchNorm2d(target_shapes[-1][0]),
                nn.Tanh(),
            )
        )

        self.decoder = nn.Sequential(*modules)
        self.prob_activation = nn.Softmax(dim=-1)

    def forward(self, hidden, **kwargs):
        y = self.decoder(hidden)
        y = y.squeeze(1)  # remove conv channel dim
        return self.prob_activation(y)


def conv_size(
    size: Union[tuple[int, int], int],
    kernel_size: Union[tuple[int, int], int] = 3,
    stride: Union[tuple[int, int], int] = 2,
    padding: Union[tuple[int, int], int] = 1,
    dilation: Union[tuple[int, int], int] = 1,
) -> np.ndarray:
    """Calculate output dimensions for 2d convolution.

    Args:
        size (Union[tuple[int, int], int]): Input size, may be integer if symetric.
        kernel_size (Union[tuple[int, int], int], optional): Kernel_size. Defaults to 3.
        stride (Union[tuple[int, int], int], optional): Stride. Defaults to 2.
        padding (Union[tuple[int, int], int], optional): Input padding. Defaults to 1.
        dilation (Union[tuple[int, int], int], optional): Dilation. Defaults to 1.

    Returns:
        np.array: Output size.
    """
    if isinstance(size, int):
        size = (size, size)
    if isinstance(kernel_size, int):
        kernel_size = (kernel_size, kernel_size)
    if isinstance(stride, int):
        stride = (stride, stride)
    if isinstance(padding, int):
        padding = (padding, padding)
    if isinstance(dilation, int):
        dilation = (dilation, dilation)
    return (
        np.array(size)
        + 2 * np.array(padding)
        - np.array(dilation) * (np.array(kernel_size) - 1)
        - 1
    ) // np.array(stride) + 1


def calc_output_padding(size: Union[tuple[int, int, int], int]) -> np.array:
    """Calculate output padding for a transposed convolution such that output dims will
    match dimensions of inputs to a convolution of given size.
    For each dimension, padding is set to 1 if even size, otherwise 0.

    Args:
        size (Union[tuple[int, int, int], int]): input size (h, w)

    Returns:
        np.array: required padding
    """
    if isinstance(size, int):
        size = (0, size, size)
    _, h, w = size
    return (int(h % 2 == 0), int(w % 2 == 0))
