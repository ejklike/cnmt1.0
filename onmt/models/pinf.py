""" Onmt Inference Model base class definition """
import torch.nn as nn
import torch.nn.functional as F

from onmt.utils.misc import masked_average


class PInfModel(nn.Module):

    def __init__(self, encoder, W_inf):
        super(PInfModel, self).__init__()
        self.encoder = encoder
        self.W_inf = W_inf

    def forward(self, src, src_lengths):
        """Forward propagate a `src` and `tgt` pair for training.
        Possible initialized with a beginning decoder state.

        Args:
            src (Tensor): A source sequence passed to encoder.
                typically for inputs this will be a padded `LongTensor`
                of size ``(len, batch, features)``. However, may be an
                image or other generic input depending on encoder.
            lengths(LongTensor): The src lengths, pre-padding ``(batch,)``.
            
        Returns:
            (FloatTensor, dict[str, FloatTensor]):

            * decoder output ``(tgt_len, batch, hidden)``
            * dictionary attention dists of ``(tgt_len, batch, src_len)``
        """
        # src: (len, batch, features)
        enc_state, memory_bank, lengths = \
            self.encoder(src, lengths)

        # mean pool
        # r: (batch, features)
        r = masked_average(memory_bank, src_lengths)

        # estimate mu and sigma
        estimates = self.W_inf(r)
        # mu, sigma: (batch, zdim)
        mu, sigma = torch.chunk(estimates, 2, dim=1)
        sigma = F.softplus(sigma)

        return mu, sigma

    def update_dropout(self, dropout):
        self.encoder.update_dropout(dropout)