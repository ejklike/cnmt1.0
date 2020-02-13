""" Onmt Inference Model base class definition """
import torch.nn as nn
import torch.nn.functional as F

from onmt.utils.misc import masked_average


class QInfModel(nn.Module):

    def __init__(self, src_encoder, tgt_encoder, W_inf):
        super(QInfModel, self).__init__()
        self.src_encoder = src_encoder
        self.tgt_encoder = tgt_encoder
        self.W_inf = W_inf

    def forward(self, src, tgt, src_lengths, tgt_lengths):
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
        src_enc_state, src_memory_bank, src_lengths = \
            self.src_encoder(src, src_lengths)
        tgt_enc_state, tgt_memory_bank, tgt_lengths = \
            self.tgt_encoder(tgt, tgt_lengths)

        # mean pool
        # r: (batch, features)
        r_src = masked_average(src_memory_bank, src_lengths)
        r_tgt = masked_average(tgt_memory_bank, tgt_lengths)

        # concatenate
        # r: (batch, features * 2)
        r = torch.cat([r_src, r_tgt], dim=1)

        # estimate mu and sigma
        estimates = self.W_inf(r)
        # mu, sigma: (batch, zdim)
        mu, sigma = torch.chunk(estimates, 2, dim=1)
        sigma = F.softplus(sigma)

        return mu, sigma

    def update_dropout(self, dropout):
        self.src_encoder.update_dropout(dropout)
        self.tgt_encoder.update_dropout(dropout)