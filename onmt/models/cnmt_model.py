""" Onmt NMT Model base class definition """
import torch.nn as nn


# def batch_z(z, lengths, max_len=None):
#     """
#     input z: (batch, zdim)
#     output z: (len, batch, zdim)
#     """
#     max_len = max_len or lengths.max()
#     return z.unsqueeze(0).repeat(maxlen, 1, 1)


class CNMTModel(nn.Module):
    """
    Core trainable object in OpenNMT. Implements a trainable interface
    for a simple, generic encoder + decoder model.

    Args:
      encoder (onmt.encoders.EncoderBase): an encoder object
      decoder (onmt.decoders.DecoderBase): a decoder object
    """

    def __init__(self, q_inf, src_p_inf, tgt_p_inf, src2tgt, tgt2src):
        super(CNMTModel, self).__init__()
        self.q_inf = q_inf
        self.src_p_inf = src_p_inf
        self.tgt_p_inf = tgt_p_inf
        self.src2tgt = src2tgt
        self.tgt2src = tgt2src

    def forward(self, src, tgt, src_lengths, tgt_lengths, 
                bptt=False, with_align=False):
        """Forward propagate a `src` and `tgt` pair for training.
        Possible initialized with a beginning decoder state.

        Args:
            src (Tensor): A source sequence passed to encoder.
                typically for inputs this will be a padded `LongTensor`
                of size ``(len, batch, features)``.
            tgt (LongTensor): A target sequence passed to decoder.
                Size ``(tgt_len, batch, features)``.
            lengths(LongTensor): The src lengths, pre-padding ``(batch,)``.
            bptt (Boolean): A flag indicating if truncated bptt is set.
                If reset then init_state
            with_align (Boolean): A flag indicating whether output alignment,
                Only valid for transformer decoder.

        Returns:
            (FloatTensor, dict[str, FloatTensor]):

            * decoder output ``(tgt_len, batch, hidden)``
            * dictionary attention dists of ``(tgt_len, batch, src_len)``
        """
        # Q(z|x,y)
        mu, sigma = self.q_inf(src, tgt, src_lengths, tgt_lengths)

        # p(z|x), p(z|y)
        mu_src, sigma_src = self.src_p_inf(src, src_lengths)
        mu_tgt, sigma_tgt = self.tgt_p_inf(tgt, tgt_lengths)
        
        # Sample z
        noise = mu.clone().normal_()
        z = mu + noise * sigma # [bs, zdim]

        # # create batches of z
        # z_src = batch_z(z, src_lengths)
        # z_tgt = batch_z(z, tgt_lengths)

        # p(y|x,z)
        tgt_out = self.src2tgt(src, tgt, src_lengths, bptt=bptt, with_align=with_align)

        dec_out, attns = self.decoder(dec_in, memory_bank,
                                    memory_lengths=lengths,
                                    with_align=with_align)

        dec_in_tgt2src = src[:-1]

        

        
        return dec_out, attns

    def update_dropout(self, dropout):
        self.q_inf.update_dropout(dropout)
        self.src_p_inf.update_dropout(dropout)
        self.tgt_p_inf.update_dropout(dropout)
        self.src2tgt.update_dropout(dropout)
        self.tgt2src.update_dropout(dropout)
