""" Onmt NMT Model base class definition """
import torch.nn as nn


def batch_z(z, lengths, max_len=None):
    """
    input z: (batch, zdim)
    output z: (len, batch, zdim)
    """
    max_len = max_len or lengths.max()
    return z.unsqueeze(0).repeat(maxlen, 1, 1)


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

    def forward(self, src, tgt, lengths, bptt=False, with_align=False):
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
        
        # Sample z
        noise = mu.clone().normal_()
        z = mu + noise * sigma

        # create batches of z
        z_src = batch_z(z, src_lengths)
        z_tgt = batch_z(z, tgt_lengths)

        # LM(x), LM(y)
        prob_srclm = self.srclm(z_src, src, src_lengths, src_emb)
        prob_tgtlm = self.tgtlm(z_tgt, tgt, tgt_lengths, tgt_emb)

        

        dec_in_tgt = tgt[:-1]  # exclude last target from inputs

        enc_state, memory_bank, lengths = self.encoder(src, lengths)

        if bptt is False:
            self.decoder.init_state(src, memory_bank, enc_state)
        dec_out, attns = self.decoder(dec_in, memory_bank,
                                    memory_lengths=lengths,
                                    with_align=with_align)
        return dec_out, attns

    def update_dropout(self, dropout):
        self.q_inf.update_dropout(dropout)
        self.src_p_inf.update_dropout(dropout)
        self.tgt_p_inf.update_dropout(dropout)
        self.src2tgt.update_dropout(dropout)
        self.tgt2src.update_dropout(dropout)
