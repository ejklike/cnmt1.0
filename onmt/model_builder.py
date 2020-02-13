"""
This file is for models creation, which consults options
and creates each encoder and decoder accordingly.
"""
import re
import torch
import torch.nn as nn
from torch.nn.init import xavier_uniform_

import onmt.inputters as inputters
import onmt.modules
from onmt.modules import (Embeddings, 
                          CopyGenerator,
                          TransformerEncoder, 
                          TransformerDecoder)
from onmt.modules.util_class import Cast
from onmt.utils.misc import use_gpu
from onmt.utils.logging import logger
from onmt.utils.parse import ArgumentParser


def build_embeddings(opt, text_field, for_encoder=True):
    """
    Args:
        opt: the option in current environment.
        text_field(TextMultiField): word and feats field.
        for_encoder(bool): build Embeddings for encoder or decoder?
    """
    emb_dim = opt.src_word_vec_size if for_encoder else opt.tgt_word_vec_size

    pad_indices = [f.vocab.stoi[f.pad_token] for _, f in text_field]
    word_padding_idx, feat_pad_indices = pad_indices[0], pad_indices[1:]

    num_embs = [len(f.vocab) for _, f in text_field]
    num_word_embeddings, num_feat_embeddings = num_embs[0], num_embs[1:]

    fix_word_vecs = opt.fix_word_vecs_enc if for_encoder \
        else opt.fix_word_vecs_dec

    emb = Embeddings(
        word_vec_size=emb_dim,
        position_encoding=opt.position_encoding,
        feat_merge=opt.feat_merge,
        feat_vec_exponent=opt.feat_vec_exponent,
        feat_vec_size=opt.feat_vec_size,
        dropout=opt.dropout[0] if type(opt.dropout) is list else opt.dropout,
        word_padding_idx=word_padding_idx,
        feat_padding_idx=feat_pad_indices,
        word_vocab_size=num_word_embeddings,
        feat_vocab_sizes=num_feat_embeddings,
        sparse=opt.optim == "sparseadam",
        fix_word_vecs=fix_word_vecs
    )
    return emb


def build_encoder(opt, embeddings):
    """
    Various encoder dispatcher function.
    Args:
        opt: the option in current environment.
        embeddings (Embeddings): vocab embeddings for this encoder.
    """
    return TransformerEncoder.from_opt(opt, embeddings)


def build_decoder(opt, embeddings):
    """
    Various decoder dispatcher function.
    Args:
        opt: the option in current environment.
        embeddings (Embeddings): vocab embeddings for this decoder.
    """
    return TransformerDecoder.from_opt(opt, embeddings)


def build_generator(opt, embeddings):
    if not opt.copy_attn:
        if opt.generator_function == "sparsemax":
            gen_func = onmt.modules.sparse_activations.LogSparsemax(dim=-1)
        else:
            gen_func = nn.LogSoftmax(dim=-1)
        generator = nn.Sequential(
            nn.Linear(opt.dec_rnn_size,
                    len(fields["tgt"].base_field.vocab)),
            Cast(torch.float32),
            gen_func
        )
        if opt.share_decoder_embeddings:
            generator[0].weight = embeddings.word_lut.weight
    else:
        tgt_base_field = fields["tgt"].base_field
        vocab_size = len(tgt_base_field.vocab)
        pad_idx = tgt_base_field.vocab.stoi[tgt_base_field.pad_token]
        generator = CopyGenerator(opt.dec_rnn_size, vocab_size, pad_idx)
        if opt.share_decoder_embeddings:
            generator.linear.weight = embeddings.word_lut.weight


def init_param(model):
    for p in model.parameters():
        p.data.uniform_(-model_opt.param_init, model_opt.param_init)


def init_xavier_param(model):
    for p in model.parameters():
        if p.dim() > 1:
            xavier_uniform_(p)


def load_test_model(opt, model_path=None):
    if model_path is None:
        model_path = opt.models[0]
    checkpoint = torch.load(model_path,
                            map_location=lambda storage, loc: storage)

    model_opt = ArgumentParser.ckpt_model_opts(checkpoint['opt'])
    ArgumentParser.update_model_opts(model_opt)
    ArgumentParser.validate_model_opts(model_opt)
    vocab = checkpoint['vocab']
    if inputters.old_style_vocab(vocab):
        fields = inputters.load_old_vocab(
            vocab, dynamic_dict=model_opt.copy_attn
        )
    else:
        fields = vocab

    model = build_base_model(model_opt, fields, use_gpu(opt), checkpoint,
                             opt.gpu)
    if opt.fp32:
        model.float()
    model.eval()
    model.generator.eval()
    return fields, model, model_opt


def build_base_model(model_opt, fields, gpu, checkpoint=None, gpu_id=None):
    """Build a model from opts.

    Args:
        model_opt: the option loaded from checkpoint. It's important that
            the opts have been updated and validated. See
            :class:`onmt.utils.parse.ArgumentParser`.
        fields (dict[str, torchtext.data.Field]):
            `Field` objects for the model.
        gpu (bool): whether to use gpu.
        checkpoint: the model gnerated by train phase, or a resumed snapshot
                    model from a stopped training.
        gpu_id (int or NoneType): Which GPU to use.

    Returns:
        the NMTModel.
    """

    # for back compat when attention_dropout was not defined
    try:
        model_opt.attention_dropout
    except AttributeError:
        model_opt.attention_dropout = model_opt.dropout

    # Build embeddings.
    src_field = fields["src"]
    src_emb = build_embeddings(model_opt, src_field, for_encoder=True)
    tgt_field = fields["tgt"]
    tgt_emb = build_embeddings(model_opt, tgt_field, for_encoder=False)
    # Share the embedding matrix - preprocess with share_vocab required.
    if model_opt.share_embeddings:
        # src/tgt vocab should be the same if `-share_vocab` is specified.
        assert src_field.base_field.vocab == tgt_field.base_field.vocab, \
            "preprocess with -share_vocab if you use share_embeddings"
        tgt_emb.word_lut.weight = src_emb.word_lut.weight

    # Q(z|x,y)
    src_q_encoder = build_encoder(model_opt, src_emb)
    tgt_q_encoder = build_encoder(model_opt, tgt_emb)
    W_q = nn.Linear(model_opt.enc_rnn_size * 2, model_opt.zdim * 2)
    q_inf_model = onmt.models.QInfModel(src_q_encoder, tgt_q_encoder, W_q) #
    
    # p(z|x)
    src_p_encoder = build_encoder(model_opt, src_emb)
    W_src_p = nn.Linear(model_opt.enc_rnn_size, model_opt.zdim * 2)
    src_p_inf_model = onmt.models.PInfModel(src_p_encoder, W_src_p) #

    # p(z|y)
    tgt_p_encoder = build_encoder(model_opt, tgt_emb)
    W_tgt_p = nn.Linear(model_opt.enc_rnn_size, model_opt.zdim * 2)
    tgt_p_inf_model = onmt.models.PInfModel(tgt_p_encoder, W_tgt_p) #

    # Build NMTModel(= encoder + decoder).
    shared_encoder = build_encoder(model_opt, src_emb)
    src2tgt_decoder = build_decoder(model_opt, tgt_emb)
    tgt2src_decoder = build_decoder(model_opt, src_emb)
    # p(y|x,z), p(x|y,z)
    src2tgt_model = onmt.models.NMTModel(shared_encoder, src2tgt_decoder) #
    tgt2src_model = onmt.models.NMTModel(shared_encoder, tgt2src_decoder) #

    # Build Generator.
    src2tgt_generator = build_generator(model_opt, tgt_emb) #
    tgt2src_generator = build_generator(model_opt, src_emb) #

    # Load the model states from checkpoint or initialize them.
    if checkpoint is not None:
        # This preserves backward-compat for models using customed layernorm
        def fix_key(s):
            s = re.sub(r'(.*)\.layer_norm((_\d+)?)\.b_2',
                       r'\1.layer_norm\2.bias', s)
            s = re.sub(r'(.*)\.layer_norm((_\d+)?)\.a_2',
                       r'\1.layer_norm\2.weight', s)
            return s

        for key in ['q_inf_model', 'src_p_inf_model', 'tgt_p_inf_model', 
                    'src2tgt_model', 'tgt2src_model']:
            checkpoint[key] = {fix_key(k): v
                               for k, v in checkpoint[key].items()}
        # end of patch for backward compatibility

        q_inf_model.load_state_dict(
            checkpoint['q_inf_model'], strict=False)
        src_p_inf_model.load_state_dict(
            checkpoint['src_p_inf_model'], strict=False)
        tgt_p_inf_model.load_state_dict(
            checkpoint['tgt_p_inf_model'], strict=False)
        src2tgt_model.load_state_dict(
            checkpoint['src2tgt_model'], strict=False)
        tgt2src_model.load_state_dict(
            checkpoint['tgt2src_model'], strict=False)
        src2tgt_generator.load_state_dict(
            checkpoint['src2tgt_generator'], strict=False)
        tgt2src_generator.load_state_dict(
            checkpoint['tgt2src_generator'], strict=False)

    else:
        if model_opt.param_init != 0.0:
            for model in [q_inf_model, src_p_inf_model, tgt_p_inf_model, 
                          src2tgt_model, tgt2src_model, 
                          src2tgt_generator, tgt2src_generator]:
                init_param(model)
        if model_opt.param_init_glorot:
            for model in [q_inf_model, src_p_inf_model, tgt_p_inf_model, 
                          src2tgt_model, tgt2src_model, 
                          src2tgt_generator, tgt2src_generator]:
                init_xavier_param(model)

        for model in [q_inf_model.src_encoder,
                      q_inf_model.tgt_encoder,
                      src_p_inf_model.encoder,
                      tgt_p_inf_model.encoder,
                      src2tgt_model.encoder,
                      src2tgt_model.decoder,
                      tgt2src_model.encoder,
                      tgt2src_model.decoder]:
            if hasattr(model, 'embeddings'):
                model.embeddings.load_pretrained_vectors(
                    model_opt.pre_word_vecs_enc)

    src2tgt_model.generator = src2tgt_generator
    tgt2src_model.generator = tgt2src_generator

    if gpu and gpu_id is not None:
        device = torch.device("cuda", gpu_id)
    elif gpu and not gpu_id:
        device = torch.device("cuda")
    elif not gpu:
        device = torch.device("cpu")
    
    model = CNMTModel(q_inf_model, src_p_inf_model, tgt_p_inf_model, 
                      src2tgt_model, tgt2src_model)
    model.to(device)
    if model_opt.model_dtype == 'fp16' and model_opt.optim == 'fusedadam':
        model.half()
    return model


def build_model(model_opt, opt, fields, checkpoint):
    logger.info('Building model...')
    model = build_base_model(model_opt, fields, use_gpu(opt), checkpoint)
    logger.info(model)
    return model
