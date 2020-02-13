# cnmt1.0

NMT experiments based on [OpenNMT](https://github.com/OpenNMT/OpenNMT)

  * Work on Python 3.6 and PyTorch 1.2 (should work with 1.1)
  * Required packages: `configargparse`, `torchtext==0.4.0`

## Features

- [Seq2Seq models (encoder-decoder) with multiple RNN cells (lstm/gru) and attention (dotprod/mlp) types](http://opennmt.net/OpenNMT-py/options/train.html#model-encoder-decoder)
- [Transformer models](http://opennmt.net/OpenNMT-py/FAQ.html#how-do-i-use-the-transformer-model)
- [Copy and Coverage Attention](http://opennmt.net/OpenNMT-py/options/train.html#model-attention)
- [Pretrained Embeddings](http://opennmt.net/OpenNMT-py/FAQ.html#how-do-i-use-pretrained-embeddings-e-g-glove)
- [Source word features](http://opennmt.net/OpenNMT-py/options/train.html#model-embeddings)
- [TensorBoard logging](http://opennmt.net/OpenNMT-py/options/train.html#logging)
- [Multi-GPU training](http://opennmt.net/OpenNMT-py/FAQ.html##do-you-support-multi-gpu)
- [Data preprocessing](http://opennmt.net/OpenNMT-py/options/preprocess.html)
- [Inference (translation) with batching and beam search](http://opennmt.net/OpenNMT-py/options/translate.html)
- Inference time loss functions.
- [Conv2Conv convolution model]
- SRU "RNNs faster than CNN" paper
- Mixed-precision training with [APEX](https://github.com/NVIDIA/apex), optimized on [Tensor Cores](https://developer.nvidia.com/tensor-cores)

## Quickstart

[Full Documentation](http://opennmt.net/OpenNMT-py/)


### Step 1: Preprocess the data

```bash
onmt_preprocess -train_src data/src-train.txt -train_tgt data/tgt-train.txt -valid_src data/src-val.txt -valid_tgt data/tgt-val.txt -save_data data/demo
```

We will be working with some example data in `data/` folder.

The data consists of parallel source (`src`) and target (`tgt`) data containing one sentence per line with tokens separated by a space:

* `src-train.txt`
* `tgt-train.txt`
* `src-val.txt`
* `tgt-val.txt`

Validation files are required and used to evaluate the convergence of the training. It usually contains no more than 5000 sentences.


After running the preprocessing, the following files are generated:

* `demo.train.pt`: serialized PyTorch file containing training data
* `demo.valid.pt`: serialized PyTorch file containing validation data
* `demo.vocab.pt`: serialized PyTorch file containing vocabulary data


Internally the system never touches the words themselves, but uses these indices.

### Step 2: Train the model

```bash
onmt_train -data data/demo -save_model demo-model
```

The main train command is quite simple. Minimally it takes a data file
and a save file.  This will run the default model, which consists of a
2-layer LSTM with 500 hidden units on both the encoder/decoder.
If you want to train on GPU, you need to set, as an example:
`CUDA_VISIBLE_DEVICES=1,3`
`-world_size 2 -gpu_ranks 0 1` to use (say) GPU 1 and 3 on this node only.
To know more about distributed training on single or multi nodes, read the FAQ section.

### Step 3: Translate

```bash
onmt_translate -model demo-model_acc_XX.XX_ppl_XXX.XX_eX.pt -src data/src-test.txt -output pred.txt -replace_unk -verbose
```

Now you have a model which you can use to predict on new data. We do this by running beam search. This will output predictions into `pred.txt`.