



> python preprocess.py -train_src data/src-train.txt -train_tgt data/tgt-train.txt -valid_src data/src-val.txt -valid_tgt data/tgt-val.txt -save_data data/data -src_vocab_size 1000 -tgt_vocab_size 1000 

<!-- option: -share_vocab -->

> python train.py -data data/data -save_model data/tmp -rnn_size 10 -word_vec_size 10 -heads 1 -layers 1 -train_steps 100 -optim adam  -learning_rate 0.001

<!-- option: -world_size 1 -gpu_ranks 0  -->

> python translate.py -model ./data/tmp_step_100.pt -src data/src-test.10.txt -output data/pred.10.txt -replace_unk -beam_size 10 -n_best 10

<!-- option: -gpu 0 -->