# Copyright 2019 The Texar Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Attentional Seq2seq.
from https://github.com/LiqunChen0606/OT-Seq2Seq/tree/master/texar

## Run text summarization
Using **GIGAWORD** dataset.
```bash
# Using Pytorch nn.Embedding Layer
nohup python seq2seq_attn.py --embedding_type nn.Embedding --gpu 4 --runName G_000 --config-model config_model --config-data config_giga &> outputs/giga_run000.txt &
#Model  Rouge-1 Rouge-2 Rouge-L
#MLE    36.11 ± 0.21    16.39 ± 0.16    32.32 ± 0.19

# Using EmbeddingKet Layer
nohup python seq2seq_attn.py --embedding_type EmbeddingKet   --gpu 0 --runName V2K_G_000    --logdir V2K    --config-model config_model --config-data config_giga &> outputs/V2K_G_000.txt    &

# Using EmbeddingKetXS Layer
nohup python seq2seq_attn.py --embedding_type EmbeddingKetXS --gpu 0 --runName V2K_XS_G_000 --logdir V2K_XS --config-model config_model --config-data config_giga &> outputs/V2K_XS_G_000.txt &
```

## Run German-English machine translation
Using **IWSLT2014** (DE-EN) dataset
```bash
# Using Pytorch nn.Embedding Layer
nohup python seq2seq_attn.py --embedding_type nn.Embedding --gpu 4 --runName I_000 --config-model config_model --config-data config_iwslt14 &> outputs/iwslt14_run000.txt &
#Model  BLEU Score
#MLE    26.44 ± 0.18

# Using EmbeddingKet Layer
nohup python seq2seq_attn.py --embedding_type EmbeddingKet   --gpu 0 --runName V2K_I_000    --logdir V2K    --config-model config_model --config-data config_iwslt14 &> outputs/V2K_I_000.txt    &

# Using EmbeddingKetXS Layer
nohup python seq2seq_attn.py --embedding_type EmbeddingKetXS --gpu 0 --runName V2K_XS_I_000 --logdir V2K_XS --config-model config_model --config-data config_iwslt14 &> outputs/V2K_XS_I_000.txt &

```
"""
from word2ket import EmbeddingKet, EmbeddingKetXS, ketify, summary
import importlib
from rouge import Rouge
import os
import argparse
parser = argparse.ArgumentParser()

parser.add_argument('--embedding_type', type=str, default="nn.Embedding", help="nn.Embedding or EmbeddingKet or EmbeddingKetXS")
parser.add_argument('--order', type=int, default=4)
parser.add_argument('--rank', type=int, default=1)
parser.add_argument('--embedding_dim', type=int, default=256)
parser.add_argument(
    '--config-model', type=str, default="config_model_OTMY",
    help="The model config.")
parser.add_argument(
    '--config-data', type=str, default="config_giga",
    help="The dataset config.")
parser.add_argument('--gpu', type=str, default="0")
parser.add_argument('--runName', type=str, default="orgG_999_")
parser.add_argument('--logdir', type=str, default="logs")
parser.add_argument('--epochs', type=int, default=35)
args = parser.parse_args()


os.environ["CUDA_VISIBLE_DEVICES"]=args.gpu
embedding_type = args.embedding_type # 'EmbeddingKet' or 'EmbeddingKetXS'
order = args.order
rank = args.rank
embedding_dim = args.embedding_dim
print('embedding_type', embedding_type, 'order', order, 'embedding_dim', embedding_dim, 'rank', rank)
epochs = args.epochs
runName = args.runName
config_model = importlib.import_module(args.config_model)
config_data = importlib.import_module(args.config_data)

import torch
import torch.nn as nn
import torch.nn.functional as F
import texar.torch as tx
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Seq2SeqAttn(nn.Module):
    def __init__(self, train_data):
        super().__init__()
        self.doOnlyMLE=False
        self.source_vocab_size = train_data.source_vocab.size
        self.target_vocab_size = train_data.target_vocab.size
        print('train_data.source_vocab.size', train_data.source_vocab.size)
        print('train_data.target_vocab.size', train_data.target_vocab.size)
        self.trunc_len_src = train_data.hparams.source_dataset.max_seq_length
        self.trunc_len_tgt = train_data.hparams.target_dataset.max_seq_length
        self.bos_token_id = train_data.target_vocab.bos_token_id
        self.eos_token_id = train_data.target_vocab.eos_token_id
        
        if embedding_type == 'nn.Embedding':
            self.source_embedder = nn.Embedding(
                num_embeddings=self.source_vocab_size,
                embedding_dim=embedding_dim)

            self.target_embedder = nn.Embedding(
                num_embeddings=self.target_vocab_size,
                embedding_dim=embedding_dim)

        elif embedding_type == 'EmbeddingKet':
            self.source_embedder = EmbeddingKet(
                num_embeddings=self.source_vocab_size,
                embedding_dim=embedding_dim,
                order=order,
                rank=rank)

            self.target_embedder = EmbeddingKet(
                num_embeddings=self.target_vocab_size,
                embedding_dim=embedding_dim,
                order=order,
                rank=rank)
        
        elif embedding_type == 'EmbeddingKetXS':
            self.source_embedder = EmbeddingKetXS(
                num_embeddings=self.source_vocab_size,
                embedding_dim=embedding_dim,
                order=order,
                rank=rank)

            self.target_embedder = EmbeddingKetXS(
                num_embeddings=self.target_vocab_size,
                embedding_dim=embedding_dim,
                order=order,
                rank=rank)

        self.encoder = tx.modules.BidirectionalRNNEncoder(
            input_size=embedding_dim,
            hparams=config_model.encoder)

        self.decoder = tx.modules.AttentionRNNDecoder(
            token_embedder=self.target_embedder,
            encoder_output_size=(self.encoder.cell_fw.hidden_size +
                                 self.encoder.cell_bw.hidden_size),
            input_size=embedding_dim,
            vocab_size=self.target_vocab_size,
            hparams=config_model.decoder)

    def mleonly(self,val):
        self.doOnlyMLE=val

    def forward(self, batch, mode):
        enc_outputs, _ = self.encoder(
            inputs=self.source_embedder(batch['source_text_ids']),
            sequence_length=batch['source_length'])
        


        memory = torch.cat(enc_outputs, dim=2)

        if mode == "train":
            #https://github.com/asyml/texar-pytorch/issues/161
            #https://texar.readthedocs.io/en/latest/code/modules.html#texar.tf.modules.GumbelSoftmaxEmbeddingHelper
            #https://texar.readthedocs.io/en/latest/code/modules.html
            helper_train = self.decoder.create_helper(
                decoding_strategy="train_greedy")

            training_outputs, _, _ = self.decoder(
                memory=memory,
                memory_sequence_length=batch['source_length'],
                helper=helper_train,
                inputs=batch['target_text_ids'][:, :-1],
                sequence_length=batch['target_length'] - 1)


            mle_loss = tx.losses.sequence_sparse_softmax_cross_entropy(
                labels=batch['target_text_ids'][:, 1:],
                logits=training_outputs.logits,
                sequence_length=batch['target_length'] - 1)


            return mle_loss
        else:
            start_tokens = memory.new_full(
                batch['target_length'].size(), self.bos_token_id,
                dtype=torch.int64)

            infer_outputs = self.decoder(
                start_tokens=start_tokens,
#                end_token=self.eos_token_id.item(),
                end_token=self.eos_token_id,
                memory=memory,
                memory_sequence_length=batch['source_length'],
                beam_width=config_model.beam_width)
            #print("WHEW... NO ERROR")

            return infer_outputs

def print_stdout_and_file(content, file):
    print(content)
    print(content, file=file)

def main():
    train_data = tx.data.PairedTextData(
        hparams=config_data.train, device=device)
    val_data = tx.data.PairedTextData(
        hparams=config_data.val, device=device)
    test_data = tx.data.PairedTextData(
        hparams=config_data.test, device=device)
    data_iterator = tx.data.TrainTestDataIterator(
        train=train_data, val=val_data, test=test_data)


    
    model = Seq2SeqAttn(train_data)
    summary(model)
    model.mleonly(True)
    model.to(device)
    train_op = tx.core.get_train_op(
        params=model.parameters(), hparams=config_model.opt)

    def _train_epoch():
        data_iterator.switch_to_train_data()
        model.train()

        step = 0
        for batch in data_iterator:
            loss = model(batch, mode="train")
            loss.backward()
            train_op()
            if step % config_data.display == 0:
                print("TRAIN: step={}, loss={:.4f}".format(step, loss),flush=True)
            step += 1

    @torch.no_grad()
    def _eval_epoch(mode):
        if mode == 'val':
            data_iterator.switch_to_val_data()
        else:
            data_iterator.switch_to_test_data()
        model.eval()

        refs, hypos = [], []
        evalStart=0
        for batch in data_iterator:

            infer_outputs = model(batch, mode="val")
            output_ids = infer_outputs["sample_id"][:, :, 0].cpu()
            target_texts_ori = [text[1:] for text in batch['target_text']]
            target_texts = tx.utils.strip_special_tokens(
                target_texts_ori, is_token_list=True)
            output_texts = tx.data.vocabulary.map_ids_to_strs(
                ids=output_ids, vocab=val_data.target_vocab)

            if (evalStart==0):
                src_words = model.source_embedder(batch['source_text_ids'])
                print('src wrds',src_words[0,0:2,0:10].flatten(), target_texts[0])
                evalStart=1
            else:
                pass

            for hypo, ref in zip(output_texts, target_texts):
                #ADD ROUGE
                if config_data.eval_metric == 'bleu':
                    hypos.append(hypo)
                    refs.append([ref])
                elif config_data.eval_metric == 'rouge':
                    hh=str(hypo)
                    if (len(hh)==0):
                        hh=" "
                    rr=' '.join(ref)
                    hypos.append(hh)
                    refs.append(rr)                    

#ADD ROUGE
        if config_data.eval_metric == 'bleu':
            return tx.evals.corpus_bleu_moses(
            list_of_references=refs, hypotheses=hypos)
        elif config_data.eval_metric == 'rouge':
            rouge = Rouge()
            print('HH',type(hypos),type(hypos[0]),hypos[0])
            print('RR',type(refs),type(refs[0]),refs[0])
            return rouge.get_scores(hyps=hypos, refs=refs, avg=True)

    def _calc_reward(score):
        """
        Return the bleu score or the sum of (Rouge-1, Rouge-2, Rouge-L).
        """
        if config_data.eval_metric == 'bleu':
            return score
        elif config_data.eval_metric == 'rouge':
            return sum([value['f'] for key, value in score.items()])
        

    best_val_score = -1.
    print("test RUN %s %s %s"%(runName,args.config_model,args.config_data),flush=True)
    final_val_score = -1.0
    final_test_score = -1.0
    final_epoch = -1
    curr_rouge1 = -1
    curr_rouge2 = -1
    curr_rougeL = -1
    final_rouge1 = -1
    final_rouge2 = -1
    final_rougeL = -1

    fileName='%s/logs_%s_%s_%s.log'%(args.logdir,runName,args.config_model,args.config_data)
    scores_file = open(fileName, 'w', encoding='utf-8')

    for i in range(epochs):
        if (epochs>=5):
            model.mleonly(False)

        _train_epoch()

        val_score = _eval_epoch('val')
        test_score = _eval_epoch('test')

        val_score_c = _calc_reward(val_score)
        test_score_c = _calc_reward(test_score)


        best_val_score = max(best_val_score, val_score_c)


        if config_data.eval_metric == 'bleu':
            if (val_score_c > final_val_score):
                final_val_score = val_score_c
                final_test_score = test_score_c
                final_epoch = i
                ckpt_fn = 'modelsORG/org_%s_%02d_%s_model.ckpt'%(args.config_data,i,runName)
                torch.save(model.state_dict(), ckpt_fn)

            print_stdout_and_file(
                'val epoch={}, BLEU={:.4f}; best-ever={:.4f}; report_val={:.4f}; report_test={:.4f}; report_epoch={}'.format(
                    i, val_score_c, best_val_score,final_val_score,final_test_score,final_epoch), file=scores_file)

            print_stdout_and_file(
                'TEST epoch={}, BLEU={:.4f}'.format(i, test_score_c),
                file=scores_file)
            print_stdout_and_file('=' * 50, file=scores_file)

        elif config_data.eval_metric == 'rouge':
            if (val_score_c > final_val_score):
                final_val_score = val_score_c
                final_test_score = test_score_c
                final_epoch = i
                ckpt_fn = 'modelsORG/org_%s_%02d_%s_model.ckpt'%(args.config_data,i,runName)
                torch.save(model.state_dict(), ckpt_fn)

            valStr=''
            testStr=''
            
            print_stdout_and_file(
                'valid_epoch {}:'.format(i), file=scores_file)
            for key, value in val_score.items():
                print_stdout_and_file(
                    '{}: {}'.format(key, value), file=scores_file)
                valStr=valStr+" "+'{}: {:.3f}'.format(key, value['f'])


            print_stdout_and_file(
                'test_epoch {}:'.format(i), file=scores_file)
            for key, value in test_score.items():
                print_stdout_and_file(
                    '{}: {}'.format(key, value), file=scores_file)
                testStr=testStr+" "+'{}: {:.4f}'.format(key, value['f'])
                if (final_epoch == i):
                    if (key=='rouge-1'):
                        final_rouge1 = value['f']
                    if (key=='rouge-2'):
                        final_rouge2 = value['f']
                    if (key=='rouge-l'):
                        final_rougeL = value['f']

            print_stdout_and_file('TEST: epoch={}, val:'.format(i)+valStr+' test:'+testStr+
                ' fsum: {:.4f}; fsum_test: {:.4f}; best_val_fsum: {:.4f}; report_val={:.4f}; report_test={:.4f}; report_test-ROUGE12L={:.4f},{:.4f},{:.4f}; report_epoch={}'.format(
                val_score_c, test_score_c, best_val_score,final_val_score,final_test_score,final_rouge1,final_rouge2,final_rougeL,final_epoch), file=scores_file)


            print_stdout_and_file('=' * 110, file=scores_file)
        scores_file.flush()

    print('FINAL',final_epoch,final_val_score,final_test_score,flush=True)
    ckpt_fn = 'modelsORG/org_%s_%02d_%s_model_FINAL.ckpt'%(args.config_data,i,runName)
    torch.save(model.state_dict(), ckpt_fn)


if __name__ == '__main__':
    main()
