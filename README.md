# word_segmentation
This model infers word breaks, predicate identification, and phrase breaks for input sentences.

## Dataset
- The preprocessed data is already in `data` directory

## Train
`$ python train.py --dataset ./data --out_dir ./out --early_stopping 5`

## Test
### Evaluation
`$ python test.py --eval --model ./out/best.model --test_file ./data/test.json`

### Decode
`$ python test.py --decode --model ./out/best.model --char2index ./data/char2index.json --decode_file [PATH] --out_file [PATH] `

### interactive
`$ python test.py --interactive --model ./out/best.model --char2index ./data/char2index.json`

