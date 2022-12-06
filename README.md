General usage:

```
usage: qa_multiconer.py [-h] [-p DATASET_PATH] [-g LANG] [-q MODEL] [-m MAX_LENGTH] [-s DOC_STRIDE] [-b BATCH_SIZE_TRAIN] [-t BATCH_SIZE_TEST] [-e EPOCHS] [-r {train,test,all}] [-x SUFFIX]

optional arguments:
  -h, --help            show this help message and exit
  -p DATASET_PATH, --dataset_path DATASET_PATH
                        QA dataset folder (default: ./MultiCoNER_2_train_dev/train_dev/)
  -g LANG, --lang LANG  Language for QA dataset (default: en)
  -q MODEL, --model MODEL
                        Pretrained QA model (default: xlm-roberta-base)
  -m MAX_LENGTH, --max_length MAX_LENGTH
                        Maximum length of a feature (question and context), 512 for xlm-roberta-base (default: 512)
  -s DOC_STRIDE, --doc_stride DOC_STRIDE
                        Authorized overlap between two part of the context when splitting it is needed (default: 128)
  -b BATCH_SIZE_TRAIN, --batch_size_train BATCH_SIZE_TRAIN
                        Batch size (default: 4)
  -t BATCH_SIZE_TEST, --batch_size_test BATCH_SIZE_TEST
                        Batch size (default: 16)
  -e EPOCHS, --epochs EPOCHS
                        Number of epochs (default: 4)
  -r {train,test,all}, --mode {train,test,all}
                        Operation mode (default: all)
  -x SUFFIX, --suffix SUFFIX
                        Suffix to output names (default: )


```

Train and test with `xlm-roberta-base` LM:

```
CUDA_VISIBLE_DEVICES=0,1,2 python qa_multiconer.py  -m 64 -s 16 -b 16 -t 64 -e 4 -r all -q xlm-roberta-base
```

Train and test with `bigbird-roberta-base` LM:

```
CUDA_VISIBLE_DEVICES=0,1,2 python qa_multiconer.py -m 64 -s 16 -b 32 -t 128 -r all -q google/bigbird-roberta-base
```

Test fine-tuned QAmodel with multiconerII train dataset:

```
CUDA_VISIBLE_DEVICES=0 python qa_multiconer.py -m 64 -s 16 -t 64 -r test -q ./xlm-roberta-base-finetuned-multiconer2comp
```

