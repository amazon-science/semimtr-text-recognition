# DATA Structure

- Training and evaluation require download preprocessed lmdb. [Link](https://github.com/ku21fan/STR-Fewer-Labels/blob/main/data.md#download-preprocessed-lmdb-dataset-for-traininig-and-evaluation)
- Pretraining the language model requires the WikiText103. [Link](https://github.com/FangShancheng/ABINet#datasets)
- The final structure of `data` directory is:

    ```
    data
    ├── charset_36.txt
    ├── training
    │   ├── label
    │   │   ├── real
    │   │   │   ├── 1.SVT
    │   │   │   ├── 2.IIIT
    │   │   │   ├── 3.IC13
    │   │   │   ├── 4.IC15
    │   │   │   ├── 5.COCO
    │   │   │   ├── 6.RCTW17
    │   │   │   ├── 7.Uber
    │   │   │   ├── 8.ArT
    │   │   │   ├── 9.LSVT
    │   │   │   ├── 10.MLT19
    │   │   │   └── 11.ReCTS
    │   │   └── synth (for synthetic data, follow guideline at https://github.com/ku21fan/STR-Fewer-Labels/blob/main/data.md)
    │   │       ├── MJ
    │   │       │   ├── MJ_train
    │   │       │   ├── MJ_valid
    │   │       │   └── MJ_test
    │   │       ├── ST
    │   │       ├── ST_spe
    │   │       └── SA
    │   └── unlabel
    │       ├── U1.Book32
    │       ├── U2.TextVQA
    │       └── U3.STVQA
    ├── validation
    │   ├── 1.SVT
    │   ├── 2.IIIT
    │   ├── 3.IC13
    │   ├── 4.IC15
    │   ├── 5.COCO
    │   ├── 6.RCTW17
    │   ├── 7.Uber
    │   ├── 8.ArT
    │   ├── 9.LSVT
    │   ├── 10.MLT19
    │   └── 11.ReCTS
    ├── evaluation
    │   ├── benchmark
    │   │   ├── SVT
    │   │   ├── IIIT5k_3000
    │   │   ├── IC13_1015
    │   │   ├── IC15_2077
    │   │   ├── SVTP
    │   │   └── CUTE80
    │   └── addition
    │       ├── 5.COCO
    │       ├── 6.RCTW17
    │       ├── 7.Uber
    │       ├── 8.ArT
    │       ├── 9.LSVT
    │       ├── 10.MLT19
    │       └── 11.ReCTS 
    ├── WikiText-103.csv (for training LM)
    └── WikiText-103_eval_d1.csv (for training LM)
    ```
