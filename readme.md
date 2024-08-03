## Quick Links

- [Installation](#installation)
- [Usage](#usage)
- [Data](#data)
- [Acknowledgements](#acknowledgements)
- [Contact](#contact)

## Installation

For training, a GPU is recommended to accelerate the training speed.

### PyTroch

The code is based on PyTorch 1.6+. You can find tutorials [here](https://pytorch.org/tutorials/).

### Dependencies

The code is written in Python 3.7.7. Its dependencies are summarized in the file ```requirements.txt```. 

```
transformers==4.11.3
tensorboardX==2.4
TorchCRF==1.1.0
wandb==0.12.1
torchvision==0.8.2
torch==1.7.1
scikit-learn==1.0
seqeval==1.2.2
```

You can install these dependencies like this:

```
pip3 install -r requirements.txt
```

## Usage

* Run the full model on MRE dataset with default hyperparameter settings<br>

```sh run_single.sh```<br>


## Data

### Format

Each dataset is a folder under the ```./data``` folder:

```
./data
└── RE_data
    ├── img_org  
    ├── img_vg	#object
    ├── txt    #text data
    |    ├── mre_dev_dict.pth
    |    ├── mre_test_dict.pth
    |    ├── mre_train_dict.pth
    |    ├── ours_test.txt
    |    ├── ours_train.txt
    |    └── ours_val.txt
    └── ours_rel2id.json

```

each line in `txt/ours_{split}.txt` includes

- token ：input tokens of origin text
- h：first object and its position in text
- t： second object and its position in text
- img_id：the image path is `img_org/{split}/{img_id}`
- relation： the relation between h and t
- caption：a list includes 10 retrieval captions
- entity ： a list includes 11 entities

