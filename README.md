# Variational Graph Auto-Encoders
The [VGAE-based model](https://arxiv.org/pdf/1611.07308.pdf) is an unsupervised-based model which takes as enconder the [Graph Convolutional Network (GCN)](https://tkipf.github.io/graph-convolutional-networks/) model. VGAE extracts
latent variables from a given connected graph, then the decoding process is realized through a simple inner product.

## 1. Files
- **train.py**: Main file to train the model.
- **src/model.py**: Contains the ```VGAE``` class.
- **src/utils.py**: Contains functions to load raw files and transform them into a graph format.
- **src/parser.py**: Contains parser values.
- **benchmarks/**: Holds datasets as csv format. 

### 1.1 Datasets
All datasets will be stored in the directory ``benchmarks/``. Inside ``benchmark/`` you need to store the directory of your dataset such as ``benchmarks/[your_dataset_directory]/``, inside of ``[your_dataset_directory/]`` you have to store the graph file with csv format such as: 
``dataset.csv``.

The ```dataset.csv``` needs to meet the following format:
```
  node1 node2
  node2 node3
  node3 node4
  ...
```

## 2. Installation
It is recommended to use a virtual environment such as ``pipenv`` to install the dependencies and run the model. If you do not have installed ``pipenv`` yet, just type: ``pip install pipenv``. Then, you need to launch the virtual environment by typing:

```
pipenv shell
```

Once the virtual environment has been launched, you need to install the dependencies:

```
pipenv install
```

in case the previous command does not work, try to type the next command:

```
pipenv install --ignore-pipfile
```

## 3. How to run
To run the model you need to use the file ``train.py`` such as:
```
python train.py [--embedding_size EMBEDDING_SIZE] [--epochs EPOCHS]
                [--learning_rate LEARNING_RATE] [--neurons NUM_NEURONS]
                [--dataset DATASET] [--directory DIRECTORY]
                [--test_size TEST_SIZE]
```

## 4. Further work
The  model can be improved by adding:
- A function to visualize embeddings
- A function to evaluate results

**Note**: Feel free to send me a pull request if you want to collaborate in this proyect.
