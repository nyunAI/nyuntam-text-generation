# Nyuntam Text Generation
Nyuntam Text Generation contains the sota compression methods and algorithms to achieve efficiency on text-generation tasks (primarily operated on large language models).
This module implements model efficiency mixins via 
- pruning
- quantization
- accelerations with tensorrtllm


## Installation
Installation can be performed either via installing requirements in a virtual environment or by utilizing our docker images. To quickly run Kompress for experimentation and usage, utilize Nyun CLI to get Kompress running in no time. For contributing to Nyuntam build docker containers from the available docker image or create a virtual enviroment from the provided requirements.txt.

### Nyunzero CLI
The recommended method to install and use nyuntam is via the nyunzero-cli. Further details can be found here : [NyunZero CLI](https://github.com/nyunAI/nyunzero-cli)

### Git + Docker
Nyuntam (Kompress) can also be used by cloning the repository and pulling hhe required docker. 

1. **Git Clone** : First, clone the repository to your local machine:
    ```bash
    $ git clone --recursive https://github.com/nyunAI/nyuntam.git
    $ cd nyuntam
    ```

2. **Docker Pull**: Next, pull the corresponding docker container(s) and run it :

[(list of nyunzero dockers)](https://hub.docker.com/?search=nyunzero)

    ```bash 
    $ docker pull <docker>

    $ docker run -it -d --gpus all -v $(pwd):/workspace <docker_image_name_or_id> bash 
    ```

<span style="color:red">**NOTE:**</span> 
- nvidia-container-toolkit is expected to be installed before the execution of this
- all docker mount tags and environment tags holds (add hf tokens for access to gated repos within the dockers)


### Git + virtual environment

Nyuntam can also be used by cloning the repository and setting up a virtual environment. 

1. **Git Clone** : First, clone the repository to your local machine:
    ```bash
    $ git clone --recursive https://github.com/nyunAI/nyuntam.git
    $ cd nyuntam
    ```

2. **Create a virtual environment using Venv**
   ```sh
   python3 -m venv {ENVIRONMENT_NAME}
   source {ENVIRONMENT_NAME}/bin/activate
   ```

3. **Pip install requirements**
   ```sh
   pip install -r nyuntam-text-generation/requirements.txt
   ```

   **note:** for tensorrtllm, we recommend using the dockers directly.

## Usage 

### Setting up the YAML files
If the dataset and models weights exist online (huggingface hub) then the experiments can be started withing no time. Kompress requires a recipes with all the required hyperparameters and arguments to compress a model.
find a set of [compression scripts here](https://github.com/nyunAI/nyuntam-text-generation/tree/main/scripts)

The major hyperparameters are metioned below : 

- ***Dataset***
```yaml
DATASET_NAME: "wikitext"
DATASET_SUBNAME: "wikitext-2-raw-v1"
DATA_PATH: ""  # custom data path (loadable via datasets.load_from_disk)
TEXT_COLUMN: "text" # if multiple, separate by comma e.g. 'instruction,input,output'
SPLIT: "train"
FORMAT_STRING: # format string for multicolumned datasets

```

- ***Model***

```yaml
MODEL: Llama-3
MODEL: "meta-llama/Meta-Llama-3-8B"  # hf repo id's
CUSTOM_MODEL_PATH: ""
```

for details on dataset and model configurations checkout [nyuntam-docs/nyuntam_text_generation](https://nyunai.github.io/nyuntam-docs/nyuntam_text_generation/)

### Run command
```sh
python main.py --yaml_path {path/to/recipe}
```

This command runs the main file with the configuration setup in the recipe.
