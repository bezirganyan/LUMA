# LUMA: Learning from Uncertain and Multimodal Data
![Screenshot-20240708174237-1456x489](https://github.com/bezirganyan/LUMA/assets/16824193/f97782ad-d28e-4559-b32b-f0660b4d46cd)

## Overview

LUMA is a multimodal dataset designed for benchmarking multimodal learning and multimodal uncertainty quantification. This dataset includes audio, text, and image modalities, enabling researchers to study uncertainty quantification in multimodal classification settings.

Please find the Dataset Paper preprint [here](https://arxiv.org/abs/2406.09864).

## Dataset Summary

LUMA consists of:
- **Audio Modality**: `wav` files of people pronouncing the class labels of the selected 50 classes.
- **Text Modality**: Short text passages about the class labels, generated using large language models.
- **Image Modality**: Images from a 50-class subset from CIFAR-10/100 datasets, as well as generated images from the same distribution.

The dataset allows controlled injection of uncertainties, facilitating the study of uncertainty quantification in multimodal data.

## Getting Started

### Prerequisites

- Anaconda / Miniconda
- Git

### Installation
Clone the repository and navigate into the project directory:

```bash
git clone https://github.com/bezirganyan/LUMA.git 
cd LUMA
```
Install and activate the conda enviroment
```bash
conda env create -f environment.yml
conda activate luma_env
```

Make sure you have git-lfs installed (https://git-lfs.com), it will be automatically installed by conda if you did previous steps. Then perform:
```
git lfs install
```
Download the dataset under the `data` folder (you can also choose other folder names, and updated config files, `data` folder is the default in the default configurations)
```bash
git clone https://huggingface.co/datasets/bezirganyan/LUMA data
```

### Usage
The provided Python tool allows compiling different versions of the dataset with various amounts and types of uncertainties.

To compile the dataset with specified uncertainties, create or edit the configuration file similar to the files in `cfg` directory, and run:
```
python compile_dataset.py -c <your_yaml_config_file>
```

### Usage in Deep Learning models
After compiling the dataset, you can use the `LUMADataset` class from the `dataset.py` file. Example of the usage can be found in `run_baselines.py` file.

### Unprocessed & Unaigned data
If you want to get all the data (without sampling or noise) without alignment (to perform your own alignment, or use the data without alignment for other tasks) you can run the following command:

```
python get_unprocessed_data.py
```

If you use the dataset, please cite our paper preprint with:
```
@article{bezirganyan2024lumabenchmarkdatasetlearning,
      title={LUMA: A Benchmark Dataset for Learning from Uncertain and Multimodal Data}, 
      author={Grigor Bezirganyan and Sana Sellami and Laure Berti-Équille and Sébastien Fournier},
      year={2024},
      eprint={2406.09864},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2406.09864}, 
}
```

## Contact

* <a href="mailto:grigor.bezirganyan98@gmail.com">Grigor Bezirganyan</a>
* <a href="mailto:sana.sellami@univ-amu.fr">Sana Sellami</a>
* <a href="mailto:laure.berti@ird.fr">Laure Berti-Équille</a>
* <a href="mailto:sebastien.fournier@univ-amu.fr">Sébastien Fournier</a>
