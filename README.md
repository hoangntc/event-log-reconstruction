# Event log Reconstruction

### Directory structure:

```
event-log-reconstruction
│   README.md
│   requirement.txt
|
|--- data: original dataset
│   │   bpi_2012.csv
|   |   bpi_2013.csv
| 
|--- data_preprocessing
|   |   induce_missing_data.py
|   │   preprocess_variables.py
|   |   real_log_preprocessing.sh
|
|--- utils
|   |   utils.py
|   |   models.py
|
|--- input: preprocessed data
|
|--- experiment
|   |   output
|   |   AE.ipynb
|   |   VAE.ipynb
|   |   LSTMAE.ipynb
|
|-- base_model
|   |   dummy_imputation.ipynb
|   |   dummy_imputation_full.ipynb
|   |   statistical_description.ipynb
|


```
### Reference


1. Install requirement

- Install pytorch: ```conda install pytorch torchvision -c soumith```
- ```pip install -r requirements.txt```

2. ```cd data_preprocessing```
3. For preprocessing: ```source real_log_preprocessing.sh```
4. For training and evaluating: Run ```AE.ipynb``` or ```VAE.ipynb```

