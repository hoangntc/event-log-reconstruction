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
|   |   helpdesk.csv
| 
|--- data_preprocessing
|   |   induce_missing_data.ipynb
|   │   preprocess_variables.ipynb
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
|
|-- base_model
|   |   dummy_imputation.ipynb
|


```
### Reference


1. Install requirement

- Install pytorch: ```conda install pytorch torchvision -c soumith```
- ```pip install -r requirements.txt```

2. Run ```induce_missing_data.ipynb```
2. Run ```preprocess_variables.ipynb```
3. Run ```AE.ipynb``` or ```VAE.ipynb```

