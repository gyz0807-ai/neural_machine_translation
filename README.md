# Neural Machine Translation

## How to use

1. Install environment  
`conda env create -f environment.yml`  

2. Use the installed environment  
`conda activate nmt_tf`

3. Download dataset  
Source of dataset: http://www.manythings.org/anki/  
`sh download_data.sh`

4. (Optional) Pull up the helper document for hyperparameters  
`python main.py --help`

5. Train the model  
`python main.py`  
The default is to reparse the dataset everytime we train the model. However, once we've parsed the dataset and want to use the same dataset for training a different model, we can skip the reparsing process in this way  
`python main.py --reparse_vocab 0` 

### Translation Demo
Please refer to the [demo notebook](https://github.com/gyz0807-ai/neural_machine_translation/blob/master/demo.ipynb).