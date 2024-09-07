# 2024 IMBD 透過對物理量的預測了解物體的品質狀態
We use conda to build our enviornment. The following will provide step-by-step instructions on how to set up the environment and run the code.
## Directory structure
```bash
113061_Source/
├── src/
│   ├── main.py
│   └── utils.py
├── environment.yml
└── README.md
```
## Installation environment / Execution environment / Package selection
* Conda version: 23.1.0
* Python version: 3.6.13
* Main libraries used: scikit-learn, pandas, numpy, math, pickle, matplotlib

## Environment setup
Unzip 113061_Source.zip and move into the folder.
```bash
unzip 113061_Source.zip
cd 113061_Source/
```
An environment can be created with all the Python dependencies.
```bash
conda env create -f environment.yml
```
Activate environment to run .py files.
```bash
conda activate 113061_Source
```
## Execution method
### 1. Transfer CSV files to longformat
Convert the CSV files under various resistor and voltage conditions into long format for easier reading.
```bash
python src/longformat.py
```
### 2. Prepare training data
Divide the dataframe under each resistor and voltage condition into the first 50 timesteps and the following 3950 timesteps, while also separating the 10 experiments individually. Store them in a dictionary structure.
```bash
pyhon src/preprocessing.py
```
### 3. Train MLP model
```bash
python train.py
```
