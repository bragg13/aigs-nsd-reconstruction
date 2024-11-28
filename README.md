# aigs-nsd-reconstruction
AIGS final project repository

# How to run
1. Install python version 3.11.10
```bash
# macosx
brew install pyenv

# windows powershell
Invoke-WebRequest -UseBasicParsing -Uri "https://raw.githubusercontent.com/pyenv-win/pyenv-win/master/pyenv-win/install-pyenv-win.ps1" -OutFile "./install-pyenv-win.ps1"; &"./install-pyenv-win.ps1"

# then
pyenv install 3.11.10 # or 3.11.9 if .10 is not available
```

2. Create and activate virtual environment
```bash
pyenv exec python3 -m venv .venv
source .venv/bin/activate # macosx
.venv-aigs/Scripts/activate # windows powershell
```

3. Install dependencies
    ```bash
    pip install -r requirements.txt
    ```

4. Download datasets
- download coco annotations
```bash
mkdir annotations && cd annotations
wget http://images.cocodataset.org/annotations/annotations_trainval2017.zip
unzip annotations_trainval2017.zip
```

- download algonauts dataset
```bash
...
```

5. Run the code
- for models
    ```bash
    python3 ae_main.py --config=configs/default.py # for simple AE
    ```
- for datasets
    ```bash
    ...
    ```
    there must be a `results` folder inside of the model project folder to save the results, otherwise it won't work.

# How to join the coco and algonauts datasets

The file `nsd_coco.csv`, taken directly from the NSD dataset, contains the following columns:
- `index`: the image id of the coco dataset
- `cocoId`: the image id of the coco dataset
- `cocoSplit`: the image file name of the coco dataset
