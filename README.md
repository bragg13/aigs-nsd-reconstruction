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
pyenv exec python3 -m venv .venv-aigs
source .venv-aigs/bin/activate # macosx
.venv-aigs/Scripts/activate # windows powershell
```

3. Install dependencies
    ```bash
    pip install -r requirements.txt
    ```


4. Run the code
- for models
    ```bash
    python3 vae_main.py --config=configs/default.py # for VAE
    python3 ae_main.py --config=configs/default.py # for simple AE
    ```
- for datasets
    ```bash
    ...
    ```

there must be a `results` folder inside of the model project folder to save the results, otherwise it won't work.
