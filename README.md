# aigs-nsd-reconstruction
AIGS final project repository


# How to run
1. Install python version 3.11.10
```bash
brew install pyenv # macosx
choco install pyenv # windows powershell
pyenv install 3.11.10
```

2. Create and activate virtual environment
```bash
pyenv exec python3 -m venv .venv-aigs
source .venv/bin/activate # macosx
.venv/Scripts/activate # windows powershell
```

3. Install dependencies
- for models
    ```bash
    pip install -r requirements_models.txt
    ```

3. b) for the datasets
    ```bash
    pip install -r requirements_nsd.txt
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
