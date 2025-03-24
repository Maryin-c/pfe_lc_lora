import kaggle
# you need to configure API key through https://www.kaggle.com/docs/api
kaggle.api.dataset_download_files('rickyyyyyyy/torchvision-stanford-cars', path="./data/", unzip=True)
