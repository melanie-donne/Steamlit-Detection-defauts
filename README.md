# Streamlit Détection de défauts industriels sur de l'acier (FR)

## Description

Il est d'usage de retrouver des problèmes sur les machines de production et des défauts sur l'acier. Le but de ce projet est de donner une image de l'acier, après quoi nous devons détecter des défauts de segmentation dans l'acier. Créé avec HarDNet pour les modèles de segmentation et rationalisé pour le déploiement de sites Web.
<p align="center">
  <img src="https://user-images.githubusercontent.com/44894678/176181206-a9b90ec5-5e46-4a23-97bd-5fca1be01ce3.png" />
</p>
## Installation

```bash
# Python version 3.7.9 or newer
$ git clone https://github.com/Morue/Steamlit-Detection-defauts.git
$ pip3 install -r requirements.txt
$ python3 download_model.py
```

## Usage

```bash
$ streamlit run app.py
usage: app.py [-h] [-d DEVICE] [-m MODEL]

optional arguments:
  -h, --help            show this help message and exit
  -d DEVICE, --device DEVICE
            Input your device inference between CPU or CUDA
  -d MODEL, --model MODEL
            Path to model onnx

# Example arguments input
$ streamlit run app.py -- --device CUDA --model model.onnx
```

© Developed by [MDI](https://github.com/Morue)



# Steamlit Defect Detection (ENG)

## Description

Usually, there is some trouble in production machines and steel defects. The purpose of this project is to give an image of steel, after that we need to detect segmentation defects in steel. Created with HarDNet for segmentation models and streamlit for website deployment.

## Installation

```bash
# Python version 3.7.9 or newer
$ git clone https://github.com/Morue/Steamlit-Detection-defauts.git
$ pip3 install -r requirements.txt
$ python3 download_model.py
```

## Usage

```bash
$ streamlit run app.py
usage: app.py [-h] [-d DEVICE] [-m MODEL]

optional arguments:
  -h, --help            show this help message and exit
  -d DEVICE, --device DEVICE
            Input your device inference between CPU or CUDA
  -d MODEL, --model MODEL
            Path to model onnx

# Example arguments input
$ streamlit run app.py -- --device CUDA --model model.onnx
```



