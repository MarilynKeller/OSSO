# Installation

Clone this repository:
```
git clone https://github.com/MarilynKeller/OSSO
cd OSSO
```

# Environment

Create a virtual environment and activate it:
```
python3.8 -m venv skel_venv # create a new venv
source skel_venv/bin/activate # activate it
```

Install the required packages:
```
pip install -r requirements.txt
```

## Download the skeleton model

From OSSO folder, execute:
`git clone https://github.com/silviazuffi/gloss_skeleton.git`

You should have the following folder structure:

OSSO/
├── data/
├── figures/
├── gloss_skeleton/
│   ├── gloss/
│   └── models/
└── osso/
    ├── star_model/
    └── utils/


## Install MPI Mesh package

With the virtual environment sourced, run:
```
git clone https://github.com/MPI-IS/mesh.git
cd /path/to/mesh
make all
```
The compilation takes some minutes.

## Download the models

Download STAR from the website https://star.is.tue.mpg.de/downloads. You will need to create an account. 

Place the extracted files in the `data` folder.

Likewise, download the skeleton data from https://osso.is.tue.mpg.de/download.php, and place it in the `data` folder.

Your data folder should look like this:

```
data/
├── demo/
├── loss/
├── skeleton/
│   ├── betas_regressor_female.pkl
│   ├── betas_regressor_male.pkl
│   ├── ldm_indices.pkl
│   ├── ldm_regressor_female.pkl
│   ├── ldm_regressor_male.pkl
│   ├── lying_pose_female.pkl
│   ├── lying_pose_male.pkl
│   ├── skeleton_model.pkl
│   ├── skeleton_pca_female.pkl
│   └── skeleton_pca_male.pkl
└── star/
    ├── female/
    │   └── model.npz
    ├── LICENSE.txt
    ├── male/
    │   └── model.npz
    └── neutral/
        └── model.npz
```



## Install OSSO
```
cd path/to/OSSO
python setup.py install
```
