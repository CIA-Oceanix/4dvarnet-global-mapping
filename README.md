# 4DVarNet

## Prerequisite
- git
- conda

## Install
### Install project dependencies
```
git clone https://github.com/CIA-Oceanix/4dvarnet-global-mapping.git
cd 4dvarnet-global-mapping
conda install -c conda-forge mamba
conda create -n 4dvarnet-gm
conda activate 4dvarnet-gm
mamba env update -f environment.yaml
pip install ocean4dvarnet
```

### Download example data
From the directory
```
wget https://s3.eu-central-1.wasabisys.com/sla-data-registry/6d/206c6be2dfe0edf1a53c29029ed239 -O data/natl_gf_w_5nadirs.nc
```

## Run
The model uses hydra see [#useful-links]
```
python main.py xp=glo12-sla
```

## Inference
```
cd inference
edit the checkpoint path of params.yaml with your checkpoint
dvc repro
```

## Saved weights:

<!-- You can find pre-trained weights [here](https://s3.eu-central-1.wasabisys.com/melody/quentin_cloud/starter_big_mod_07a265.ckpt) -->

The test metrics of this model are ([see here for the details])(https://github.com/ocean-data-challenges/2023a_SSH_mapping_OSE):


## Useful links:
- [Hydra documentation](https://hydra.cc/docs/intro/)
- [Pytorch lightning documentation](https://pytorch-lightning.readthedocs.io/en/stable/index.html#get-started)
- 4DVarNet papers:
	- Fablet, R.; Amar, M. M.; Febvre, Q.; Beauchamp, M.; Chapron, B. END-TO-END PHYSICS-INFORMED REPRESENTATION LEARNING FOR SA℡LITE OCEAN REMOTE SENSING DATA: APPLICATIONS TO SA℡LITE ALTIMETRY AND SEA SURFACE CURRENTS. ISPRS Annals of the Photogrammetry, Remote Sensing and Spatial Information Sciences 2021, V-3–2021, 295–302. https://doi.org/10.5194/isprs-annals-v-3-2021-295-2021.
	- Fablet, R.; Chapron, B.; Drumetz, L.; Mmin, E.; Pannekoucke, O.; Rousseau, F. Learning Variational Data Assimilation Models and Solvers. Journal of Advances in Modeling Earth Systems n/a (n/a), e2021MS002572. https://doi.org/10.1029/2021MS002572.
	- Fablet, R.; Beauchamp, M.; Drumetz, L.; Rousseau, F. Joint Interpolation and Representation Learning for Irregularly Sampled Satellite-Derived Geophysical Fields. Frontiers in Applied Mathematics and Statistics 2021, 7. https://doi.org/10.3389/fams.2021.655224.

