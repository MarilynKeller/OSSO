# OSSO: Obtaining Skeletal Shape from Outside (CVPR 2022)

This repository contains the official implementation of the Skeleton Inference from:

**OSSO: Obtaining Skeletal Shape from Outside** <br>*Marilyn Keller, Silvia Zuffi, Michael J. Black and Sergi Pujades* <br>[Full paper](https://download.is.tue.mpg.de/osso/OSSO_supmat.pdf) | [Project website](https://osso.is.tue.mpg.de/index.html#Dataset) 


Given a body shape with SMPL or STAR topology, we infer the underlying skeleton.
![teaser](./figures/skeleton_results.png)


## Installation
Please follow the installation instruction in [installation.md](installation.md) to setup all the required packages and models.


## Run skeleton inference

The skeleton can be inferred either from a SMPL or STAR mesh.

``` 
python main.py  --mesh_input data/demo/body.ply --gender female -D
```


If the reposing does not converge, rerun the reposing with more iterations:
```
python main.py -g female --pkl_input data/demo/rp_ellie_posed_008_0_0_star.pkl -p
```

The intermediate and result meshes will be saved in the `out/tmp` folder.


## License

Please refer for LICENSE.txt for using this code.


## Citation

If you find this Model & Software useful in your research, please:

```
@inproceedings{Keller:CVPR:2022,  
  title = {OSSO: Obtaining Skeletal Shape form Outside},  
  author = {Keller, Marylin and Zuffi, Silvia and Black, Michael J. and Pujades, Sergi},  
  booktitle = {IEEE Conference on Computer Vision and Pattern Recognition (CVPR)},  
  pages = {},  
  publisher = {IEEE Computer Society},  
  year = {2022},  
}
```

## Acknowledgements

OSSO uses the [Stitched Puppet](https://stitch.is.tue.mpg.de/) by Silvia Zuffi and the body model [STAR](https://github.com/ahmedosman/STAR) by Ahmed Osman

## License

This code and model are available for non-commercial scientific research purposes as defined in the LICENSE file. By downloading and using the code and model you agree to the terms in the LICENSE.

## Contact

For more questions, please contact osso@tue.mpg.de

For commercial licensing, please contact ps-licensing@tue.mpg.de