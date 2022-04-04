# OSSO: Obtaining Skeletal Shape from Outside (CVPR 2022)

This repository contains the official implementation of the skeleton inference from:

**OSSO: Obtaining Skeletal Shape from Outside** <br>*Marilyn Keller, Silvia Zuffi, Michael J. Black and Sergi Pujades* <br>[Full paper](https://download.is.tue.mpg.de/osso/OSSO.pdf) | [Project website](https://osso.is.tue.mpg.de/index.html#Dataset) 


Given a body shape with SMPL or STAR topology (in blue), we infer the underlying skeleton (in yellow).
![teaser](./figures/skeleton_results.png)


## Installation
Please follow the installation instruction in [installation.md](installation.md) to setup all the required packages and models.


## Run skeleton inference

The skeleton can be inferred either from a [SMPL](https://smpl.is.tue.mpg.de/) or [STAR](https://github.com/ahmedosman/STAR) mesh.

``` 
python main.py  --mesh_input data/demo/body_female.ply --gender female -D
```

The final infered skeleton will be saved in the `out` folder and the intermediate meshes in `out/tmp`.


## Citation

If you find this model & software useful in your research, please consider citing:

```
@inproceedings{Keller:CVPR:2022,
  title = {{OSSO}: Obtaining Skeletal Shape from Outside},
  author = {Keller, Marilyn and Zuffi, Silvia and Black, Michael J. and Pujades, Sergi},
  booktitle = {Proceedings IEEE/CVF Conf.~on Computer Vision and Pattern Recognition (CVPR)},
  month = jun,
  year = {2022},
  month_numeric = {6}}
```

## Acknowledgements

OSSO uses the [Stitched Puppet](https://stitch.is.tue.mpg.de/) by Silvia Zuffi and the body model [STAR](https://github.com/ahmedosman/STAR) by Ahmed Osman.

## License

This code and model are available for non-commercial scientific research purposes as defined in the [LICENSE.txt](LICENSE.txt) file.

## Contact

For more questions, please contact osso@tue.mpg.de

For commercial licensing, please contact ps-licensing@tue.mpg.de