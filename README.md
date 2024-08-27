# [Pattern Recognition] Exposure Difference Network for Low-light Image Enhancement
### [Paper]() | [Code](https://github.com/Tanyjiang/ExdNet)

**Exposure Difference Network for Low-light Image Enhancement**
<br>_Shengqin Jiang, Yongyue Mei, Peng Wang, Qingshan Liu_<br>
Pattern Recognition, 156, 110796.

## Overall

Low-light image enhancement aims to simultaneously improve the brightness and contrast of low-light images and recover the details of the visual content. This is a challenging task that makes typical data-driven methods suffer, especially when faced with severe information loss in extreme low-light conditions. In this work, we approach this task by proposing a novel exposure difference network. The proposed network generates a set of possible exposure corrections derived from the differences between synthesized images under different exposure levels, which are fused and adaptively combined with the raw input for light compensation. By modeling the intermediate exposure differences, our model effectively eliminates the redundancy existing in the synthesized data and offers the flexibility to handle image quality degradation resulting from varying levels of inadequate illumination. To further enhance the naturalness of the output image, we propose a global-aware color calibration module to derive low-frequency global information from inputs, which is further converted into a projection matrix to calibrate the RGB output. Extensive experiments show that our method can achieve competitive light enhancement performance both quantitatively and qualitatively.

## Get Started
### Dependencies and Installation
- Python 3.10
- Pytorch 1.11.0

Clone Repo
```
git clone https://github.com/Tanyjiang/ExdNet.git
```


### Dataset
You can refer to the following links to download the datasets:
[LSRW](https://pan.baidu.com/s/1XHWQAS0ZNrnCyZ-bq7MKvA) (code: wmrr), and
[RELLISUR](https://zenodo.org/records/5234969).

### Test

1. Test the model


```bash
python test.py 

```

You can check the output in `result_img`.

### Train
1.Train the network.
```bash
python train.py 
```

## Citation
If you find our work useful for your research, please cite our paper:
```
@article{jiang2024exposure,
  title={Exposure difference network for low-light image enhancement},
  author={Jiang, Shengqin and Mei, Yongyue and Wang, Peng and Liu, Qingshan},
  journal={Pattern Recognition},
  volume={156},
  pages={110796},
  year={2024},
  publisher={Elsevier}
}
```

## Acknowledgement
This project is based on [IAT](https://github.com/cuiziteng/Illumination-Adaptive-Transformer). Thanks for these awesome codes!

