# [Pattern Recognition] Exposure Difference Network for Low-light Image Enhancement
### [Paper]() | [Code](https://github.com/Tanyjiang/ExdNet)

**Exposure Difference Network for Low-light Image Enhancement**
<br>_Shengqin Jiang, Yongyue Mei, Peng Wang, Qingshan Liu_<br>
In Pattern Recognition

## Overall
![Framework](images/network.png)


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
@article{jiang2024low,
  title={Exposure Difference Network for Low-light Image Enhancement},
  author={Jiang, Shengqin and Mei, Yongyue and Wang, Peng and Liu, Qingshan},
  journal={},
  year={2024}
}
```

## Contact
If you have any questions, please feel free to contact us via xx.
