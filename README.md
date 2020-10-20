# Enhancing-the-Alignment-between-Target-Words-and-Corresponding-Frames-for-Video-Captioning

This is the Theano code for our paper "Enhancing the Alignment between Target Words and Corresponding Frames for Video Captioning", which has been accepted for publication in Pattern Recognition. Tu, Yunbin, et al. [The version of pre-proof is available in this link.](https://www.sciencedirect.com/science/article/pii/S0031320320305057) 


## We illustrate the training details as follows:

## usage

### Installation

Firstly, Clone our repository:
```
$ git clone https://github.com/tuyunbin/Enhancing-the-Alignment-between-Target-Words-and-Corresponding-Frames-for-Video-Captioning.git
```

Here, the folder data contains 8 pkl files needed to train and test the model.
### Dependencies
Ubantu; Python 2.7; A Titan xp. The results may be differnt when using different GPUs.

For Theano, I recommend you to install it via anaconda:
```
$ conda install theano pygpu
```

[coco-caption](https://github.com/tylin/coco-caption). Install it by simply adding it into your $PYTHONPATH.

[Jobman](http://deeplearning.net/software/jobman/install.html). After it has been git cloned, please add it into $PYTHONPATH as well.

Finally, you will also need to install [h5py](https://pypi.org/project/h5py/), since we will use hdf5 files to store the preprocessed features.

### Video Datas and Pre-extracted Features on MSVD Dataset.

[The pre-processed MSVD dataset used in our paper are available at this link](https://drive.google.com/file/d/1LyfN6s8xKju-iad8M3OvaqFeoPT4aQV9/view?usp=sharing), [and there is the baidu cloud link.](https://pan.baidu.com/s/1o-RlsSaLlxYJHzkhhKwQxw)

[The extracted ResNet-152 features on MSVD can be download at this link.](https://drive.google.com/file/d/15iEsdfPe1JwhEKlVjiunB8mj7B5BOOSh/view?usp=sharing)


### Train your own model
Here, you need to set 'False' with reload in ```config.py```.

Now ready to launch the training
```
$ THEANO_FLAGS=mode=FAST_RUN,device=cuda0,floatX=float32 python train_model.py
```

If you find this helps your research, please consider citing:
```
@article{tu2020enhancing,
  title={Enhancing the Alignment between Target Words and Corresponding Frames for Video Captioning},
  author={Tu, Yunbin and Zhou, Chang and Guo, Junjun and Gao, Shengxiang and Yu, Zhengtao},
  journal={Pattern Recognition},
  pages={107702},
  year={2020},
  publisher={Elsevier}
}
```

### Notes

Running train_model.py for the first time takes much longer since Theano needs to compile for the first time lots of things and cache on disk for the future runs. You will probably see some warning messages on stdout. It is safe to ignore all of them. Both model parameters and configurations are saved (the saving path is printed out on stdout, easy to find). The most important thing to monitor is train_valid_test.txt in the exp output folder. It is a big table saving all metrics per validation. 

### Contact
My email is tuyunbin1995@foxmail.com

Any discussions and suggestions are welcome!
