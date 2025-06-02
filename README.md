# Joint Spectrum Sensing and Resource Allocation for OFDMA-based Underwater Acoustic Communications

Source code for the paper ["Joint Spectrum Sensing and Resource Allocation for OFDMA-based Underwater Acoustic Communications"](https://ieeexplore.ieee.org/document/10962227).
## Requirements
### BELLHOP
Install AcousticToolbox from "http://oalib.hlsresearch.com/AcousticsToolbox/".

GFortran is also required.

Go to `at` and run the following commands to finish installation.
```commandline
	make clean
	make
	make install
```

(See the [Acoustics Toolbox doc](http://oalib.hlsresearch.com/AcousticsToolbox/).)

## Training
The codes are located at `a3c_24/`.

Run configurations are located at `a3c_24/configs.py`.

After setting the configurations, run `a3c_24/main.py`.

## Acknowledgments
I have referred to the following repo:
- [ikostrikov/pytorch-a3c](https://github.com/ikostrikov/pytorch-a3c)

## Citation
```
@ARTICLE{10962227,
  author={Kim, Minwoo and Choi, Youngchol and Kim, Yeongjun and Seo, Eojin and Yang, Hyun Jong},
  journal={IEEE Communications Letters}, 
  title={Joint Spectrum Sensing and Resource Allocation for OFDMA-based Underwater Acoustic Communications}, 
  year={2025},
  volume={},
  number={},
  pages={1-1},
  keywords={Sensors;OFDM;Interference;Symbols;Signal to noise ratio;Resource management;Vectors;Deep reinforcement learning;Reliability;Underwater acoustics;Spectrum sensing;underwater acoustic communication;deep reinforcement learning;OFDMA},
  doi={10.1109/LCOMM.2025.3559513}}

```