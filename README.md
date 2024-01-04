# E2RealSR 

### 

> **E2-RealSR:Efficient and Effective Real-World Super-resolution Network Based on Partial Degradation Modulation** <br>
> Jiajun Zhang, [Yuanbo Zhou](https://github.com/fzuzyb). <br>
> In The Visual Computer.



#### Getting started

- Clone this repo.
```bash
cd E2RealSR
```

- Install dependencies. (Python 3 + NVIDIA GPU + CUDA. Recommend to use Anaconda)
```bash
pip install -r requirements.txt
```

- Prepare the training and testing dataset by following this [instruction](datasets/README.md).

#### Training

First, check and adapt the yml file ```options/train/E2RealSR/train_E2RealSR.yml```, then

- Single GPU:
```bash
PYTHONPATH="./:${PYTHONPATH}" CUDA_VISIBLE_DEVICES=0 python E2RealSR/train.py -opt options/train/E2RealSR/train_E2RealSR.yml --auto_resume
```

- Distributed Training:
```bash
PYTHONPATH="./:${PYTHONPATH}" CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --nproc_per_node=4 --master_port=4335 E2RealSR/train.py -opt options/train/E2RealSR/train_E2RealSR.yml --launcher pytorch --auto_resume

```

Training files (logs, models, training states and visualizations) will be saved in the directory ```./experiments/{name}```


#### Testing

First, check and adapt the yml file ```options/test/DASR/test_DASR.yml```, then run:
```bash
PYTHONPATH="./:${PYTHONPATH}" CUDA_VISIBLE_DEVICES=0 python basicsr/test.py -opt options/test/DASR/test_E2RealSR.yml
```
Evaluating files (logs and visualizations) will be saved in the directory ```./results/{name}```

if you want to test parameters and Flops:
```bash
PYTHONPATH="./:${PYTHONPATH}" CUDA_VISIBLE_DEVICES=3 python E2RealSR/parameters.py -opt options/test/E2RealSR/test_E2RealSR.yml
```
### License

This project is released under the Apache 2.0 license.



### Acknowledgement
This project is built based on the excellent [BasicSR](https://github.com/xinntao/BasicSR) project.
This project is built based on the excellent [DASR](https://github.com/csjliang/DASR) project.

### Contact
Should you have any questions, please contact me via `jjiajunzhang@gmail.com`.
