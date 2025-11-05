# PRISM
PRISM: Accurate Spatial Domain Prediction by Multi-View Integration of Cellular Morphology and scRNA-seq
## Table of contents
- [Network diagram](#diagram)
- [Installation](#Installation)
- [Usage](#Usage)
- [Citation](#Citation)
- [Contact](#contact)

## <a name="diagram"></a>Network diagram

## <a name="Installation"></a>Installation
**Environment requirements**:  
CroSP requires Python 3.9.x and Pytorch.   
For example, we suggest to install the dependencies in a conda environment.  

```
conda create -n PRISM
conda activate PRISM
```
and then you can use pip to install the following dependencies within the CroSP environment.
- python==3.9.18
- torch==1.13.0+cu116 
- torch_scatter==2.1.1
- torch_sparse==0.6.17
- Pillow==9.5.0
- pandas==1.5.3
- scanpy==1.9.2
- pot==0.9.1
- scikit-learn==1.2.1
- rpy2==3.4.1
- R==4.0.3
## <a name="Usage"></a>Usage 
- #### CroSP on DLPFC from 10x Visium.

1. python analyze_DLPFC.py 

2. Further analysis and drawing:  Tutorial/analyze_DLPFC.ipynb

  Links to 151674 result files that can be used in the tutorial：
## <a name="Citation"></a>Citation
## <a name="contact"></a>Contact
Feixiong Cheng chengf@ccf.org Genomic Medicine Institute, Lerner Research Institute, Cleveland Clinic, Cleveland, OH 44195, USA<br> 
Xiangxiang Zeng xzeng@hnu.edu.cn College of Computer Science and Electronic Engineering, Hunan University, Changsha, Hunan, 410082, China<br> 
Junlin Xu xjl@hnu.edu.cn School of Computer Science and Technology, Wuhan University of Science and Technology, Wuhan, Hubei 430065, China

