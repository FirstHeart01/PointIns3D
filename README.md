## Code structure

### Dependencies :memo:
The main dependencies of the project are the following:
```yaml
python: 3.7.x
cuda: 11.6
```
You can set up a conda environment as follows
```
conda create --name=pointins3d python=3.7
conda activate pointins3d

conda update -n base -c defaults conda

[//]: # (conda install openblas-devel -c anaconda)

[//]: # (pip install torch==1.12.1+cu116 torchvision==0.13.1+cu116 torchaudio==0.12.1 --extra-index-url https://download.pytorch.org/whl/cu116)
conda install pytorch==1.12.1 torchvision==0.13.1 torchaudio==0.12.1 cudatoolkit=11.6 -c pytorch -c conda-forge
[//]: # (pip install ninja==1.10.2)
pip install spconv-cu116
pip install -r requirements.txt

sudo apt install build-essential python3-dev libopenblas-dev

python setup.py build_ext develop

```
higher environment, python=3.11, cuda=11.7, pytorch=2.0
```
conda create -n pointins3d python=3.11
conda activate pointins3d

conda update -n base -c defaults conda

pip3 install numpy --pre torch torchvision torchaudio --force-reinstall --index-url https://download.pytorch.org/whl/nightly/cu117
pip install spconv-cu117

pip install -r requirements.txt

sudo apt install build-essential python3-dev libopenblas-dev libsparsehash-dev
python setup.py build_ext develop
```
## Training
### Training STPLS3D dataset
```
./tools/dist_train.sh configs/pointins3d_stpls3d_backbone.yaml 4
./tools/dist_train.sh configs/pointins3d_stpls3d.yaml 4
```

## Testing
### Testing STPLS3D dataset
```
./tools/dist_test.sh configs/pointins3d_stpls3d.yaml work_dirs/pointins3d_stpls3d/latest.pth 4 --out ./results/pointinsd_stpls3d
```