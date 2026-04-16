# Ernie-Image Inference Guide on Ascend

## Install Dependencies

- diffusers
```
git clone https://github.com/huggingface/diffusers.git

cd diffusers

python setup.py install

```

- MindIE-SD
```
git clone https://gitcode.com/Ascend/MindIE-SD.git && cd MindIE-SD
python setup.py bdist_wheel

cd dist
pip install mindiesd-*.whl

```

## Prepare Weights
Download the weights to your local path and configure the weight path in `run_infer_npu.sh`.

## Inference
```
sh run_infer_npu.sh

```
