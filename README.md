# Causal Disentangled Recommendation Against Preference Shifts
This is the pytorch implementation of our paper
> Causal Disentangled Recommendation Against Preference Shifts

## Environment
- Anaconda 3
- python 3.7.3
- pytorch 1.4.0
- numpy 1.16.4

## Usage
### Data
The experimental data are in './data' folder, including Yelp, Book and Electronics.

### Training
```
python main.py --dataset=$1 --lr=$2 --batch_size=$3 --mlp_dims=$4 --mlp_p1_dims=$5 --mlp_p2_dims=$6 --a_dims=$7 --z_dims=$8 --c_dims=$9 --tau=$10 --T=$11 --lam1=$12 --lam2=$13 --lam3=$14 --regs=$15 --bn=$16 --dropout=$17 --std=$18 --add_T=$19 --log_name=$20 --gpu=$21 --cuda
```
or use run.sh
```
sh run.sh model_name dataset lr batch_size mlp_dims mlp_p1_dims mlp_p2_dims a_dims z_dims c_dims tau T lam1 lam2 lam3 regs bn dropout std add_T log_name gpu_id
```
- The log file will be in the './code/log/' folder. 
- The explanation of hyper-parameters can be found in './code/main.py'. 
- The default hyper-parameter settings are detailed in './code/hyper-parameters.txt'.

### Inference
Get the results of CDR over three datasets by running inference.py:
```
python inference.py --dataset=$1 --T=$2 --add_T=$3 --ckpt=$4 --gpu=$5 --cuda
```
### Examples

1. Train CDR on Yelp:

```
cd ./code
sh run.sh yelp 1e-4 500 [800] [] [] 400 400 2 0.5 3 0.5 1 0.0001 0 1 0.6 1 2 log 0
```

2. Inference on Book:

```
cd ./code
python inference.py --dataset book --T 3 --add_T 3 --ckpt <pre-trained model directory> --gpu 0 --cuda
```
