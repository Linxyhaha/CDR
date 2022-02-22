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
python main.py --dataset=$1 --lr=$2 --wd=$3 --batch_size=$4 --epochs=$5 --mlp_dims=$6 --mlp_p1_dims=$7 --mlp_p2_dims=$8 --a_dims=$9 --z_dims=$10 --c_dims=$11 --tau=$12 --T=$13 --sigma=$14 --total_anneal_steps=$15 --anneal_cap=$16 --bn=$17 --dropout=$18 --regs=$19 --lam1=$20 --lam2=$21 --std=$22 --w_sigma=$23 --cuda --add_T=$24 --log_name=$25 --gpu=$26
```
or use run.sh
```
sh run.sh model_name dataset lr wd batch_size epochs mlp_dims mlp_p1_dims mlp_p2_dims a_dims z_dims c_dims tau T sigma total_anneal_steps anneal_cap bn dropout regs lam1 lam2 std w_sigma add_T log_name gpu_id
```
- The log file will be in the './code/log/' folder. 
- The explanation of hyper-parameters can be found in './code/main.py'. 
- The default hyper-parameter settings are detailed in './code/hyper-parameters.txt'.

### Inference
Get the results of CDR over three datasets by running inference.py:
```
python inference.py --dataset=$1 --ckpt=$2 --cuda
```
### Examples

1. Train CDR on Yelp:

```
cd ./code
sh run.sh yelp 1e-3 0 500 [800] [] [] 400 400 2 0.5 3 0.1 0 0.5 1 0.6 0 0.0001 1 1 0.5 2 log 0
```

2. Inference on Book:

```
cd ./code
python inference.py --dataset book --ckpt <pre-trained model directory> --cuda
```
