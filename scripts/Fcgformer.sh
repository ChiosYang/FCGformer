export CUDA_VISIBLE_DEVICES=0

model_name=FCGformer

python -u run.py \
  --is_training 1 \
  --root_path ./dataset/AIR-EW/ \
  --data_path AW-NEW.csv \
  --model_id FCGformer \
  --model $model_name \
  --data custom \
  --features MS \
  --seq_len 96 \
  --pred_len 24 \
  --enc_in 20 \
  --dec_in 20 \
  --c_out 20 \
  --des 'Exp' \
  --d_model 512 \
  --d_ff 512 \
  --itr 1 \
  --batch_size 2048 \
  --patience 3 \
  --dropout 0.1 \
  --e_layers 4 \
  --n_heads 8 \
  --d_layers 1 \
  --num_workers 1 \
  --learning_rate 0.0001 \
  --dilation_channel 28 \
  --residual_channels 28 \
  --features_len 20 \
  --lradj 'type1'

