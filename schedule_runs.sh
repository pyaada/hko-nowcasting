#ConvLSTM cloudy
# CUDA_VISIBLE_DEVICES=1 python train_hko7.py data/HKO-7/samplers/hko7_cloudy_days_t20_train.txt.pkl data/HKO-7/samplers/hko7_cloudy_days_t20_test.txt.pkl -m CONVLSTM_HKO7 --loss mse
# #PredRNN
# CUDA_VISIBLE_DEVICES=1 python train_hko7.py data/HKO-7/samplers/hko7_cloudy_days_t20_train.txt.pkl data/HKO-7/samplers/hko7_cloudy_days_t20_test.txt.pkl -m PREDRNN_HKO7 --loss mse
# #SimVP
# CUDA_VISIBLE_DEVICES=1 python train_hko7.py data/HKO-7/samplers/hko7_cloudy_days_t20_train.txt.pkl data/HKO-7/samplers/hko7_cloudy_days_t20_test.txt.pkl -m SIMVP_HKO7 --loss mse
# # Earthformer
# CUDA_VISIBLE_DEVICES=1 python train_hko7.py data/HKO-7/samplers/hko7_cloudy_days_t20_train.txt.pkl data/HKO-7/samplers/hko7_cloudy_days_t20_test.txt.pkl -m EARTHFORMER_HKO7 --loss mse --scheduler cosine

# #ConvLSTM rainy
# CUDA_VISIBLE_DEVICES=1 python train_hko7.py data/HKO-7/old_samplers/hko7_rainy_train.pkl data/HKO-7/old_samplers/hko7_rainy_valid.pkl -m CONVLSTM_HKO7 -o checkpoints/hko-7/rainy --loss mse
# #PredRNN rainy
# CUDA_VISIBLE_DEVICES=0 python train_hko7.py data/HKO-7/old_samplers/hko7_rainy_train.pkl data/HKO-7/old_samplers/hko7_rainy_valid.pkl -m PREDRNN_HKO7 -o checkpoints/hko-7/rainy --loss mse
# #SimVP rainy
# CUDA_VISIBLE_DEVICES=1 python train_hko7.py data/HKO-7/old_samplers/hko7_rainy_train.pkl data/HKO-7/old_samplers/hko7_rainy_valid.pkl -m SIMVP_HKO7 -o checkpoints/hko-7/rainy --loss mse
# # Earthformer rainy
# CUDA_VISIBLE_DEVICES=1 python train_hko7.py data/HKO-7/old_samplers/hko7_rainy_train.pkl data/HKO-7/old_samplers/hko7_rainy_valid.pkl -m EARTHFORMER_HKO7 -o checkpoints/hko-7/rainy --loss mse --scheduler cosine

#ConvLSTM FACL cloudy
CUDA_VISIBLE_DEVICES=0 python train_hko7.py data/HKO-7/samplers/hko7_cloudy_days_t20_train.txt.pkl data/HKO-7/samplers/hko7_cloudy_days_t20_test.txt.pkl -m CONVLSTM_HKO7_SIGMOID --loss facl-0.1
#PredRNN FACL
# CUDA_VISIBLE_DEVICES=0 python train_hko7.py data/HKO-7/samplers/hko7_cloudy_days_t20_train.txt.pkl data/HKO-7/samplers/hko7_cloudy_days_t20_test.txt.pkl -m PREDRNN_HKO7_SIGMOID --loss facl-0.1
#SimVP FACL
CUDA_VISIBLE_DEVICES=0 python train_hko7.py data/HKO-7/samplers/hko7_cloudy_days_t20_train.txt.pkl data/HKO-7/samplers/hko7_cloudy_days_t20_test.txt.pkl -m SIMVP_HKO7_SIGMOID --loss facl-0.1
# Earthformer FACL
# CUDA_VISIBLE_DEVICES=0 python train_hko7.py data/HKO-7/samplers/hko7_cloudy_days_t20_train.txt.pkl data/HKO-7/samplers/hko7_cloudy_days_t20_test.txt.pkl -m EARTHFORMER_HKO7_SIGMOID --loss facl-0.1 --scheduler cosine
