#ConvLSTM learned cloudy test on rainy
# CUDA_VISIBLE_DEVICES=0 python save_preds.py -d HKO7_rainy_5_20 -m CONVLSTM_HKO7 -f checkpoints/hko-7/convlstm/20241221044446/convlstm_last_activation-none_final.pt -o test_eval
# CUDA_VISIBLE_DEVICES=1 python save_preds.py -d HKO7_rainy_5_20 -m CONVLSTM_HKO7_SIGMOID -f checkpoints/hko-7/convlstm/20250112162042/convlstm_last_activation-sigmoid_final.pt -o test_eval

#SimVP learned cloudy test on rainy
# CUDA_VISIBLE_DEVICES=0 python save_preds.py -d HKO7_rainy_5_20 -m SIMVP_HKO7 -f "checkpoints/hko-7/simvp/20250103010436/simvp_shape_in-5-1-480-480_shape_out-20-1-480-480_hid_S-16_hid_T-256_N_S-4_N_T-8_groups-8_last_activation-none_final.pt" -o test_eval
CUDA_VISIBLE_DEVICES=1 python save_preds.py -d HKO7_rainy_5_20 -m SIMVP_HKO7_SIGMOID -f "checkpoints/hko-7/simvp/20250113041000/simvp_shape_in-5-1-480-480_shape_out-20-1-480-480_hid_S-16_hid_T-256_N_S-4_N_T-8_groups-8_last_activation-sigmoid_final.pt" -o test_eval

#PredRNN learned cloudy test on rainy
# CUDA_VISIBLE_DEVICES=0 python save_preds.py -d HKO7_rainy_5_20 -m PREDRNN_HKO7 -f checkpoints/hko-7/predrnn/20241221044423/predrnn_num_layers-4_num_hidden-64-64-64-64_configs_step-33000.pt -o test_eval
CUDA_VISIBLE_DEVICES=1 python save_preds.py -d HKO7_rainy_5_20 -m PREDRNN_HKO7_SIGMOID -f checkpoints/hko-7/predrnn/20250113095721/predrnn_num_layers-4_num_hidden-64-64-64-64_configs-sigmoid_final.pt -o test_eval

# #Earthformer learned cloudy test on rainy
# CUDA_VISIBLE_DEVICES=0 python benchmark.py -d HKO7_rainy_5_20 -m CONVLSTM_HKO7 -f checkpoints/hko-7/convlstm/20241221044446/convlstm_last_activation-none_final.pt -o test_eval_mse
# CUDA_VISIBLE_DEVICES=0 python benchmark.py -d HKO7_rainy_5_20 -m CONVLSTM_HKO7 -f checkpoints/hko-7/convlstm/20250112162042/convlstm_last_activation-sigmoid_final.pt -o test_eval_mse

#ConvLSTM learned rainy test on rainy
# CUDA_VISIBLE_DEVICES=0 python benchmark.py -d HKO7_rainy_5_20 -m CONVLSTM_HKO7 -f checkpoints/hko-7/rainy/hko-7/convlstm/20250103175715/convlstm_last_activation-none_final.pt -o test_eval_rainy
# #PredRNN learned rainy test on rainy
# CUDA_VISIBLE_DEVICES=0 python benchmark.py -d HKO7_rainy_5_20 -m PREDRNN_HKO7 -f "checkpoints/hko-7/rainy/hko-7/predrnn/20250104195028/predrnn_num_layers-4_num_hidden-64-64-64-64_configs-{'input_length': 5, 'total_length': 25, 'img_channel': 1, 'patch_size': 4, 'img_width': 480, 'img_height': 480, 'filter_size': 5, 'stride': 1, 'layer_norm': 1}_final.pt" -o test_eval_rainy
# #SimVP learned rainy test on rainy
# CUDA_VISIBLE_DEVICES=0 python benchmark.py -d HKO7_rainy_5_20 -m SIMVP_HKO7 -f checkpoints/hko-7/rainy/hko-7/simvp/20250105190614/simvp_shape_in-5-1-480-480_shape_out-20-1-480-480_hid_S-16_hid_T-256_N_S-4_N_T-8_groups-8_last_activation-none_final.pt -o test_eval_rainy

# python eval_saved_preds.py /workspace_bk/piyush/results/facl/CONVLSTM_HKO7
# python eval_saved_preds.py /workspace_bk/piyush/results/facl/CONVLSTM_HKO7_SIGMOID
