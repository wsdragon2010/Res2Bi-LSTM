echo "python eval_as-norm_by_score.py --eval --K 200"
python eval_as-norm_by_score.py --eval --K 200
echo python eval_as-norm_by_score.py --eval --K 500
python eval_as-norm_by_score.py --eval --K 500
echo "python eval_as-norm_by_score.py --eval --K 800"
python eval_as-norm_by_score.py --eval --K 800
echo "python eval_as-norm_by_score.py --eval --K 1000"
python eval_as-norm_by_score.py --eval --K 1000
echo "python eval_as-norm_by_score.py --eval --K 1500"
python eval_as-norm_by_score.py --eval --K 1500
echo "python eval_as-norm_by_score.py --eval --K 2000"
python eval_as-norm_by_score.py --eval --K 2000
echo "python eval_as-norm_by_score.py --eval --K 3000"
python eval_as-norm_by_score.py --eval --K 3000


echo "---------------------------------------------------"
python eval_get_score.py --eval
echo "python eval_as-norm_by_score.py --eval --K 200"
python eval_as-norm_by_score.py --eval --K 200 --n_class 5994 --score_list "/root/autodl-tmp/exp/Bi_ECAPA_TDNN_Add_CyclicLR_018/CNC_E_file_1w_Bi_ECAPA_TDNN_Add_CyclicLR_diff_score_96.txt" --model_path "/root/autodl-tmp/exp/Bi_ECAPA_TDNN_Add_CyclicLR_018/model/model_0096.model" 
echo python eval_as-norm_by_score.py --eval --K 500
python eval_as-norm_by_score.py --eval --K 500 --n_class 5994 --score_list "/root/autodl-tmp/exp/Bi_ECAPA_TDNN_Add_CyclicLR_018/CNC_E_file_1w_Bi_ECAPA_TDNN_Add_CyclicLR_diff_score_96.txt" --model_path "/root/autodl-tmp/exp/Bi_ECAPA_TDNN_Add_CyclicLR_018/model/model_0096.model" 
echo "python eval_as-norm_by_score.py --eval --K 800"
python eval_as-norm_by_score.py --eval --K 800 --n_class 5994 --score_list "/root/autodl-tmp/exp/Bi_ECAPA_TDNN_Add_CyclicLR_018/CNC_E_file_1w_Bi_ECAPA_TDNN_Add_CyclicLR_diff_score_96.txt" --model_path "/root/autodl-tmp/exp/Bi_ECAPA_TDNN_Add_CyclicLR_018/model/model_0096.model" 
echo "python eval_as-norm_by_score.py --eval --K 1000"
python eval_as-norm_by_score.py --eval --K 1000 --n_class 5994 --score_list "/root/autodl-tmp/exp/Bi_ECAPA_TDNN_Add_CyclicLR_018/CNC_E_file_1w_Bi_ECAPA_TDNN_Add_CyclicLR_diff_score_96.txt" --model_path "/root/autodl-tmp/exp/Bi_ECAPA_TDNN_Add_CyclicLR_018/model/model_0096.model" 
echo "python eval_as-norm_by_score.py --eval --K 1500"
python eval_as-norm_by_score.py --eval --K 1500 --n_class 5994 --score_list "/root/autodl-tmp/exp/Bi_ECAPA_TDNN_Add_CyclicLR_018/CNC_E_file_1w_Bi_ECAPA_TDNN_Add_CyclicLR_diff_score_96.txt" --model_path "/root/autodl-tmp/exp/Bi_ECAPA_TDNN_Add_CyclicLR_018/model/model_0096.model" 
echo "python eval_as-norm_by_score.py --eval --K 2000"
python eval_as-norm_by_score.py --eval --K 2000 --n_class 5994 --score_list "/root/autodl-tmp/exp/Bi_ECAPA_TDNN_Add_CyclicLR_018/CNC_E_file_1w_Bi_ECAPA_TDNN_Add_CyclicLR_diff_score_96.txt" --model_path "/root/autodl-tmp/exp/Bi_ECAPA_TDNN_Add_CyclicLR_018/model/model_0096.model" 
echo "python eval_as-norm_by_score.py --eval --K 3000"
python eval_as-norm_by_score.py --eval --K 3000 --n_class 5994 --score_list "/root/autodl-tmp/exp/Bi_ECAPA_TDNN_Add_CyclicLR_018/CNC_E_file_1w_Bi_ECAPA_TDNN_Add_CyclicLR_diff_score_96.txt" --model_path "/root/autodl-tmp/exp/Bi_ECAPA_TDNN_Add_CyclicLR_018/model/model_0096.model" 