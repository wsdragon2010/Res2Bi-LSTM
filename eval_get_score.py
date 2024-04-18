'''
This is the main code of the ECAPATDNN project, to define the parameters and build the construction
'''

import argparse, glob, os, torch, warnings, time
from tools import *
from dataLoader import train_loader
from ECAPAModel import ECAPAModel

parser = argparse.ArgumentParser(description = "ECAPA_eval")
## Training Settings
parser.add_argument('--num_frames', type=int,   default=200,     help='Duration of the input segments, eg: 200 for 2 second')
parser.add_argument('--max_epoch',  type=int,   default=120,      help='Maximum number of epochs')
parser.add_argument('--batch_size', type=int,   default=400,     help='Batch size')
parser.add_argument('--n_cpu',      type=int,   default=10,       help='Number of loader threads')
parser.add_argument('--test_step',  type=int,   default=1,       help='Test and save every [test_step] epochs')
parser.add_argument('--lr',         type=float, default=0.001,   help='Learning rate')
parser.add_argument("--lr_decay",   type=float, default=0.97,    help='Learning rate decay every [test_step] epochs')
parser.add_argument("--min_speed_rate",   type=float, default=0.9,    help='new sample rate (floor)')
parser.add_argument("--max_speed_rate",   type=float, default=1.1,    help='new sample rate (top)')

## Training and evaluation path/lists, save path
# parser.add_argument('--eval_list_no_dup',  type=str,   default="/root/autodl-tmp/data/Vox1_H_no_dup.txt",              help='The path of the evaluation list, veri_test2.txt comes from https://www.robots.ox.ac.uk/~vgg/data/voxceleb/meta/veri_test2.txt')
# parser.add_argument('--eval_list',  type=str,   default="/root/autodl-tmp/cnceleb/CN-Celeb_wav/eval/list/trials_test.txt",              help='The path of the evaluation list, veri_test2.txt comes from https://www.robots.ox.ac.uk/~vgg/data/voxceleb/meta/veri_test2.txt')
# parser.add_argument('--eval_list',  type=str,   default="/root/autodl-tmp/ECAPA-TDNN/veri_test2.txt",              help='The path of the evaluation list, veri_test2.txt comes from https://www.robots.ox.ac.uk/~vgg/data/voxceleb/meta/veri_test2.txt')
# parser.add_argument('--eval_list',  type=str,   default="/root/autodl-tmp/data/Vox1_no_clean.txt",              help='The path of the evaluation list, veri_test2.txt comes from https://www.robots.ox.ac.uk/~vgg/data/voxceleb/meta/veri_test2.txt')
# parser.add_argument('--eval_list',  type=str,   default="/root/autodl-tmp/data/Vox1_H_file_paths_split_1.txt",              help='The path of the evaluation list, veri_test2.txt comes from https://www.robots.ox.ac.uk/~vgg/data/voxceleb/meta/veri_test2.txt')
# parser.add_argument('--eval_list',  type=str,   default="/root/autodl-tmp/data/Vox1_E.txt",              help='The path of the evaluation list, veri_test2.txt comes from https://www.robots.ox.ac.uk/~vgg/data/voxceleb/meta/veri_test2.txt')
# parser.add_argument('--eval_path',  type=str,   default="/root/autodl-tmp/data/vox1_wav",                    help='The path of the evaluation data, eg:"/data08/VoxCeleb1/test/wav" in my case')
parser.add_argument('--eval_list',  type=str,   default="/root/autodl-tmp/cnceleb/CN-Celeb_wav/eval/lists/trials_test.txt",              help='The path of the evaluation list, veri_test2.txt comes from https://www.robots.ox.ac.uk/~vgg/data/voxceleb/meta/veri_test2.txt')
parser.add_argument('--eval_path',  type=str,   default="/root/autodl-tmp/cnceleb/CN-Celeb_wav/eval/",                    help='The path of the evaluation data, eg:"/data08/VoxCeleb1/test/wav" in my case')
# parser.add_argument('--eval_path',  type=str,   default="/root/autodl-tmp/data/vox1_test",                    help='The path of the evaluation data, eg:"/data08/VoxCeleb1/test/wav" in my case')
# parser.add_argument('--eval_path',  type=str,   default="/root/autodl-tmp/cnceleb/CN-Celeb_wav/eval/",                    help='The path of the evaluation data, eg:"/data08/VoxCeleb1/test/wav" in my case')
# parser.add_argument('--musan_path', type=str,   default="/data08/Others/musan_split",                    help='The path to the MUSAN set, eg:"/data08/Others/musan_split" in my case')
parser.add_argument('--diff_id_scores_path',  type=str,   default="/root/autodl-tmp/exp/Bi_ECAPA_TDNN_Add_CyclicLR_018/CNC_E_file_1w_Bi_ECAPA_TDNN_Add_CyclicLR_diff_score_96.txt")                  
parser.add_argument('--model_path',  type=str,   default="/root/autodl-tmp/exp/Bi_ECAPA_TDNN_Add_CyclicLR_018/model/model_0096.model")

## Model and Loss settings
parser.add_argument('--C',       type=int,   default=1024,   help='Channel size for the speaker encoder')
parser.add_argument('--m',       type=float, default=0.2,    help='Loss margin in AAM softmax')
parser.add_argument('--s',       type=float, default=30,     help='Loss scale in AAM softmax')
parser.add_argument('--n_class', type=int,   default=5994,   help='Number of speakers')
# parser.add_argument('--n_class', type=int,   default=8790,   help='Number of speakers')
# parser.add_argument('--n_class', type=int,   default=2796,   help='Number of speakers')
parser.add_argument('--save_path',  type=str,   default="/private/ecapa_exp_stu30_a30_tempo091011/ecapa-tdnn_fuxian_09_10_11",                                     help='Path to save the score.txt and models')

## Command
parser.add_argument('--eval',    dest='eval', action='store_true', help='Only do evaluation')

## Initialization
warnings.simplefilter("ignore")
torch.multiprocessing.set_sharing_strategy('file_system')
args = parser.parse_args()
args = init_args(args)


## Only do evaluation, the initial_model is necessary
if args.eval == True:
	s = ECAPAModel(**vars(args))
	print("Model %s loaded from previous state!"%args.model_path)
	s.load_parameters(args.model_path)
	# EER, minDCF = s.eval_network(eval_list = args.eval_list, eval_path = args.eval_path)
	# print("EER %2.2f%%, minDCF %.4f%%"%(EER, minDCF))
	s.get_score_file(eval_list = args.eval_list, eval_path = args.eval_path, diff_id_path = args.diff_id_scores_path)
	quit()



