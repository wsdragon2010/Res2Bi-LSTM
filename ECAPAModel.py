'''
This part is used to train the speaker model and evaluate the performances
'''

import random
from matplotlib import pyplot as plt
import torch, sys, os, tqdm, numpy, soundfile, time, pickle
import torch.nn as nn
from tools import *
from loss import AAMsoftmax
from model import ECAPA_TDNN, Bi_ECAPA_TDNN_Add, ECAPA_TDNN_lstm, Bi_ECAPA_TDNN_lstm_Add, ECAPA_TDNN_BiBottle, \
	ECAPA_TDNN_CNN_LSTM, ECAPA_TDNN_LSTM_CNN, Res2Net, Res2Net_1d, Res2Net_1d_LSTM, ERes2Net
import numpy as np

class ECAPAModel(nn.Module):
	def __init__(self, lr, lr_decay, C , n_class, m, s, test_step, **kwargs):
		super(ECAPAModel, self).__init__()
		## ECAPA-TDNN
		# self.speaker_encoder = ECAPA_TDNN(C = C).cuda()
		self.speaker_encoder = Bi_ECAPA_TDNN_Add(C = C).cuda()
		# self.speaker_encoder = Bi_ECAPA_TDNN_lstm_Add(C = C).cuda()
		# self.speaker_encoder = ECAPA_TDNN_lstm(C = C).cuda()
		# self.speaker_encoder = ECAPA_TDNN_BiBottle(C = C).cuda()
		# self.speaker_encoder = ECAPA_TDNN_CNN_LSTM(C = C).cuda()
		# self.speaker_encoder = ECAPA_TDNN_LSTM_CNN(C = C).cuda()
		# self.speaker_encoder = ERes2Net().cuda() 
		## Classifier
		self.speaker_loss    = AAMsoftmax(n_class = n_class, m = m, s = s).cuda()

		self.optim           = torch.optim.Adam(self.parameters(), lr = lr, weight_decay = 2e-5)
		# self.scheduler       = torch.optim.lr_scheduler.StepLR(self.optim, step_size = test_step, gamma=lr_decay)
		self.scheduler = torch.optim.lr_scheduler.CyclicLR(self.optim, base_lr=1e-8, max_lr=1e-3, mode='triangular2', step_size_up=65000, cycle_momentum=False)
		print(time.strftime("%m-%d %H:%M:%S") + " Model para number = %.2f"%(sum(param.numel() for param in self.speaker_encoder.parameters()) / 1024 / 1024))

	def train_network(self, epoch, loader):
		self.train()
		## Update the learning rate based on the current epcoh
		# self.scheduler.step(epoch - 1)
		index, top1, loss = 0, 0, 0
		# lr = self.optim.param_groups[0]['lr']
		iters = []
		lrs = []
		for num, (data, labels) in enumerate(loader, start = 1):
			# print(f'loader:{len(loader)}-----')
			iter = len(loader)*(epoch-1) + num - 1
			self.scheduler.step()
			lr = self.optim.param_groups[0]['lr']
			# iters.append(iter)
			# lrs.append(lr)
			# # 画出lr的变化  
			# plt.figure()  
			# plt.plot(iters, lrs)
			# plt.xlabel("iter")
			# plt.ylabel("lr")
			# plt.title("CyclicLR")
			# plt.show()
			# plt.savefig("/root/autodl-tmp/exp/ECAPA_TDNN_fuxian_CyclicLR/lr.png")

			self.zero_grad()
			labels            = torch.LongTensor(labels).cuda()
			speaker_embedding = self.speaker_encoder.forward(data.cuda(), aug = True)
			nloss, prec       = self.speaker_loss.forward(speaker_embedding, labels)			
			nloss.backward()
			self.optim.step()
			index += len(labels)
			top1 += prec
			loss += nloss.detach().cpu().numpy()
			sys.stderr.write(time.strftime("%m-%d %H:%M:%S") + \
			" [%2d] Lr: %5f, Training: %.2f%%, "    %(epoch, lr, 100 * (num / loader.__len__())) + \
			" Loss: %.5f, ACC: %2.2f%% \r"        %(loss/(num), top1/index*len(labels)))
			sys.stderr.flush()
		sys.stdout.write("\n")
		return loss/num, lr, top1/index*len(labels)

	def eval_network(self, eval_list, eval_path):
		self.eval()
		files = []
		embeddings = {}
		lines = open(eval_list).read().splitlines()
		for line in lines:
			files.append(line.split()[1])
			files.append(line.split()[2])
		setfiles = list(set(files))
		setfiles.sort()

		for idx, file in tqdm.tqdm(enumerate(setfiles), total = len(setfiles)):
			audio, _  = soundfile.read(os.path.join(eval_path, file))
			# Full utterance
			data_1 = torch.FloatTensor(numpy.stack([audio],axis=0)).cuda()

			# Spliited utterance matrix
			max_audio = 300 * 160 + 240
			if audio.shape[0] <= max_audio:
				shortage = max_audio - audio.shape[0]
				audio = numpy.pad(audio, (0, shortage), 'wrap')
			feats = []
			startframe = numpy.linspace(0, audio.shape[0]-max_audio, num=5)
			for asf in startframe:
				feats.append(audio[int(asf):int(asf)+max_audio])
			feats = numpy.stack(feats, axis = 0).astype(numpy.float)
			data_2 = torch.FloatTensor(feats).cuda()
			# Speaker embeddings
			with torch.no_grad():
				embedding_1 = self.speaker_encoder.forward(data_1, aug = False)
				embedding_1 = F.normalize(embedding_1, p=2, dim=1)
				embedding_2 = self.speaker_encoder.forward(data_2, aug = False)
				embedding_2 = F.normalize(embedding_2, p=2, dim=1)
			embeddings[file] = [embedding_1, embedding_2]
		scores, labels  = [], []

		for line in lines:			
			embedding_11, embedding_12 = embeddings[line.split()[1]]
			embedding_21, embedding_22 = embeddings[line.split()[2]]
			# Compute the scores
			score_1 = torch.mean(torch.matmul(embedding_11, embedding_21.T)) # higher is positive
			score_2 = torch.mean(torch.matmul(embedding_12, embedding_22.T))
			score = (score_1 + score_2) / 2
			score = score.detach().cpu().numpy()
			scores.append(score)
			labels.append(int(line.split()[0]))
			
		# Coumpute EER and minDCF
		EER = tuneThresholdfromScore(scores, labels, [1, 0.1])[1]
		fnrs, fprs, thresholds = ComputeErrorRates(scores, labels)
		minDCF, _ = ComputeMinDcf(fnrs, fprs, thresholds, 0.01, 1, 1)

		return EER, minDCF

	def get_score_file(self, eval_list, eval_path, diff_id_path):
		self.eval()
		files = []
		embeddings = {}
		lines = open(eval_list).read().splitlines()
		for line in lines:
			# files.append(line)
			files.append(line.split()[1])
			files.append(line.split()[2])
		setfiles = list(set(files))
		setfiles.sort()

		for idx, file in tqdm.tqdm(enumerate(setfiles), total = len(setfiles)):
			audio, _  = soundfile.read(os.path.join(eval_path, file))
			# Full utterance
			data_1 = torch.FloatTensor(numpy.stack([audio],axis=0)).cuda()

			# Spliited utterance matrix
			max_audio = 300 * 160 + 240
			if audio.shape[0] <= max_audio:
				shortage = max_audio - audio.shape[0]
				audio = numpy.pad(audio, (0, shortage), 'wrap')
			feats = []
			startframe = numpy.linspace(0, audio.shape[0]-max_audio, num=5)
			for asf in startframe:
				feats.append(audio[int(asf):int(asf)+max_audio])
			feats = numpy.stack(feats, axis = 0).astype(numpy.float)
			data_2 = torch.FloatTensor(feats).cuda()
			# Speaker embeddings
			with torch.no_grad():
				embedding_1 = self.speaker_encoder.forward(data_1, aug = False)
				embedding_1 = F.normalize(embedding_1, p=2, dim=1)
				embedding_2 = self.speaker_encoder.forward(data_2, aug = False)
				embedding_2 = F.normalize(embedding_2, p=2, dim=1)
			embeddings[file] = [embedding_1, embedding_2]

		# scores, labels  = [], []

		# for line in lines:			
		# 	embedding_11, embedding_12 = embeddings[line.split()[1]]
		# 	embedding_21, embedding_22 = embeddings[line.split()[2]]
		# 	# Compute the scores
		# 	score_1 = torch.mean(torch.matmul(embedding_11, embedding_21.T)) # higher is positive
		# 	score_2 = torch.mean(torch.matmul(embedding_12, embedding_22.T))
		# 	score = (score_1 + score_2) / 2
		# 	score = score.detach().cpu().numpy()
		# 	scores.append(score)
		# 	labels.append(int(line.split()[0]))
			
		# # Coumpute EER and minDCF
		# EER = tuneThresholdfromScore(scores, labels, [1, 0.1])[1]
		# fnrs, fprs, thresholds = ComputeErrorRates(scores, labels)
		# minDCF, _ = ComputeMinDcf(fnrs, fprs, thresholds, 0.01, 1, 1)
		# print(f'EER:{EER},minDCF:{minDCF}')
		
		diff = []
		diff_scores = {}
		for enroll in setfiles:	
			diff_one = []	
			enroll_id = enroll.split('/')[1].split('-')[0]
			embedding_11, embedding_12 = embeddings[enroll]
			cnt = 0 #计数器，控制计算多少条冒认者语音
			setfiles_copy = setfiles.copy()
			random.shuffle(setfiles_copy)
			test_score = {}
			for test in setfiles_copy:
				test_id = test.split('/')[1].split('-')[0]
				if enroll_id != test_id: #只计算冒认者
					cnt += 1
					if cnt > 10000:
						break
					if test in diff_scores:
						if enroll in diff_scores[test]:
							score = diff_scores[test][enroll]
							# print(f'score:{score}')
						else:
							embedding_21, embedding_22 = embeddings[test]
							score_1 = torch.mean(torch.matmul(embedding_11, embedding_21.T)) # higher is positive
							score_2 = torch.mean(torch.matmul(embedding_12, embedding_22.T))
							score = (score_1 + score_2) / 2
							score = score.detach().cpu().item()
							
							test_score[test] = score
							diff_scores[enroll] = test_score
					else:
						embedding_21, embedding_22 = embeddings[test]
						score_1 = torch.mean(torch.matmul(embedding_11, embedding_21.T)) # higher is positive
						score_2 = torch.mean(torch.matmul(embedding_12, embedding_22.T))
						score = (score_1 + score_2) / 2
						score = score.detach().cpu().item()
						
						test_score[test] = score
						diff_scores[enroll] = test_score
					diff_one.append(score)

			diff_one = sorted(diff_one,reverse=True)
			diff_one = list(map(str,diff_one))
			with open(diff_id_path,'a') as f:
				line = enroll + ' ' + ' '.join(diff_one)
				f.write(line + '\n')
		return 

		# # 旧方案，慢，vox-E-4k-BILSTM 5.5天
		# for enroll in setfiles:	
		# 	# same_one = []
		# 	diff_one = []	
		# 	# print(enroll)
		# 	# same_one.append(enroll)
		# 	# diff_one.append(enroll)
		# 	enroll_id = enroll.split('/')[1].split('-')[0]
		# 	embedding_11, embedding_12 = embeddings[enroll]
		# 	# cnt = 0
		# 	setfiles_copy = setfiles.copy()
		# 	random.shuffle(setfiles_copy)
		# 	setfiles_copy = setfiles_copy[:4000]
		# 	for test in setfiles_copy:
		# 		if enroll == test:
		# 			continue
		# 		# cnt += 1
		# 		# if cnt > 2000:
		# 		# 	break
		# 		test_id = test.split('/')[1].split('-')[0]
		# 		embedding_21, embedding_22 = embeddings[test]
		# 		score_1 = torch.mean(torch.matmul(embedding_11, embedding_21.T)) # higher is positive
		# 		score_2 = torch.mean(torch.matmul(embedding_12, embedding_22.T))
		# 		score = (score_1 + score_2) / 2
		# 		score = score.detach().cpu().item()
		# 		# print(score)
		# 		# if enroll_id == test_id:
		# 		# 	same_one.append(str(score))
		# 		# else:
		# 		if enroll_id != test_id:
		# 			diff_one.append(score)
		# 	# same.append(same_one)
		# 	# diff.append(diff_one)
		# 	diff_one = sorted(diff_one,reverse=True)
		# 	# diff_one = diff_one[:1000] # 只取分数最高的1000个，因为K值最大只取1000
		# 	diff_one = list(map(str,diff_one))
		# 	with open(diff_id_path,'a') as f:
		# 		# for line in diff:
		# 		# 	# print(line)
		# 		# 	line = ' '.join(line)
		# 		# 	f.write(line + '\n')
		# 		line = enroll + ' ' + ' '.join(diff_one)
		# 		f.write(line + '\n')
		# return 
	
	def eval_as_norm(self, eval_list, eval_path, score_list, K):
		self.eval()
		files = []
		embeddings = {}
		lines = open(eval_list).read().splitlines()
		for line in lines:
			files.append(line.split()[1])
			files.append(line.split()[2])
		setfiles = list(set(files))
		setfiles.sort()

		for idx, file in tqdm.tqdm(enumerate(setfiles), total = len(setfiles)):
			audio, _  = soundfile.read(os.path.join(eval_path, file))
			# Full utterance
			data_1 = torch.FloatTensor(numpy.stack([audio],axis=0)).cuda()

			# Spliited utterance matrix
			max_audio = 300 * 160 + 240
			if audio.shape[0] <= max_audio:
				shortage = max_audio - audio.shape[0]
				audio = numpy.pad(audio, (0, shortage), 'wrap')
			feats = []
			startframe = numpy.linspace(0, audio.shape[0]-max_audio, num=5)
			for asf in startframe:
				feats.append(audio[int(asf):int(asf)+max_audio])
			feats = numpy.stack(feats, axis = 0).astype(numpy.float)
			data_2 = torch.FloatTensor(feats).cuda()
			# Speaker embeddings
			with torch.no_grad():
				embedding_1 = self.speaker_encoder.forward(data_1, aug = False)
				embedding_1 = F.normalize(embedding_1, p=2, dim=1)
				embedding_2 = self.speaker_encoder.forward(data_2, aug = False)
				embedding_2 = F.normalize(embedding_2, p=2, dim=1)
			embeddings[file] = [embedding_1, embedding_2]
		scores, labels  = [], []

		diff_scores = {} # 定义一个字典，用于存放各个语音和冒认者语音之间的分数
		with open(score_list,'r') as f:
			score_lines = f.readlines()
			for score_line in score_lines:
				file_path = score_line.split(' ')[0]
				diff_score = list(map(float, score_line.split(' ')[1:]))
				diff_score = sorted(diff_score,reverse=True) # 将读取出的score进行排序，降序排列，取前topK个元素
				diff_scores[file_path] = diff_score[:K] # 取topK个分数存入字典
		# print(diff_scores)
		for line in lines:	
			enroll_file = line.split()[1]	
			test_file = line.split()[2]	
			embedding_11, embedding_12 = embeddings[enroll_file]
			embedding_21, embedding_22 = embeddings[test_file]
			# Compute the scores
			score_1 = torch.mean(torch.matmul(embedding_11, embedding_21.T)) # higher is positive
			score_2 = torch.mean(torch.matmul(embedding_12, embedding_22.T))
			score = (score_1 + score_2) / 2
			score = score.detach().cpu().numpy()

			# 从scores_list中提取注册语音、测试语音分别和冒认语音的topK得分
			enroll_mean = np.array(diff_scores[enroll_file]).mean()
			enroll_std = np.array(diff_scores[enroll_file]).std()
			test_mean = np.array(diff_scores[test_file]).mean()
			test_std = np.array(diff_scores[test_file]).std()
			# print(f'enroll_mean:{enroll_mean}')

			as_score = 0.5 * ((score - enroll_mean)/enroll_std + (score - test_mean)/test_std)
			scores.append(as_score)
			labels.append(int(line.split()[0]))
			
		# Coumpute EER and minDCF
		EER = tuneThresholdfromScore(scores, labels, [1, 0.1])[1]
		fnrs, fprs, thresholds = ComputeErrorRates(scores, labels)
		minDCF, _ = ComputeMinDcf(fnrs, fprs, thresholds, 0.01, 1, 1)

		return EER, minDCF
	
	def save_parameters(self, path):
		torch.save(self.state_dict(), path)

	def load_parameters(self, path):
		self_state = self.state_dict()
		loaded_state = torch.load(path)
		for name, param in loaded_state.items():
			origname = name
			if name not in self_state:
				name = name.replace("module.", "")
				if name not in self_state:
					print("%s is not in the model."%origname)
					continue
			if self_state[name].size() != loaded_state[origname].size():
				print("Wrong parameter length: %s, model: %s, loaded: %s"%(origname, self_state[name].size(), loaded_state[origname].size()))
				continue
			self_state[name].copy_(param)