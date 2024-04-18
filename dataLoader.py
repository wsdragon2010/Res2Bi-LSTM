'''
DataLoader for training
'''

import glob, numpy, os, random, soundfile, torch
from scipy import signal
import numpy as np
import librosa

class train_loader(object):
	def __init__(self, train_list, train_path, musan_path, rir_path, num_frames, min_speed_rate, max_speed_rate, **kwargs):
		self.train_path = train_path
		self.num_frames = num_frames
		self.min_speed_rate = min_speed_rate
		self.max_speed_rate = max_speed_rate
		# Load and configure augmentation files
		self.noisetypes = ['noise','speech','music']
		self.noisesnr = {'noise':[0,15],'speech':[13,20],'music':[5,15]}
		self.numnoise = {'noise':[1,1], 'speech':[3,8], 'music':[1,1]}
		self.noiselist = {}
		augment_files   = glob.glob(os.path.join(musan_path,'*/*/*/*.wav'))
		for file in augment_files:
			if file.split('/')[-4] not in self.noiselist:
				self.noiselist[file.split('/')[-4]] = []
			self.noiselist[file.split('/')[-4]].append(file)
		self.rir_files  = glob.glob(os.path.join(rir_path,'*/*/*.wav'))
		# Load data & labels
		self.data_list  = []
		self.data_label = []
		lines = open(train_list).read().splitlines()
		dictkeys = list(set([x.split()[0] for x in lines]))
		dictkeys.sort()
		dictkeys = { key : ii for ii, key in enumerate(dictkeys) }
		for index, line in enumerate(lines):
			speaker_label = dictkeys[line.split()[0]]
			file_name     = os.path.join(train_path, line.split()[1])
			self.data_label.append(speaker_label)
			self.data_list.append(file_name)

	def __getitem__(self, index):
		# Read the utterance and randomly select the segment
		audio, sr = soundfile.read(self.data_list[index])		
		length = self.num_frames * 160 + 240
		if audio.shape[0] <= length:
			shortage = length - audio.shape[0]
			audio = numpy.pad(audio, (0, shortage), 'wrap')
		start_frame = numpy.int64(random.random()*(audio.shape[0]-length))
		audio = audio[start_frame:start_frame + length]
		audio = numpy.stack([audio],axis=0)
		# Data Augmentation
		augtype = random.randint(0,5)
		if augtype == 0:   # Original
			audio = audio
		elif augtype == 1: # Reverberation
			audio = self.add_rev(audio)
		elif augtype == 2: # Babble
			audio = self.add_noise(audio, 'speech')
		elif augtype == 3: # Music
			audio = self.add_noise(audio, 'music')
		elif augtype == 4: # Noise
			audio = self.add_noise(audio, 'noise')
		elif augtype == 5: # Television noise
			audio = self.add_noise(audio, 'speech')
			audio = self.add_noise(audio, 'music')
		return torch.FloatTensor(audio[0]), self.data_label[index]
		# augtype = random.randint(0,6)
		# if augtype == 6: # Speed perturb
		# 	audio = self.time_stretch(self.data_list[index])
		# else:
		# 	audio, sr = soundfile.read(self.data_list[index])		
		# 	length = self.num_frames * 160 + 240
		# 	speed_length = int((self.num_frames * 160 + 240)/0.9)
			
		# 	if audio.shape[0] <= speed_length:
		# 		speed_shortage = speed_length - audio.shape[0]
		# 		speed_audio = numpy.pad(audio, (0, speed_shortage), 'wrap')
		# 	speed_start_frame = numpy.int64(random.random()*(audio.shape[0]-speed_length))
		# 	speed_audio = audio[speed_start_frame:speed_start_frame + speed_length]
		# 	speed_audio = numpy.stack([speed_audio],axis=0)

		# 	if audio.shape[0] <= length:
		# 		shortage = length - audio.shape[0]
		# 		audio = numpy.pad(audio, (0, shortage), 'wrap')
		# 	start_frame = numpy.int64(random.random()*(audio.shape[0]-length))
		# 	audio = audio[start_frame:start_frame + length]
		# 	audio = numpy.stack([audio],axis=0)

			
		# 	# Data Augmentation
		# 	# augtype = random.randint(0,6)
		# 	if augtype == 0:   # Original
		# 		audio = audio
		# 	elif augtype == 1: # Reverberation
		# 		audio = self.add_rev(audio)
		# 	elif augtype == 2: # Babble
		# 		audio = self.add_noise(audio, 'speech')
		# 	elif augtype == 3: # Music
		# 		audio = self.add_noise(audio, 'music')
		# 	elif augtype == 4: # Noise
		# 		audio = self.add_noise(audio, 'noise')
		# 	elif augtype == 5: # Television noise
		# 		audio = self.add_noise(audio, 'speech')
		# 		audio = self.add_noise(audio, 'music')
		# 	# elif augtype == 6: # Speed perturb
		# 	# 	audio = self.speed_perturb(speed_audio)
		# return torch.FloatTensor(audio[0]), self.data_label[index]


	def __len__(self):
		return len(self.data_list)

	def add_rev(self, audio):
		rir_file    = random.choice(self.rir_files)
		rir, sr     = soundfile.read(rir_file)
		rir         = numpy.expand_dims(rir.astype(numpy.float),0)
		rir         = rir / numpy.sqrt(numpy.sum(rir**2))
		return signal.convolve(audio, rir, mode='full')[:,:self.num_frames * 160 + 240]

	def add_noise(self, audio, noisecat):
		clean_db    = 10 * numpy.log10(numpy.mean(audio ** 2)+1e-4) 
		numnoise    = self.numnoise[noisecat]
		noiselist   = random.sample(self.noiselist[noisecat], random.randint(numnoise[0],numnoise[1]))
		noises = []
		for noise in noiselist:
			noiseaudio, sr = soundfile.read(noise)
			length = self.num_frames * 160 + 240
			if noiseaudio.shape[0] <= length:
				shortage = length - noiseaudio.shape[0]
				noiseaudio = numpy.pad(noiseaudio, (0, shortage), 'wrap')
			start_frame = numpy.int64(random.random()*(noiseaudio.shape[0]-length))
			noiseaudio = noiseaudio[start_frame:start_frame + length]
			noiseaudio = numpy.stack([noiseaudio],axis=0)
			noise_db = 10 * numpy.log10(numpy.mean(noiseaudio ** 2)+1e-4) 
			noisesnr   = random.uniform(self.noisesnr[noisecat][0],self.noisesnr[noisecat][1])
			noises.append(numpy.sqrt(10 ** ((clean_db - noise_db - noisesnr) / 10)) * noiseaudio)
		noise = numpy.sum(numpy.concatenate(noises,axis=0),axis=0,keepdims=True)
		return noise + audio
	
	def time_stretch(self, audio):

		speed_rate = random.uniform(self.min_speed_rate, self.max_speed_rate)

		y, sr = librosa.load(audio)
		# print(f'in:{y}')
		if speed_rate == 1.0:
			return y
		y_time_stretch = librosa.effects.time_stretch(y, rate=speed_rate)

		y_time_stretch = y_time_stretch[:self.num_frames * 160 + 240]
		y_time_stretch = numpy.stack([y_time_stretch],axis=0)
		# print(f'out:{y_time_stretch}')
		return y_time_stretch
	
	# def speed_perturb(self, audio):

	# 	speed_rate = random.uniform(self.min_speed_rate, self.max_speed_rate)
	# 	# print(f'in:{audio}')
	# 	if speed_rate == 1.0:
	# 		return audio
		
	# 	old_length = audio.shape[1]
	# 	# print(old_length)
	# 	new_length = int(old_length / speed_rate)
	# 	old_indices = np.arange(old_length)
	# 	new_indices = np.linspace(start=0, stop=old_length, num=new_length)
	# 	# print(f'new_indices:{new_indices.shape}')
	# 	# print(f'old_indices:{old_indices.shape}')
	# 	# print(f'audio:{audio.squeeze().shape}')
	# 	audio = np.interp(new_indices, old_indices, audio.squeeze())
	# 	audio = audio[:self.num_frames * 160 + 240]
	# 	audio = numpy.stack([audio],axis=0)
	# 	# print(f'out:{audio}')
	# 	return audio
	
	# def change_audio_speed(self, input_audio):
	# 	speed_rate = random.uniform(self.min_speed_rate, self.max_speed_rate)
	# 	try:
	# 		# output_temp = "./temp_output.wav"
	# 		# # 使用 SoX 完成变速操作
	# 		# subprocess.run(['sox', input_audio, output_temp, 'tempo', str(speed_rate)])
	# 		print(f'input_audio:{input_audio}')
	# 		audio_data, sr = soundfile.read(input_audio)
	# 		print(audio_data)
	# 		input_audio_bytes = audio_data.tobytes()
	# 		ffmpeg_command = [
	# 			'ffmpeg',
	# 			'-i', 'pipe:0',             # Input from pipe
	# 			'-filter:a', f'atempo={speed_rate}',  # Apply tempo filter
	# 			'-f', 'wav', 'pipe:1'       # Output to pipe
	# 		]

	# 		output_audio_bytes = subprocess.check_output(ffmpeg_command, input=input_audio_bytes, stderr=subprocess.PIPE)

	# 		# output_audio = np.frombuffer(output_audio_bytes, dtype=np.int16)  # Adjust dtype according to audio format
			
	# 		# 将处理后的音频数据从字节转换为数组
	# 		output_audio = np.frombuffer(output_audio_bytes, dtype=np.float32)
	# 		print(f'out_audio:{output_audio}')

	# 		# 读取处理后的音频数据
	# 		# output_audio, _ = soundfile.read(output_temp)

	# 		length = self.num_frames * 160 + 240
	# 		if output_audio.shape[0] <= length:
	# 			shortage = length - output_audio.shape[0]
	# 			output_audio = numpy.pad(output_audio, (0, shortage), 'wrap')
	# 		start_frame = numpy.int64(random.random()*(output_audio.shape[0]-length))
	# 		output_audio = output_audio[start_frame:start_frame + length]
	# 		output_audio = numpy.stack([output_audio],axis=0)
			
	# 		# 删除临时文件
	# 		# subprocess.run(['rm', output_temp])
			
	# 		return output_audio
	# 	except Exception as e:
	# 		print("Error:", e)
	# 		return None

	# def change_audio_speed(self,input_audio):
	# 	speed_rate = random.uniform(self.min_speed_rate, self.max_speed_rate)
	# 	input_audio = input_audio.squeeze()
	# 	input_audio_bytes = input_audio.tobytes()
	# 	audio = AudioSegment.from_buffer(input_audio_bytes)
	# 	output_audio = audio.speedup(playback_speed=speed_rate)
	# 	output_audio_array = np.array(output_audio.get_array_of_samples())
	# 	output_audio_array = np.stack([output_audio_array],axis=0)
	# 	return output_audio_array
