import numpy as np 
import cv2 
import subprocess
import argparse 
from model import KeyGenerator, Wav2Lip
import audio
import torch  
import numpy as np 
import os 
import face_detection as fd
from train import Dataset, load_checkpoint, generate_target, getOriginalKeypoints
from torch import nn
import warnings
import torch.nn.functional as F
import librosa
import cv2
import seaborn as sns 
import subprocess 


warnings.filterwarnings(action='ignore') 
parser = argparse.ArgumentParser()  
parser.add_argument("--sample_rate", dest='sr', default=16000, help="audio sampling rate")
parser.add_argument("--path", default='./test', help="test mp4 path")
parser.add_argument("--result", default='./result', help="path of output images")
parser.add_argument("--input", default='test.mp4', help="test mp4 file name")
parser.add_argument("--output", default='out.mp4', help="write output avi file name or output willbe image frames", type=str)
parser.add_argument("--fps", default=25, help="frame per sec")
parser.add_argument("--mel_step_size", default=80, help="time step length of mel spectrogram")
parser.add_argument("--multigpu", default=True, help="use torch dataparallel")
parser.add_argument('--keypoint', help='keypoint model weight path', default='./checkpoints/keypoint_checkpoint.pth', type=str)
parser.add_argument('--model', help='wav2lip model weight path', default='./checkpoints/train_checkpoint_mouth_resize.pth', type=str)

args = parser.parse_args()

def inference(mel, img, data, model, keygenerator, device):  
	model.eval() 
	keygenerator.eval()  
	
	mel = mel.unsqueeze(0).to(device)
	img = img.unsqueeze(0).to(device) 

	# mouth and face line landmarks prediction
	keypoints = keygenerator(mel) 
	keypoints = keypoints.view(keypoints.size(-2),keypoints.size(-1)).cpu().detach()
	N, tilt, mean, y_keypoints = data   

	# restores tilt and scale of landmarks 
	try:
		kps = getOriginalKeypoints(keypoints, N, torch.tensor(tilt), mean) 
	except Exception as e:
		print(f'original keypoint restruction error : {e}')    

	# add the keypoints to masked images
	for j in range(len(kps)): 
		try:
			img[0, int(kps[j,1]), int(kps[j,0])] = 1 
		except Exception as e:
			print(f'{e} : at  {j}')


	# keypoint to heatmap transformation
	# key = y_keypoints.reshape(-1,68,2)  
	# target, target_weight = generate_target(key,np.ones_like(key[:,:,:1]))
	# k = k.reshape(-1,37,2) 
 

	img = img.permute(0,3,1,2) 
	pred = model(img)   
	return pred.permute(0,2,3,1).squeeze(0)    

def run(args, model, keygen, detector, orig_mel, dataset, device, cap):
	i = 0
	while(cap.isOpened()):
		ret, frame = cap.read()  
		try:
			img, box = detector.run(frame, box=True) #box l = rect.left() r = rect.right() t = rect.top() b = rect.bottom() 
			cv2.rectangle(frame, tuple([box[2][0],box[3][0]]), tuple([box[2][1],box[3][1]]),  (0, 0, 255), 1)
			preprocessed_img, img = dataset.load_image(img, return_img=True, mask=True) 
			landmark = detector.landmark_detection(img)     

			# data [N, theta, mouthMean, keypoints]
			landmark, data = dataset.preprocess(landmark)
			'''
				N, tilt, mean, keypoints = data 
				try:
					kps = getOriginalKeypoints(pred[i], N, tilt, mean)
				except:
					break
				keypoints[:17] = kps[:17]
				keypoints[48:68] = kps[17:] 
				
				keypoints[:, 0] = ((box[2][1] - box[2][0]) / img_size) * keypoints[:, 0] + box[2][0]
				keypoints[:, 1] = ((box[3][1] - box[3][0]) / img_size) * keypoints[:, 1] + box[3][0] 

				new_img = drawLips(keypoints.astype(int) , frame, c = (0, 0, 255), show = False)'''
			mel, start, end = dataset.crop_audio_window(orig_mel.copy(), i, True) 
			mel = torch.FloatTensor(mel.T)  


			if mel.size(-1) != args.mel_step_size : 
				mel = F.pad(mel, (0, args.mel_step_size - mel.size(-1)), "constant", 0)  

			pred_img = inference(mel, preprocessed_img, data, model, keygen, device=device) 
			
			# rescale to original bounding box size
			pred_img = np.array(pred_img.cpu().detach() * 255.).astype('uint8')  
			pred_img = cv2.resize(pred_img, (int(box[3][1]-box[3][0]+1), int(box[2][1]-box[2][0]+1)))  
			
			frame[box[3][0]:box[3][1]+1, box[2][0]:box[2][1]+1] = pred_img 
 
			if args.output:
				args.writer.write(frame)
			else:
				cv2.imwrite(os.path.join(args.result, f'{i}_img.png'), frame)   
			i += 1 
		except Exception as e:
			print(e)
			break	
		
	
		
	print(f'Done {i} frames')
	cap.release()
	args.writer.release()
	cv2.destroyAllWindows()
 
	return

if __name__=="__main__":
	
	if not os.path.isdir(args.path) or not os.path.isdir(args.result):
		os.makedirs(args.path, exist_ok=True)
		os.makedirs(args.result, exist_ok=True)
 
	detector = fd.face_detection('whole_face')

	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")   

	dataset = Dataset() 
	wavpath = os.path.join(args.path, "audio.wav")

	if not os.path.isfile(wavpath):
		print(f'audio not exists {wavpath}') 
		subprocess.call(
			f"ffmpeg -i {os.path.join(args.path, args.input)} -ab 160k -ac 1 -ar 16000 -vn {wavpath}"\
					,shell=True)
	
	wav = librosa.load(wavpath, args.sr)[0]
	orig_mel = audio.melspectrogram(wav).T
  
	# Model
	keygen = KeyGenerator().to(device) 
	model = Wav2Lip().to(device)

	if args.multigpu:
		model = nn.DataParallel(model)
		keygen = nn.DataParallel(keygen)
		print('use dataparallel') 

	if os.path.isfile(args.model) :
		try:
			model = load_checkpoint(args.model, model, reset_optimizer=True)
			print('wav2lip model loaded') 
		except:
			pass
	if args.keypoint is not None:
		try:
			keygen = load_checkpoint(args.keypoint, keygen, reset_optimizer=True)
			print('keypoint model loaded') 
		except:
			pass


	print('open video capture')  
	cap = cv2.VideoCapture(os.path.join(args.path, args.input))
	cap.set(cv2.CAP_PROP_FPS, args.fps)

	if args.output:
		# fourcc = cv2.VideoWriter_fourcc(*'XVID')
		fourcc = cv2.VideoWriter_fourcc(*'MP4V')
		w = round(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
		h = round(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
		args.writer = cv2.VideoWriter(os.path.join(args.path, args.output), fourcc, args.fps, (w, h)) 
	run(args, model, keygen, detector, orig_mel, dataset, device, cap)

	if args.output:
		cmd = f'ffmpeg -i {os.path.join(args.path, args.output)} -i {wavpath} -c:v copy -c:a aac -strict experimental -strftime 1 {os.path.join(args.path, "output.mp4")}'
		subprocess.Popen(cmd, shell=True)   
		subprocess.Popen(f'rm {os.path.join(args.path, args.output)}', shell=True) 