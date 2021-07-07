 
from tqdm import tqdm 
from model import KeyGenerator, Wav2Lip
import audio
import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
from torch.utils import data as data_utils
import numpy as np
from glob import glob
import os, cv2, argparse 
import face_detection as fd
import warnings
import librosa
from collections import deque
from loss import MDFLoss

#########################################################################################

warnings.filterwarnings(action='ignore') 
parser = argparse.ArgumentParser(description='Code to train the expert lip-sync discriminator')

parser.add_argument('--checkpoint_dir', help='Save checkpoints to this directory', default='./checkpoints/', type=str)
parser.add_argument('--checkpoint_path', help='Resume from this checkpoint', default='./checkpoints/train_checkpoint.pth') 
parser.add_argument('--fps', help='image frame per second', default=25, type=int) 
parser.add_argument('--batch_size', help='train data batch size', default=32, type=int) 
parser.add_argument('--lr', help='optimizer learning rate', default=1e-4, type=float) 
parser.add_argument('--num_workers', help='dataloader workers', default=6) 
parser.add_argument('--epoch', help='number of train epochs', default=500) 
parser.add_argument('--multigpu', help='multigpu', default=True) 
parser.add_argument('--keypoint_model', dest='keypoint', help='keypoint model checkpoint path', default='./checkpoints/keypoint_checkpoint.pth') 
parser.add_argument('--sample_rate', help='sampling rate of audio file', default=16000, type=int)  
parser.add_argument('--train_dataset_path', dest='train_path', help='train data dir path', default='../data/train/', type=str) 
parser.add_argument('--test_dataset_path', dest='test_path', help='test data dir path', default='../data/test', type=str)  
args = parser.parse_args() 

global_step = 0
global_epoch = 0
train_loss = 1
min_loss = 1 
 
def dump(string, file='error.txt'):
    with open(file, mode='a', encoding='utf-8') as e:
        e.write(string + '\n')
  
class Dataset(object):
    def __init__(self, 
                path=None,  
                args=args, 
                img_size = 128, 
                ch = 3, 
                mul = 5,
                frames = 5,
                time_delay = 15,
                mel_step_size = 16
                ):
        '''
        model inferences when path is None. 
        '''
        self.config = args   
        self.path = path  
        if path:  
            self.dir = glob(os.path.join(self.path, '*' + os.path.sep))[:500] # list of image folders
            self.data = deque()
            print(f'number of dirs : {len(self.dir)}')
            for p in self.dir: 
                # files = sorted(list(glob(os.path.join(self.path, p, '*.png'))), key= lambda x : int(x[:x.find('.')].split('/')[-1])) 
                files = sorted(list(glob(os.path.join(self.path, p, '*.jpg'))), key= lambda x : int(os.path.basename(x)[:os.path.basename(x).find('.')]))
                # self.data.extend(files[:-(frames * (mul-2))])
                # self.data.extend(files[:-(frames * (mul - 1))])
                self.data.extend(files)
            print(f'all image data is {len(self.data)}')

        '''subprocess.call(
            f"ffmpeg -i {os.path.join(path,i)} -ab 160k -ac 1 -ar 16000 -vn {os.path.join(out, name + '.wav')}"\
                ,shell=True) '''
         
        self.img_size = img_size
        self.ch = ch
        self.frames = frames * mul,
        self.mul = mul,
        self.mel_step_size = mel_step_size * mul
        self.detector = fd.face_detection('whole_face')
        self.time_delay = time_delay 
        self.audio = audio.Audio()

    def get_frame_id(self, frame):
        return int(os.path.basename(frame).split('.')[0])

    def get_window(self, start_frame, img_names): 
        start_id = start_frame   # self.get_frame_id(start_frame)
         
        window_fnames = []
        for frame_id in range(start_id, start_id + self.frames): 
            if frame_id >= len(img_names) - 1 :
                break
            frame = img_names[frame_id] 
            if not isfile(frame):
                return None
            window_fnames.append(frame)
        return window_fnames

    def crop_audio_window(self, spec, start_frame, end=False):
        # num_frames = (T x hop_size * fps) / sample_rate
        # self.get_frame_id(start_frame)  
        start_idx = int(80. * (start_frame / float(self.config.fps))) # fps =25 
        end_idx = start_idx + self.mel_step_size  # 16
        if end : 
            return spec[start_idx: end_idx, :], start_idx, end_idx
        else:
            return spec[start_idx: end_idx, :]
        
    def getTilt(self, keypointsMean):
        # Remove in plane rotation using the eyes
        eyes = np.array(keypointsMean[36:48])
        x = eyes[:, 0]
        y = -1 * eyes[:, 1] 
        m = np.polyfit(x, y, 1)
        tilt = np.degrees(np.arctan(m[0]))
        return tilt

    def getKeypointFeatures(self, keypoints):
        # Mean Normalize the keypoints wrt the center of the mouth
        # Leads to face position invariancy
        mouth_kp_mean = np.average(keypoints[48:68])
        keypoints_mn = keypoints - mouth_kp_mean
        
        # Remove tilt
        x_dash = keypoints_mn[:, 0]
        y_dash = keypoints_mn[:, 1]
        theta = np.deg2rad(getTilt(keypoints_mn))
        c = np.cos(theta);	s = np.sin(theta)
        x = x_dash * c - y_dash * s	# x = x'cos(theta)-y'sin(theta)
        y = x_dash * s + y_dash * c # y = x'sin(theta)+y'cos(theta)
        keypoints_tilt = np.hstack((x.reshape((-1,1)), y.reshape((-1,1))))

        # Normalize
        N = np.linalg.norm(keypoints_tilt, 2)
        return [keypoints_tilt/N, N, theta, mouth_kp_mean]

    def preprocess(self, keypoints):
        '''preprocess keypoints and store tilt, mean data'''
        key = np.vstack([keypoints[:17],keypoints[48:]]) 
        mouthMean = np.average(key, 0)

        # mouthMean = np.average(keypoints[48:68], 0)
        keypointsMean = keypoints - mouthMean

        xDash = keypointsMean[:, 0]
        yDash = keypointsMean[:, 1]

        theta = np.deg2rad(self.getTilt(keypointsMean))

        c = np.cos(theta);	
        s = np.sin(theta)

        x = xDash * c - yDash * s	# x = x'cos(theta)-y'sin(theta)
        y = xDash * s + yDash * c   # y = x'sin(theta)+y'cos(theta)

        keypointsTilt = np.hstack((x.reshape((-1, 1)), y.reshape((-1, 1))))

        # Normalize
        N = np.linalg.norm(keypointsTilt, 2)
        
        #print N
        keypointsNorm = keypointsTilt / N
        # key = keypointsNorm[:17]
        landmark = np.vstack([keypointsNorm[:17],keypointsNorm[48:]])  
        # landmark = keypointsNorm[48:68]  
        return landmark, [N, theta, mouthMean, keypoints]
    
    def load_image(self, img, return_img=False, mask=False):
        try:
            img = cv2.imread(img) 
        except TypeError:
            pass
        except:
            raise 'image read error'
        img = cv2.resize(img, (self.img_size, self.img_size))  
        img_for_landmark = img.copy()  
        if mask : 
            img[self.img_size//2:,:,:] = 0 

        if return_img:
            return torch.FloatTensor(img / 255.), img_for_landmark
        return torch.FloatTensor(img / 255.) 

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):  
        file, img = os.path.dirname(self.data[idx]), os.path.basename(self.data[idx])  
        name, ex = self.get_frame_id(img), '.jpg'

        chosen = int(name) - 1
        adjust = 0
        while True:
            id = chosen + self.time_delay - adjust  
            aligned = os.path.join(file, img) # target image path  
            if  os.path.isfile(aligned):  
                break
            elif id <= chosen:
                break
            else:
                adjust += 1
        cnt = 0

        # load audio. try twice just in case to avoid error
        try:   
            while cnt < 2:
                try: 
                    wavpath = os.path.join(file, "audio.wav")
                    if not os.path.isfile(wavpath):
                        print(f'audio not exists {wavpath}') 
                    wav = librosa.load(wavpath, self.config.sample_rate)[0]
                    orig_mel = self.audio.melspectrogram(wav).T
                    mel = self.crop_audio_window(orig_mel.copy(), chosen) 
                    break
                except:
                    cnt+=1
            mel = torch.FloatTensor(mel.T)  
        except: 
            dump(f'audio load error : {wavpath}')   
        
        # mel shape match
        if mel.size(-1) != self.mel_step_size :      
            mel = F.pad(mel, (0, self.mel_step_size - mel.size(-1)), "constant", 0)  
        '''
        if mel.size(-1) != self.mel_step_size :      
            if mel.size(-1) < self.mel_step_size // 2 :  
                start_idx = int(80. * (chosen / float(self.config.fps))) # fps =25
                end_idx = start_idx + self.mel_step_size  
                dump(f'{self.data[idx]} : chosen {chosen} from {start_idx} to {end_idx} orig {orig_mel.shape} mel {mel.shape} -> new mel shape {torch.FloatTensor(orig_mel[-self.mel_step_size:]).shape}')
            try:
                tmp = torch.FloatTensor(orig_mel[-self.mel_step_size:])  
                assert tmp.size(1) == self.mel_step_size
                mel = tmp 
            except:
                mel = F.pad(mel, (0, self.mel_step_size - mel.size(-1)), "constant", 0)  '''
        # load image
        try: 
            masked = self.load_image(aligned, mask=True)
            original, np_img = self.load_image(aligned, return_img=True) 
            # original = original.reshape(self.ch, self.img_size, self.img_size)
            original = original.permute(2,0,1)
            landmark = self.detector.landmark_detection(np_img)  
            landmark, data = self.preprocess(landmark)
        except Exception as e: 
            dump(f'image load error : {self.data[idx]} {e}')  
        return mel, masked, original, data
 
l1_loss = nn.L1Loss()
# wing_loss = AdaptiveWingLoss() 
mdf_loss = MDFLoss()

def getOriginalKeypoints(kp_features_mouth, N, tilt, mean):
    '''return original keypoints from aligned and normalized keypoints''' 

    kp_dn = N * torch.cat((kp_features_mouth[:17], kp_features_mouth[17:] * 1.3), axis=0)
    # kp_dn = N * kp_features_mouth # * 1.5
    x, y = kp_dn[:, 0], kp_dn[:, 1]
    c, s = torch.cos(tilt), torch.sin(tilt)
    x_dash, y_dash = x*c + y*s, -x*s + y*c
    kp_tilt = torch.hstack((x_dash.view((-1,1)), y_dash.view((-1, 1))))
    kp = kp_tilt + mean
    return kp
 
def generate_target(joints, joints_vis, num_joints=68, heatmap_size=128, image_size=128, kpd=4.0, sigma=2, joints_weight=1, target_type='gaussian', return_torch=True, return_only_one_channel=True, use_different_joints_weight=False):
    '''
    :param joints:  [num_joints, 2]
    :param joints_vis: [num_joints, 1]
    
    if return_only_one_channel is True target returns [heatmap_size, heatmap_size] 0
    else [num_joints, heatmap_size, heatmap_size]
    
    :return: target, target_weight(1: visible, 0: invisible)
    '''
    # target_weight = np.ones((num_joints, 1), dtype=np.float32) 
    target_weight = joints_vis[:, :, 0]  
    batch = joints.shape[0]
    if target_type == 'gaussian':
        target = np.zeros((batch,
                            num_joints,
                            heatmap_size,
                            heatmap_size),
                            dtype=np.float32)

        tmp_size = sigma * 3
        for b in range(batch):
            for joint_id in range(num_joints):
                #Todo
                feat_stride = (image_size-1.0) / (heatmap_size-1.0)  
                # feat_stride = self.image_size / self.heatmap_size
                mu_x = int(joints[b][joint_id][0] / feat_stride + 0.5)
                mu_y = int(joints[b][joint_id][1] / feat_stride + 0.5) 
                # Check that any part of the gaussian is in-bounds
                ul = [int(mu_x - tmp_size), int(mu_y - tmp_size)]
                br = [int(mu_x + tmp_size + 1), int(mu_y + tmp_size + 1)]
                if ul[0] >= heatmap_size or ul[1] >= heatmap_size \
                        or br[0] < 0 or br[1] < 0:
                    # If not, just return the image as is
                    target_weight[b][joint_id] = 0
                    continue

                # # Generate gaussian
                size = 2 * tmp_size + 1
                x = np.arange(0, size, 1, np.float32)
                y = x[:, np.newaxis]
                x0 = y0 = size // 2
                #Todo
                mu_x_ac = joints[b][joint_id][0] / feat_stride 
                mu_y_ac = joints[b][joint_id][1] / feat_stride 
                x0 = y0 = size // 2
                x0 += mu_x_ac-mu_x
                y0 += mu_y_ac-mu_y
                # The gaussian is not normalized, we want the center value to equal 1
                g = np.exp(- ((x - x0) ** 2 + (y - y0) ** 2) / (2 * sigma ** 2))

                # Usable gaussian range
                g_x = max(0, -ul[0]), min(br[0], heatmap_size) - ul[0]
                g_y = max(0, -ul[1]), min(br[1], heatmap_size) - ul[1]
                # Image range
                img_x = max(0, ul[0]), min(br[0], heatmap_size)
                img_y = max(0, ul[1]), min(br[1], heatmap_size)

                v = target_weight[b][joint_id]
                if v > 0.5:
                    target[b][joint_id][img_y[0]:img_y[1], img_x[0]:img_x[1]] = \
                        g[g_y[0]:g_y[1], g_x[0]:g_x[1]]
        
    elif target_type == 'offset':
        # self.heatmap_size: [48,64] [w,h]
        target = np.zeros((num_joints,
                            3,
                            heatmap_size*
                            heatmap_size),
                            dtype=np.float32)
        feat_width = heatmap_size
        feat_height = heatmap_size
        feat_x_int = np.arange(0, feat_width)
        feat_y_int = np.arange(0, feat_height)
        feat_x_int, feat_y_int = np.meshgrid(feat_x_int, feat_y_int)
        feat_x_int = feat_x_int.reshape((-1,))
        feat_y_int = feat_y_int.reshape((-1,))
        kps_pos_distance_x = kpd
        kps_pos_distance_y = kpd
        feat_stride = (image_size - 1.0) / (heatmap_size - 1.0)
        for joint_id in range(num_joints):
            mu_x = joints[joint_id][0] / feat_stride 
            mu_y = joints[joint_id][1] / feat_stride 
            # Check that any part of the gaussian is in-bounds

            x_offset = (mu_x - feat_x_int) / kps_pos_distance_x
            y_offset = (mu_y - feat_y_int) / kps_pos_distance_y

            dis = x_offset ** 2 + y_offset ** 2
            keep_pos = np.where((dis <= 1) & (dis >= 0))[0]
            v = target_weight[joint_id]
            if v > 0.5:
                target[joint_id, 0, keep_pos] = 1
                target[joint_id, 1, keep_pos] = x_offset[keep_pos]
                target[joint_id, 2, keep_pos] = y_offset[keep_pos]
        target=target.reshape((joints_weight*3, heatmap_size, heatmap_size))
    if use_different_joints_weight:
        target_weight = np.multiply(target_weight, joints_weight)
    
    if return_only_one_channel:   
        landmark = np.zeros((batch, heatmap_size, heatmap_size))
        for i in range(batch):
            for j in range(num_joints):
                landmark[i] += target[i][j]
        target = (landmark.copy() - landmark.min()) / (landmark.max() - landmark.min())
    if return_torch: 
        target = torch.FloatTensor(target) 
    return target, target_weight

def train(device, model, keygenerator, train_data_loader, test_data_loader, optimizer,
          checkpoint_path=None, epoch=50):
    global global_step, global_epoch, min_loss, train_loss 
    eval_loss = 0
    verbose = len(train_data_loader) // 2
    
    while global_epoch < epoch:
        running_loss = 0.
        # prog_bar = tqdm(enumerate(train_data_loader))
        for step, (mel, mask, y, data) in enumerate(train_data_loader): 
            model.train() 
            keygenerator.eval()
            optimizer.zero_grad()

            # Transform data to CUDA device
            mel = mel.to(device)
            mask = mask.to(device)
            y = y.to(device)

            keypoints = keygenerator(mel)
            keypoints = keypoints.squeeze(1) #.cpu().detach()
            N, tilt, mean, y_keypoints = data  
            # k = np.zeros_like(keypoints)
            for i in range(len(keypoints)):
                try:
                    kps = getOriginalKeypoints(keypoints[i], N[i].to(device), tilt[i].to(device), mean[i].to(device)) 
                except Exception as e:
                    print(f'original keypoint restruction error : {e}')
                    break 
                # y_keypoints[i, :17] = kps[:17]
                # y_keypoints[i, 48:68] = kps[17:] 
                # k[i, :17] = kps[:17]
                # k[i, 17:] = kps[17:] 
                for j in range(len(kps)): 
                    mask[i, int(kps[j,1]), int(kps[j,0])] = 1 
            # lip and face edge synthesised keypoint
            # key = y_keypoints.cpu().detach().numpy().reshape(-1,68,2)  
            # target, target_weight = generate_target(key, np.ones_like(key[:,:,:1]))
            # target = target.unsqueeze(1).to(device) 
              
            mask = mask.permute(0,3,1,2)
            pred = model(mask)
            loss = mdf_loss(pred, y) + l1_loss(pred, y) 
            loss.backward()
            optimizer.step()  
            running_loss += loss.item()   

            if step % verbose == 0:
                print('Epoch: {} Loss: {} eval : {}'.format(global_epoch, running_loss / (step + 1), eval_loss))
                img = np.array(mask.permute(0,2,3,1).cpu().detach() * 255.).astype(int)   
                cv2.imwrite(f'./train.png', img[i]) 
                img = np.array(y.permute(0,2,3,1).cpu().detach() * 255.).astype(int)   
                cv2.imwrite(f'./label.png', img[i])

        txt = 'Epoch: {} Loss: {} eval : {}'.format(global_epoch, running_loss / (step + 1), eval_loss)
        dump(txt, 'train.txt')  
        if test_data_loader is not None:
            with torch.no_grad():
                eval_loss = eval_model(test_data_loader, device, model, keygenerator, checkpoint_path)
            if min_loss > eval_loss:
                min_loss = eval_loss  
                save_checkpoint(model, optimizer, global_step, checkpoint_path, global_epoch, min_loss, state='eval')
 
        if train_loss > running_loss / (step + 1): 
            print(f'train loss : {running_loss / (step + 1)}')

            save_checkpoint(model, optimizer, global_step, checkpoint_path, global_epoch, running_loss / (step + 1), state='train')
            img = np.array(pred.permute(0,2,3,1).cpu().detach() * 255.).astype(int)   

            for i in range(len(img)):
                try: 
                    # print(f'output img batch shape : {img.shape}')
                    cv2.imwrite(f'./result/reconstruct_{i}.png', img[i])
                except Exception as e:
                    print(f'{e} > at {i}')
            train_loss = running_loss / (step + 1) 
        global_epoch += 1
 
def eval_model(test_data_loader, device, model, keygenerator, checkpoint_path=None):

    losses = []  
    for (mel, mask, y, data) in test_data_loader:
        
        model.eval() 
        keygen.eval()  
        
        mel = mel.to(device)
        mask = mask.to(device)
        y = y.to(device) 

        keypoints = keygenerator(mel)
        keypoints = keypoints.to(device)
        keypoints = keypoints.squeeze(1) #.cpu().detach()
        N, tilt, mean, y_keypoints = data 

        # k = np.zeros_like(keypoints)
        for i in range(len(keypoints)):
            try:
                kps = getOriginalKeypoints(keypoints[i], N[i].to(device), tilt[i].to(device), mean[i].to(device)) 
            except Exception as e:
                print(f'original keypoint restruction error : {e}')
                break
            # y_keypoints[i, :17] = kps[:17]
            # y_keypoints[i, 48:68] = kps[17:] 
            # k[i, :17] = kps[:17]
            # k[i, 48:68] = kps[17:] 
            for j in range(len(kps)): 
                mask[i, int(kps[j,1]), int(kps[j,0])] = 1 
        '''
        # lip and face edge synthesised keypoint
        key = y_keypoints.cpu().detach().numpy().reshape(-1,68,2)  
        # target, target_weight = generate_target(key,np.ones_like(key[:,:,:1]))
        target, target_weight = generate_target(k, np.ones_like(k[:,:,:1]),num_joints=37)
        # target = target.unsqueeze(1).to(device) 
        target = target.unsqueeze(3).to(device) 
        mask = (mask + target).permute(0,3,1,2) '''
        mask = mask.permute(0,3,1,2)
        pred = model(mask)   

        loss = mdf_loss(pred, y) + l1_loss(pred, y)
        losses.append(loss.item()) 

        averaged_loss = sum(losses) / len(losses)
        print(f'eval loss: {averaged_loss}') 
            
        return averaged_loss
 
def save_checkpoint(model, optimizer, step, checkpoint_path, epoch, loss, state):
    # checkpoint_path = os.path.join(checkpoint_dir, f"{state}_checkpoint.pth")
    optimizer_state = optimizer.state_dict()
    torch.save({
        "state_dict": model.state_dict(),
        "optimizer": optimizer_state if state is 'train' else None,
        "global_step": step,
        "global_epoch": epoch,
        "min_loss": loss
    }, checkpoint_path)
    dump(f"Saved {state} checkpoint Epoch: {epoch} Loss: {loss}", 'train.txt')

def load_checkpoint(path, model, optimizer=None, reset_optimizer=False, multigpu=True):
    global global_step
    global global_epoch

    print("Load checkpoint from: {}".format(path))
    checkpoint = torch.load(path)
    new_s = {}
    s = checkpoint["state_dict"]
    if not multigpu:
        for k, v in s.items():
            new_s[k.replace('module.', '')] = v
        s = new_s
    model.load_state_dict(s)
    if not reset_optimizer:
        optimizer_state = checkpoint["optimizer"]
        if optimizer_state is not None:
            print("Load optimizer state from {}".format(path))
            optimizer.load_state_dict(checkpoint["optimizer"])
        global_step = checkpoint["global_step"]
        global_epoch = checkpoint["global_epoch"] + 1
        try:
            if optimizer:
                train_loss = checkpoint["min_loss"]
            else:
                min_loss = checkpoint["min_loss"]
        except:
            pass
    return model


if __name__ == "__main__":

    if not os.path.exists(args.checkpoint_dir): os.mkdir(args.checkpoint_dir)


    # Dataset and Dataloader setup
    train_dataset = Dataset(args.train_path)  
    if args.test_path:
        test_dataset = Dataset(args.test_path)  
    
    train_data_loader = data_utils.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers) #, drop_last=True
    try:
        test_data_loader = data_utils.DataLoader(
            test_dataset, batch_size=args.batch_size,
            num_workers=args.num_workers)    
    except:
        test_data_loader = None 
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  

    # Model
    model = Wav2Lip().to(device)
    keygen = KeyGenerator().to(device)


    if args.multigpu:
        model = nn.DataParallel(model) 
        keygen = nn.DataParallel(keygen)
        print('use dataparallel') 

    print('total trainable params {}'.format(sum(p.numel() for p in model.parameters() if p.requires_grad)))

    optimizer = optim.Adam([p for p in model.parameters() if p.requires_grad],
                           lr=args.lr)

    if os.path.isfile(args.checkpoint_path) :
        try:
            model = load_checkpoint(args.checkpoint_path, model, optimizer, reset_optimizer=False)
            print('wav2lip model loaded') 
        except:
            pass

    if args.keypoint is not None:
        try:
            keygen = load_checkpoint(args.keypoint, keygen, reset_optimizer=True)
            print('keypoint model loaded') 
        except:
            pass
    
    train(device, model, keygen, train_data_loader, test_data_loader, optimizer,
          checkpoint_path=args.checkpoint_path, 
          epoch=args.epoch)
