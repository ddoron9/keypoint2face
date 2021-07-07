import torch
from torch import nn
from torch.nn import functional as F
 
 
   
class KeyGenerator(nn.Module):

    def __init__(self,
                 time_step=16 * 5,
                 n_mels=80,  
                 frame = 5 * 5,
                 keypoints = 37, 
                 multi_landmark = 1,
                 bidirection = True,
                 hidden_size=20
                 ):
        super(KeyGenerator,self).__init__()

        '''
        [batch, 1, 80, 16] => [batch, 5, 68, 2]
        ''' 
        self.lstm = nn.LSTM(input_size = time_step, hidden_size = hidden_size, batch_first = True, bidirectional=bidirection, num_layers=2)  
        self.flatten = nn.Flatten()
        out = n_mels * hidden_size * (2 if bidirection else 1)
        self.fc1 = nn.Linear(out, out // 2)
        self.fc2 = nn.Linear(out // 2, out // 4)
        self.fc3 = nn.Linear(out // 4, keypoints * 2)
  
        self.relu = nn.LeakyReLU()
        self.drops = nn.Dropout(0.2) 
        self.frame = frame
        self.keypoints = keypoints 
        self.multi_landmark = multi_landmark
 
    
    def forward(self,x): 
        x, _ = self.lstm(x)
        x = self.relu(x)  
        x = self.flatten(x)    
        x = self.drops(self.relu(self.fc1(x)))
        x = self.drops(self.relu(self.fc2(x)))
        x = self.relu(self.fc3(x))
        return x.view(-1, self.multi_landmark, self.keypoints, 2)
 
class Conv2d(nn.Module):
    def __init__(self, cin, cout, kernel_size, stride, padding, residual=False, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.conv_block = nn.Sequential(
                            nn.Conv2d(cin, cout, kernel_size, stride, padding),
                            nn.BatchNorm2d(cout)
                            )
        self.act = nn.ReLU()
        self.residual = residual

    def forward(self, x):
        out = self.conv_block(x)
        if self.residual:
            out += x
        return self.act(out)

class nonorm_Conv2d(nn.Module):
    def __init__(self, cin, cout, kernel_size, stride, padding, residual=False, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.conv_block = nn.Sequential(
                            nn.Conv2d(cin, cout, kernel_size, stride, padding),
                            )
        self.act = nn.LeakyReLU(0.01, inplace=True)

    def forward(self, x):
        out = self.conv_block(x)
        return self.act(out)

class Conv2dTranspose(nn.Module):
    def __init__(self, cin, cout, kernel_size, stride, padding, output_padding=0, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.conv_block = nn.Sequential(
                            nn.ConvTranspose2d(cin, cout, kernel_size, stride, padding, output_padding),
                            nn.BatchNorm2d(cout)
                            )
        self.act = nn.ReLU()

    def forward(self, x):
        out = self.conv_block(x)
        return self.act(out) 


class Wav2Lip(nn.Module):
    def __init__(self):
        super(Wav2Lip, self).__init__()

        self.face_encoder_blocks = nn.ModuleList([
            nn.Sequential(Conv2d(3, 16, kernel_size=7, stride=1, padding=3)), # torch.Size([16, 16, 128, 128])
            nn.Sequential(Conv2d(16, 32, kernel_size=3, stride=2, padding=1), 
            Conv2d(32, 32, kernel_size=3, stride=1, padding=1, residual=True),
            Conv2d(32, 32, kernel_size=3, stride=1, padding=1, residual=True)), # torch.Size([16, 32, 64, 64])

            nn.Sequential(Conv2d(32, 64, kernel_size=3, stride=2, padding=1),   
            Conv2d(64, 64, kernel_size=3, stride=1, padding=1, residual=True),
            Conv2d(64, 64, kernel_size=3, stride=1, padding=1, residual=True),
            Conv2d(64, 64, kernel_size=3, stride=1, padding=1, residual=True)), # torch.Size([16, 64, 32, 32])

            nn.Sequential(Conv2d(64, 128, kernel_size=3, stride=2, padding=1),  
            Conv2d(128, 128, kernel_size=3, stride=1, padding=1, residual=True),
            Conv2d(128, 128, kernel_size=3, stride=1, padding=1, residual=True)), # torch.Size([16, 128, 16, 16])

            nn.Sequential(Conv2d(128, 256, kernel_size=3, stride=2, padding=1),      
            Conv2d(256, 256, kernel_size=3, stride=1, padding=1, residual=True),
            Conv2d(256, 256, kernel_size=3, stride=1, padding=1, residual=True)),  # torch.Size([16, 256, 8, 8])

            nn.Sequential(Conv2d(256, 512, kernel_size=3, stride=2, padding=1),    
            Conv2d(512, 512, kernel_size=3, stride=1, padding=1, residual=True),), # torch.Size([16, 512, 4, 4])
            
            nn.Sequential(Conv2d(512, 512, kernel_size=3, stride=1, padding=0),     
            Conv2d(512, 512, kernel_size=1, stride=1, padding=0)),]) # torch.Size([16, 512, 2, 2])
 

        self.face_decoder_blocks = nn.ModuleList([
            nn.Sequential(Conv2d(512, 512, kernel_size=1, stride=1, padding=0),), # torch.Size([16, 512, 2, 2])

            nn.Sequential(Conv2dTranspose(1024, 512, kernel_size=3, stride=1, padding=0),
            Conv2d(512, 512, kernel_size=3, stride=1, padding=1, residual=True),), # torch.Size([16, 512, 4, 4])

            nn.Sequential(Conv2dTranspose(1024, 512, kernel_size=3, stride=2, padding=1, output_padding=1),
            Conv2d(512, 512, kernel_size=3, stride=1, padding=1, residual=True),
            Conv2d(512, 512, kernel_size=3, stride=1, padding=1, residual=True),), # torch.Size([16, 512, 8, 8])

            nn.Sequential(Conv2dTranspose(768, 384, kernel_size=3, stride=2, padding=1, output_padding=1),
            Conv2d(384, 384, kernel_size=3, stride=1, padding=1, residual=True),
            Conv2d(384, 384, kernel_size=3, stride=1, padding=1, residual=True),), # torch.Size([16, 384, 16, 16])

            nn.Sequential(Conv2dTranspose(512, 256, kernel_size=3, stride=2, padding=1, output_padding=1),
            Conv2d(256, 256, kernel_size=3, stride=1, padding=1, residual=True),
            Conv2d(256, 256, kernel_size=3, stride=1, padding=1, residual=True),), # torch.Size([16, 256, 32, 32])

            nn.Sequential(Conv2dTranspose(320, 128, kernel_size=3, stride=2, padding=1, output_padding=1), 
            Conv2d(128, 128, kernel_size=3, stride=1, padding=1, residual=True),
            Conv2d(128, 128, kernel_size=3, stride=1, padding=1, residual=True),),  # torch.Size([16, 128, 64, 64])

            nn.Sequential( # Conv2dTranspose(160, 64, kernel_size=3, stride=2, padding=1, output_padding=1),  
            Conv2d(160, 256, kernel_size=3, stride=1, padding=1),
            nn.PixelShuffle(2), 
            ),])  # torch.Size([16, 64, 128, 128])
            #Conv2d(40, 64, kernel_size=3, stride=1, padding=1),
        self.output_block = nn.Sequential(Conv2d(80, 32, kernel_size=3, stride=1, padding=1),
            nn.Conv2d(32, 3, kernel_size=1, stride=1, padding=0),
            nn.Hardsigmoid()) 
        self.pixel = nn.PixelShuffle(2)
    def forward(self, x):
         
        feats = [] 
         
        for f in self.face_encoder_blocks:
            x = f(x)   
            feats.append(x)  

        for f in self.face_decoder_blocks:
            x = f(x) 
            try: 
                x = torch.cat((x, feats[-1]), dim=1)  
            except Exception as e: 
                # print(feats[-1].size())
                raise e
            
            feats.pop()    
        outputs = self.output_block(x)   
        return outputs

if __name__ == '__main__':
    from torchsummary import summary
    m = Wav2Lip().to('cuda')

    print(summary(m, (3,128,128)))
    exit()

    '''
    face = torch.rand((16, 3, 128, 128))  
    y = m(face)
    print(y.shape)
    '''