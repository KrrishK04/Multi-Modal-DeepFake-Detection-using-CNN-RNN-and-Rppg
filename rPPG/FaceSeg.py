
import torch
from torch.autograd import Variable
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.enabled = True
from models import UNet16, UNet11
class FaceSegGPU:
    def get_face(self, frame):
        import face_recognition
        face_locations = face_recognition.face_locations(frame)
        if face_locations:
            top, right, bottom, left = face_locations[0]
            return (left, top, right, bottom)
        else:
            return None
    def __init__(self, bs, size=256):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.net = UNet11(pretrained=True).to(self.device)
        # If you need to load weights: self.net.load_state_dict(torch.load('unet_celeba.pth'))
        # self.net = UNet16(pretrained=True).to(self.device)
        # self.net.load_state_dict(torch.load('unet16.pth'))
        self.net.eval()
        sample = Variable(torch.rand(bs,3,size,size).to(self.device))
        self.net(sample)
        #  = torch.jit.trace(self.net, sample)
        print('___init___')
    
    def get_mask(self, images, shape):
        # images = Variable(torch.tensor(images, dtype=torch.float,requires_grad=False).to(device=self.device))
        pred = self.net(images)
        pred= torch.nn.functional.interpolate(pred, size=[shape[1], shape[2]])
        pred = pred.squeeze()
        mask = (pred > 0.8)
        segmentation = mask.cpu().numpy()
        return segmentation.astype('float')

    def apply_masks(self, frames_transformed, frames):
        masks = self.get_mask(frames_transformed, frames.shape)
        frames[masks==0] = 0.0
        return frames
