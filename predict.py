import torch
import torch.nn as nn
from model import CNN
from torchvision import transforms
import argparse
from PIL import Image


# 创建 ArgumentParser 对象
parser = argparse.ArgumentParser()
# 添加命令行参数
parser.add_argument('--images', dest='input_images', type=str, required=True, help='输入文件的路径')
args = parser.parse_args()
input_image = args.input_images

source = [str(i) for i in range(0, 10)]
source += [chr(i) for i in range(97, 97+26)]
source += [chr(i) for i in range(65,65+26)]
alphabet = ''.join(source)

model_path = "./model/model.pth"
img = Image.open(input_image)
trans = transforms.Compose([
    transforms.Resize([40, 120]),
    transforms.ToTensor()
])
img_tensor = trans(img)
cnn = CNN()
if torch.cuda.is_available():
    cnn = cnn.cuda()
    cnn.eval()
    cnn.load_state_dict(torch.load(model_path))
else:
    cnn.eval()
    model = torch.load(model_path, map_location='cpu')
    cnn.load_state_dict(model)
img_tensor = img_tensor.view(1, 3, 40, 120)
output = cnn(img_tensor.to("cuda:0"))
output = output.view(-1, 62)
output = nn.functional.softmax(output, dim=1)
output = torch.argmax(output, dim=1)
output = output.view(-1, 4)[0]
label = ''.join([alphabet[i] for i in output.cpu().numpy()])

print("label:", label)