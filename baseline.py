import torch
import numpy as np
from PIL import Image
from skimage import io, transform
from lib import modules
import math
import matplotlib.pyplot as plt

# 1. 加载和预处理图像
# image_buildings = np.asarray(io.imread("/home/weiming/llama/Dataprocess/RadioMapSeer/png/buildings_complete/0.png"))
# image_gain = np.asarray(io.imread("/home/weiming/llama/Dataprocess/RadioMapSeer/gain/DPM/0_0.png"))
image_buildings = np.asarray(io.imread("/home/weiming/RUnet_baseline/RadioUNet-master/building.png"))
image_gain = np.asarray(io.imread("/home/weiming/RUnet_baseline/RadioUNet-master/rm.png"))

# 确保图像尺寸为256x256
if image_buildings.shape[0] != 256 or image_buildings.shape[1] != 256:
    image_buildings = transform.resize(image_buildings, (256, 256))
if image_gain.shape[0] != 256 or image_gain.shape[1] != 256:
    image_gain = transform.resize(image_gain, (256, 256))

# 对image_gain进行处理
image_gain = np.expand_dims(image_gain, axis=2) / 256.0

# # 阈值处理
# thresh = 0.2
# mask = image_gain < thresh
# image_gain[mask] = thresh
# image_gain = image_gain - thresh * np.ones(np.shape(image_gain))
# image_gain = image_gain / (1 - thresh)
image_gain = image_gain * 256  # 归一化

# 归一化buildings
image_buildings = image_buildings / 256.0

# 3. 加载预训练模型
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = modules.RadioWNet(inputs=2, phase="secondU")
model.load_state_dict(torch.load('RadioWNet_s_DPMcars_carInput_Thr2/200samples_SecondU.pt', weights_only=True))
model.to(device)
model.eval()

# 添加实验统计变量
total_avg_distance = 0
valid_experiments = 0
num_experiments = 1000

samples_range = [200, 220, 240,260, 280, 300]
average_distances = []

for num_samples in samples_range:
    total_avg_distance = 0
    valid_experiments = 0
    
    for experiment in range(num_experiments):
        # 构建采样点
        image_samples = np.zeros((256, 256))
        x_samples = np.random.randint(0, 255, size=num_samples)
        y_samples = np.random.randint(0, 255, size=num_samples)
        image_samples[x_samples, y_samples] = image_gain[x_samples, y_samples, 0]

        # 构建模型输入
        inputs = np.stack([image_buildings, image_samples], axis=0)
        inputs = torch.FloatTensor(inputs).unsqueeze(0)

        # 进行预测
        with torch.no_grad():
            inputs = inputs.to(device)
            pred1, pred = model(inputs)
            pred = (pred.detach().cpu().numpy()).astype(np.uint8)

        # 计算平均距离
        center = (128, 128)
        total_distance = 0
        pixel_count = 0
        image = pred[0][0]
        
        for y in range(98, 158):  
            for x in range(98, 158): 
                if image[y, x] == 0:
                    distance = math.sqrt((x - center[0]) ** 2 + (y - center[1]) ** 2)
                    total_distance += distance
                    pixel_count += 1
        
        if pixel_count > 0:
            average_distance = total_distance / pixel_count
            total_avg_distance += average_distance
            valid_experiments += 1
    
    if valid_experiments > 0:
        final_average = total_avg_distance / valid_experiments
        average_distances.append(final_average)
        print(f"采样点数量 {num_samples} 的平均距离: {final_average:.2f}")
    else:
        average_distances.append(0)
        print(f"采样点数量 {num_samples} 没有有效结果")

# 绘制曲线
plt.figure(figsize=(10, 6))
plt.plot(samples_range, average_distances, marker='o')
plt.xlabel('Number of Samples')
plt.ylabel('Average Distance')
plt.title('Average Distance vs Number of Samples')
plt.grid(True)
plt.savefig('average_distance_curve.png')
plt.close()