# 安装

CPU版本：
mkdir build && cd build && cmake ..

GPU版本：
mkdir build && cd build && cmake -DUSE_CUDA=ON -DCUDA_TOOLKIT_ROOT='C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v11.1' -DGPU_ARCH=P ..

注意：将cmake命令中的路径替换为本机路径，DGPU_ARCH在2080Ti上需要指定为T, 在Titan X上为M，在Titan xp或1080 Ti上为P

Windows上会生成NiuTensor.sln，打开后右键解决方案中的NiuTensor，选为启动项目，按F5编译

Linux上执行make -j

# 翻译
bin/NiuTensor -nmt -dev 0 -model model.bin -srcvocab vocab -tgtvocab vocab -input data/test.de -output res.txt -beam 1


参数说明：

dev: 设备号，>=0为显卡，-1为CPU

model: 模型文件路径

srcvocab: 源语词表

tgtvocab：目标语词表

input：输入文件路径

output：输出文件路径

beam: beam大小

# 评估翻译
sed -r 's/(@@ )|(@@ ?$)//g' < res.txt > output.txt

perl data/multi-bleu.perl data/gold.txt < output.txt