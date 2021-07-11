# 安装

CPU版本：
```bash
mkdir build && cd build && cmake ..
```

GPU版本：
```bash
cmake -DUSE_CUDA=ON -DUSE_HALF_PRECISION=ON -DCUDA_TOOLKIT_ROOT="/home/huchi/cuda-10.2/" -DGPU_ARCH=V .. && make -j
```

注意：将cmake命令中的路径替换为本机路径，DGPU_ARCH在2080Ti上需要指定为T, 在Titan X上为M，在Titan xp或1080 Ti上为P

Windows上会生成NiuTensor.sln，打开后右键解决方案中的NiuTensor，选为启动项目，按F5编译

# 翻译
```bash
bin/NiuTensor -nmt -dev 7 -model ../data/model.fp16 -srcvocab ../data/vocab -tgtvocab ../data/vocab -input ../data/en.txt -output ../data/res.txt -beam 1 -sbatch 256 -fp16 1 -maxlenalpha 1.25
```

参数说明：

dev: 设备号，>=0为显卡，-1为CPU

model: 模型文件路径

srcvocab: 源语词表

tgtvocab：目标语词表

input：输入文件路径

output：输出文件路径

beam: beam大小

sbatch: batch大小

maxlenalpha: 最大句长（源语长度倍数）

注意：不使用半精度需要移除`-fp16 1`

# 评估翻译
```bash
sed -r 's/(@@ )|(@@ ?$)//g' < res.txt > output.txt && perl multi-bleu.perl test.de < output.txt
```