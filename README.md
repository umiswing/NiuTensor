# 安装

CPU版本：
```bash
mkdir build && cd build && cmake ..
```

GPU版本：
```bash
cmake -DUSE_CUDA=ON -DUSE_HALF_PRECISION=ON -DCUDA_TOOLKIT_ROOT="/home/huchi/cuda-11.2/" .. && make -j
```

注意：目前只在Titan V和RTX上进行测试，因此不需要指定GPU架构，默认为多个架构生成目标代码

Windows上会生成NiuTensor.sln，打开后右键解决方案中的NiuTensor，选为启动项目，按F5编译

# 翻译

## GPU

```bash
cd data
../bin/NiuTensor -nmt -dev 0 -fp16 1 -model ../data/model.fp16 -srcvocab ../data/vocab.en -tgtvocab ../data/vocab.en < test.txt > output.txt
```

## CPU

```bash
cd data
../bin/NiuTensor -nmt -dev -1 -model ../data/model.fp32 -srcvocab ../data/vocab.en -tgtvocab ../data/vocab.en < test.txt > output.txt
```

参数说明：

dev: 设备号，>=0为显卡，-1为CPU

fp16: 使用半精度，注意不能忽略后面的1

model: 模型文件路径

srcvocab: 源语词表

tgtvocab：目标语词表

input：输入文件路径

output：输出文件路径

wbatch: batch大小 （源语最大词数）

# 评估翻译

```bash
sed -r 's/(@@ )|(@@ ?$)//g' < res.txt > output.txt && sacrebleu -i output.txt -t wmt20 -l en-de
```