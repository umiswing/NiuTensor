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

## 指定输出输出文件

```bash
bin/NiuTensor -nmt -dev 7 -fp16 1 -model ../data/model.fp16 -srcvocab ../data/vocab -tgtvocab ../data/vocab -input ../data/en.txt -output ../data/res.txt 
```

## 读取标准输入并打印结果

```bash
bin/NiuTensor -nmt -dev 7 -fp16 1 -model ../data/model.fp16 -srcvocab ../data/vocab -tgtvocab ../data/vocab < input.txt
```

## 读取标准输入并重定向结果

```bash
bin/NiuTensor -nmt -dev 7 -fp16 1 -model ../data/model.fp16 -srcvocab ../data/vocab -tgtvocab ../data/vocab < input.txt > output.txt
```

参数说明：

dev: 设备号，>=0为显卡，-1为CPU

model: 模型文件路径

srcvocab: 源语词表

tgtvocab：目标语词表

input：输入文件路径

output：输出文件路径

wbatch: batch大小 （源语最大词数）

# 评估翻译

```bash
sed -r 's/(@@ )|(@@ ?$)//g' < res.txt > output.txt && perl multi-bleu.perl test.de < output.txt
```