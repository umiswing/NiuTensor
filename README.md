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

第一次运行需要：

1. 把bin目录拷贝到data目录下，保证bin目录、moses目录和model目录都在data目录内

2. 修改fastbpe权限: chmod+x data/moses/fastbpe

## GPU

```bash
cd data
sh run.sh GPU throughput < wmt20.test.en > res.txt
```
注意：在run.sh中修改设备号

## CPU

```bash
cd data
sh run.sh CPU throughput < wmt20.test.en > res.txt
```

# 评估翻译

```bash
sacrebleu -i res.txt -t wmt20 -l en-de
```