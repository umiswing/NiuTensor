# 安装

CPU版本（无MKL/OpenBLAS）：
```bash
mkdir build && cd build && cmake .. && make -j
```

CPU版本（MKL）：
```bash
mkdir build && cd build && cmake -DUSE_MKL=ON -DINTEL_ROOT="/home/huchi/mkl" .. && make -j
```

GPU版本：
```bash
cmake -DUSE_CUDA=ON -DUSE_HALF_PRECISION=ON -DCUDA_TOOLKIT_ROOT="/home/huchi/cuda-11.2/" .. && make -j
```

注意：CUDA最低版本为9.2

Windows上去掉`make -j`命令，生成`NiuTensor.sln`，打开后右键解决方案中的`NiuTensor`，选为启动项目，按F5编译

# 训练
./bin/NiuTensor -train tools/train.data -valid tools/valid.data -dev 1 -model dlcl -enchistory 1 -shareallemb 1 -enclayer 35 -declayer 6 -maxrp 8 -wbatch 512 -updatefreq 8

增量训练与正常训练命令一致，只需要保证`-model`指向的模型文件存在即可读取并继续训练。


# 翻译

./bin/NiuTensor -dev 0 -srcvocab vocab.en -tgtvocab vocab.en -sbatch 1024 -wbatch 20480 -model model.bin -input x.txt -output y.txt