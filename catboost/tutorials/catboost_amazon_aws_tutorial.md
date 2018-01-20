CatBoost on Amazon Web Services(AWS)
====================================
We created a preconfigured Amazon Machine Image with installed CatBoost and [Epsilon](https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary.html#epsilon) data pool. It supports training with both CPU and GPU. You may exploit acceleration of one or several GPU compute units to significantly speed up the training.

To start the image on Amazon Web Services follow the next steps:

1. Switch to US West (Oregon) region -> EC2 console -> Launch Instance -> in Community AMIs, search CatBoost, select: CatBoost v0.5 CPU/GPU with Epsilon pool -> Instance Type:p2.xlarge(GPU compute) -> Instance details(as is) -> Storage: 100Gb => Review and Launch

2. ssh into the started instance, and use the following examples to start training with GPU acceleration:
- command line version: "cd ~/catboost; ./train_epsilon10K_gpu.sh"
- python version: "cd ~/catboost; python train_epsilon10K_gpu.py"
- for CPU only training check train_epsilon10K_cpu.sh and train_epsilon10K_cpu.py
- for performance run use full version of Epsilon pool and scripts train_epsilon_gpu.sh or train_epsilon1_gpu.py or CPU analogues. Please be patient, it may take a while (up to 40min) for the very first read of the pool due Amazon disk replication mechanism.

The instance contains:
- CatBoost (at /home/ubuntu/catboost/catboost)
- Epsilon pool in TSV format (/home/ubuntu/catboost/epsilon_normalized.pool) and it's short version (first 10K lines) at (/home/ubuntu/catboost/epsilon_normalized10K.pool)
- Command line and python scripts to start training on Epsilon pool using CatBoost on CPU or GPU (at /home/ubuntu/catboost)

