# 🖥️ Slurm 集群常用指令速查手册

> 📌 **核心原则**：永远不要在登录节点（Login Node）运行繁重计算任务，所有计算都通过Slurm排队到计算节点！
>
> ---
>
> ## 📋 目录
>
> - [一、资源查看类](#-一资源查看类)
> - [二、交互式申请GPU（srun）](#-二交互式申请gpusrun)
> - [三、批处理作业（sbatch）](#-三批处理作业sbatch)
> - [四、环境配置类](#-四环境配置类)
> - [五、文件与目录操作](#-五文件与目录操作)
> - [六、进入GPU节点后的标准流程](#-六进入gpu节点后的标准流程)
> - [七、实用技巧](#-七实用技巧)
> - [八、常见问题速查](#-八常见问题速查)
>              
>   
>
> ## 📊 一、资源查看类
>
> | 命令 | 说明 |
> |:-----|:-----|
> | `sinfo` | 查看所有节点状态（IDLE=空闲, MIX=部分占用, ALLOC=满载, DRAIN/DOWN=故障）|
> | `sinfo -o "%n %P %G %C"` | 查看每个节点的具体GRES配置，确认A100标识符 |
> | `squeue` | 查看所有正在排队的作业 |
> | `squeue -u $USER` | 只查看自己的作业（R=运行中, PD=排队中, CG=正在结束）|
> | `scontrol show job 123456` | 查看指定作业的详细信息（预计开始时间、申请资源等）|
> | `scontrol show node gpu3` | 查看指定节点的详细状态 |
> | `nvidia-smi` | 进入GPU节点后，确认是否拿到GPU及显存大小 |
> | `module avail cuda` | 检查可用的CUDA版本 |
>
> ---
>
> ## 🚀 二、交互式申请GPU（srun）
>
> ### ⚡ 快速抢占命令
>
> ```bash
> # 策略：指定gpu3，只要1个CPU核，30分钟，触发Backfill快速插队
> srun -p gpu --nodelist=gpu3 --gres=gpu:1 --cpus-per-task=1 --time=00:30:00
> ```
>
> ### 🎯 标准申请命令
>
> ```bash
> # 申请gpu5节点，1张A100，4个CPU核，2小时
> srun -p gpu --nodelist=gpu5 --gres=gpu:1 --cpus-per-task=4 --time=02:00:00 --pty /bin/bash
> ```
>
> ### 🔒 独占模式（大模型推荐）
>
> ```bash
> # 独占整个节点，获得全部4张A100和所有CPU
> srun -p gpu --nodelist=gpu5 --gres=gpu:1 --exclusive --pty /bin/bash
> ```
>
> ### 📝 参数说明
>
> | 参数 | 含义 |
> |:-----|:-----|
> | `-p gpu` | 指定分区为gpu |
> | `--nodelist=gpu5` | 指定具体节点 |
> | `--gres=gpu:1` | 申请1张GPU |
> | `--cpus-per-task=4` | 申请4个CPU核心 |
> | `--time=02:00:00` | 最长运行2小时 |
> | `--exclusive` | 独占模式，避免资源争抢 |
> | `--pty /bin/bash` | 进入交互式Shell |
>
> ---
>
> ## 📝 三、批处理作业（sbatch）
>
> ### 提交与管理
>
> | 命令 | 说明 |
> |:-----|:-----|
> | `sbatch job.slurm` | 提交批处理脚本 |
> | `scancel 123456` | 取消指定ID的作业 |
> | `scancel -u $USER` | ⚠️ 慎用！取消自己所有作业 |
> | `tail -f job_123456.out` | 实时查看作业输出日志 |
>
> ### 📄 SBATCH 脚本模板
>
> ```bash
> #!/bin/bash
> #SBATCH --job-name=llm_quant         # 作业名称
> #SBATCH --partition=gpu              # 分区名
> #SBATCH --nodelist=gpu[5-8]          # 锁定A100节点
> #SBATCH --gres=gpu:1                 # 申请1张GPU
> #SBATCH --cpus-per-task=16           # 申请16个CPU核（喂饱GPU）
> #SBATCH --mem=64G                    # 申请内存
> #SBATCH --time=24:00:00              # 最长24小时
> #SBATCH --exclusive                  # 独占模式
> #SBATCH --output=logs/%j.out         # 标准输出（%j=作业ID）
> #SBATCH --error=logs/%j.err          # 错误输出
>
> # 1. 加载模块
> module purge
> module load anaconda3
>
> # 2. 激活环境
> source activate /beegfs/general/your_username/envs/llm_env
>
> # 3. 进入工作目录
> cd /beegfs/general/your_username/workspace/project_name
>
> # 4. 运行程序
> python train.py --batch_size 8 --lr 2e-5
> ```
>
> ---
>
> ## 🔧 四、环境配置类
>
> ### Shell 切换
>
> ```bash
> bash                    # 从tcsh切换到bash（避免报错）
> echo $0                 # 查看当前Shell类型
> ```
>
> ### Conda 环境管理
>
> ```bash
> # 加载Conda模块
> module load anaconda3
>
> # 验证Conda版本
> conda --version
>
> # 创建新环境（指定路径避免Home目录爆满）
> conda create --prefix /beegfs/general/your_username/envs/llm_env python=3.10 -y
>
> # 激活环境
> source activate /beegfs/general/your_username/envs/llm_env
> # 或
> conda activate /beegfs/general/your_username/envs/llm_env
>
> # 查看已有环境
> conda env list
> ```
>
> ### PyTorch 安装（CUDA支持）
>
> ```bash
> pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
> pip install transformers accelerate bitsandbytes scipy safetensors
> ```
>
> ### 验证GPU可用
>
> ```python
> python -c "import torch; print(f'GPU Available: {torch.cuda.is_available()}')"
> ```
>
> ---
>
> ## 📁 五、文件与目录操作
>
> ### 目录结构
>
> ```bash
> # 查看磁盘使用情况
> df -h
>
> # 创建工作目录
> mkdir -p /beegfs/general/your_username
>
> # 创建快捷方式（软链接）
> ln -s /beegfs/general/your_username ~/work
>
> # 之后直接用
> cd ~/work
> ```
>
> ### 常用路径
>
> | 路径 | 用途 |
> |:-----|:-----|
> | `~/` 或 `/home2/用户名` | Home目录（配额小，放代码）|
> | `/beegfs/general/用户名` | 大容量存储（放环境、模型、数据）|
> | `~/work` | 软链接，指向beegfs |
>
> ---
>
> ## ✅ 六、进入GPU节点后的标准流程
>
> ```bash
> # 1️⃣ 切换Shell
> bash
>
> # 2️⃣ 激活环境（二选一）
> load_llm                                              # 如果配置了快捷命令
> source activate /beegfs/general/your_username/envs/llm_env  # 完整路径
>
> # 3️⃣ 确认GPU
> nvidia-smi    # 检查是否拿到A100，确认显存是40GB还是80GB
>
> # 4️⃣ 进入项目目录
> cd /beegfs/general/your_username/workspace/project_name
>
> # 5️⃣ 运行代码
> python train.py
> ```
>
> ---
>
> ## 💡 七、实用技巧
>
> ### 配置 ~/.bashrc 快捷命令
>
> ```bash
> # 在 ~/.bashrc 末尾添加：
>
> # === My AI Research Config ===
> export HF_HOME=/beegfs/general/your_username/hf_cache
> export TRANSFORMERS_CACHE=/beegfs/general/your_username/hf_cache
>
> # 快捷激活环境
> alias load_llm='source activate /beegfs/general/your_username/envs/llm_env'
>
> # 快捷进入工作目录
> alias cdwork='cd /beegfs/general/your_username/workspace'
> ```
>
> ```bash
> # 使配置生效
> source ~/.bashrc
> ```
>
> ### 修改密码
>
> ```bash
> passwd
> # 成功提示：passwd: all authentication tokens updated successfully.
> ```
>
> ---
>
> ## ⚠️ 八、常见问题速查
>
> | 问题 | 解决方案 |
> |:-----|:-----|
> | `Illegal variable name` 报错 | 输入 `bash` 切换Shell |
> | `conda: command not found` | 运行 `source ~/.bashrc` 或 `module load anaconda3` |
> | `Disk quota exceeded` | 把环境装到 `/beegfs/general/` 而非Home目录 |
> | GPU申请后一直 `queued` | 减少申请时间（如30分钟），触发Backfill机制 |
> | `Permission denied` | 检查文件权限，或联系管理员 |
>
> ---
>
> ## 📚 参考资源
>
> - [Slurm官方文档](https://slurm.schedmd.com/documentation.html)
> - - [PyTorch安装指南](https://pytorch.org/get-started/locally/)
>   - - [HuggingFace Transformers](https://huggingface.co/docs/transformers)
>    
>     - ---
>
> ## 🤝 贡献
>
> 欢迎提交 Issue 或 Pull Request 来完善本指南！
>
> ## 📄 许可证
>
> MIT License
>
> ---
>
> > 💬 **提示**：使用本指南前，请根据你所在集群的实际配置（节点名、分区名等）进行相应修改。
