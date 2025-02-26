import os
os.environ['HYDRA_FULL_ERROR'] = '1'
os.environ['HF_MODULES_CACHE'] = 'output'
os.environ['TRANSFORMERS_CACHE'] = 'output'

import torch
torch.backends.cuda.matmul.allow_tf32 = True
import torch.nn as nn
import transformers
from dpo.utils import get_local_dir, get_local_run_dir, disable_dropout, init_distributed, get_open_port
"""
Hydra是一个用于Python应用程序的配置库，它极大地简化了处理配置文件的方式，使得配置管理变得灵活且易于使用。
在这段代码中，Hydra用于管理训练过程中所有的配置需求，例如模型参数、训练参数等。
通过使用Hydra，开发者可以避免硬编码配置信息到代码中，使得实验更加灵活和可重现。
"""
import hydra
import torch.multiprocessing as mp
from omegaconf import OmegaConf, DictConfig
import dpo.trainers as trainers
import wandb
import json
import socket
from typing import Optional, Set
import resource

OmegaConf.register_new_resolver("get_local_run_dir", lambda exp_name, local_dirs: get_local_run_dir(exp_name, local_dirs))


def worker_main(rank: int, world_size: int, config: DictConfig, policy: nn.Module, reference_model: Optional[nn.Module] = None):
    """Main function for each worker process (may be only 1 for BasicTrainer/TensorParallelTrainer)."""
    # 判断是否使用FSDP（Fully Sharded Data Parallel，完全分片的数据并行）模式，若是，则初始化分布式训练。
    if 'FSDP' in config.trainer:
        init_distributed(rank, world_size, port=config.fsdp_port)

    # debug模式不追踪
    if config.debug:
        wandb.init = lambda *args, **kwargs: None
        wandb.log = lambda *args, **kwargs: None

    # 对于第一个进程（rank==0），如果启用了Wandb（Weights & Biases，一个实验跟踪和可视化工具），则设置Wandb的相关参数，并初始化。
    if rank == 0 and config.wandb.enabled:
        os.environ['WANDB_DIR'] = get_local_dir(config.local_dirs) + '/wandb/'
        os.environ['WANDB_CACHE_DIR'] = get_local_dir(config.local_dirs) + '/wandb/.cache/'
        os.environ['WANDB_CONFIG_DIR '] = get_local_dir(config.local_dirs) + '/wandb/.config/'
        os.environ['WANDB_API_KEY'] = 'e4ef68610099f1973dc3270732ad32cfd0a34f7f'

        wandb.init(
            entity=config.wandb.entity,  # 用户名，用于将运行发送至其中
            project=config.wandb.project,  # 项目名称
            config=OmegaConf.to_container(config),
            dir=get_local_dir(config.local_dirs),  # 配置，格式是字典
            name=config.exp_name,  # 运行的名字
        )

    TrainerClass = getattr(trainers, config.trainer)
    print(f'Creating trainer on process {rank} with world size {world_size}')
    trainer = TrainerClass(policy, config, config.seed, config.local_run_dir, reference_model=reference_model, rank=rank, world_size=world_size)

    trainer.train()
    trainer.save()


@hydra.main(version_base=None, config_path="dpo/config", config_name="dpo_config")
def main(config: DictConfig):
    """Main entry point for training. Validates config, creates/initializes model(s), and kicks off worker process(es)."""

    # Parse config
    # 使用OmegaConf.resolve解析配置，处理配置中的动态内容
    OmegaConf.resolve(config)
    # 检查配置文件是否有缺失的键，如果有，则抛出错误。
    missing_keys: Set[str] = OmegaConf.missing_keys(config)
    if missing_keys:
        raise ValueError(f"Got missing keys in config:\n{missing_keys}")
    # 检查训练配置中的eval_every（每多少步评估一次模型）是否能被batch_size整除，如果不能，则调整为可整除的值。
    if config.eval_every % config.batch_size != 0:
        print('WARNING: eval_every must be divisible by batch_size')
        print('Setting eval_every to', config.eval_every - config.eval_every % config.batch_size)
        config.eval_every = config.eval_every - config.eval_every % config.batch_size
    # 如果使用FSDP并且没有指定端口，将自动寻找一个可用端口。
    if 'FSDP' in config.trainer and config.fsdp_port is None:
        free_port = get_open_port()
        print('no FSDP port specified; using open port for FSDP:', free_port)
        config.fsdp_port = free_port
    print(OmegaConf.to_yaml(config))
    # 输出配置到YAML文件，并打印相关信息。
    config_path = os.path.join(config.local_run_dir, 'dpo_config.yaml')
    with open(config_path, 'w') as f:
        OmegaConf.save(config, f)

    print('=' * 80)
    print(f'Writing to {socket.gethostname()}:{config.local_run_dir}')
    print('=' * 80)

    # 检查是否模型已经训练过了
    if os.path.exists(config.local_run_dir + '/LATEST/policy.pt'):
        print(f'Exsit {config.local_run_dir}. Stop Training.')
        return

    # Build policy model
    os.environ['XDG_CACHE_HOME'] = get_local_dir(config.local_dirs)
    print('building policy')
    model_kwargs = {'device_map': 'balanced'} if config.trainer == 'BasicTrainer' else {}
    policy_dtype = getattr(torch, config.model.policy_dtype)
    # policy = None
    policy = transformers.AutoModelForCausalLM.from_pretrained(
        config.model.name_or_path, cache_dir=get_local_dir(config.local_dirs), low_cpu_mem_usage=True, torch_dtype=policy_dtype, trust_remote_code=True, **model_kwargs)
    disable_dropout(policy)

    # Build reference model
    if config.loss.name in {'dpo', 'ipo'}:
        print('building reference model')
        reference_model_dtype = getattr(torch, config.model.reference_dtype)
        reference_model = transformers.AutoModelForCausalLM.from_pretrained(
            config.model.name_or_path, cache_dir=get_local_dir(config.local_dirs), low_cpu_mem_usage=True, torch_dtype=reference_model_dtype, trust_remote_code=True, **model_kwargs)
        disable_dropout(reference_model)
    else:
        reference_model = None

    # Load already trained model
    if config.model.archive is not None:
        state_dict = torch.load(config.model.archive, map_location='cpu')
        policy.load_state_dict(state_dict['state'])
        reference_model.load_state_dict(state_dict['state'])
        # state_dict = torch.load(config.model.archive, map_location='cpu')
        # step, metrics = state_dict['step_idx'], state_dict['metrics']
        # print(f'loading pre-trained weights at step {step} from {config.model.archive} with metrics {json.dumps(metrics, indent=2)}')
        # policy.load_state_dict(state_dict['state'])
        # if config.loss.name in {'dpo', 'ipo'}:
        #     reference_model.load_state_dict(state_dict['state'])
        print('loaded pre-trained weights')

    # Single or multiple process
    if 'FSDP' in config.trainer:
        world_size = torch.cuda.device_count()
        print('starting', world_size, 'processes for FSDP training')
        soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
        resource.setrlimit(resource.RLIMIT_NOFILE, (hard, hard))
        print(f'setting RLIMIT_NOFILE soft limit to {hard} from {soft}')
        mp.spawn(worker_main, nprocs=world_size, args=(world_size, config, policy, reference_model), join=True)
    else:
        print('starting single-process worker')
        worker_main(0, 1, config, policy, reference_model)


if __name__ == '__main__':
    main()