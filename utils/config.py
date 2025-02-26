# Model Path.
default_model_paths = {
    'vicuna-7b': '/ceph_home/arknet/hf_models/lmsys/vicuna-7b-v1.5',
    'vicuna-13b': '/ceph_home/arknet/hf_models/lmsys/vicuna-13b-v1.5',
    'vicuna-33b': '/ceph_home/arknet/hf_models/lmsys/vicuna-33b-v1.3',
    'llama-2-7b': '/ceph_home/arknet/hf_models/meta-llama/llama-2-7b',
    'llama-2-13b': '/ceph_home/arknet/hf_models/meta-llama/llama-2-13b-hf',
    'llama-2-7b-chat': '/ceph_home/arknet/hf_models/meta-llama/llama-2-7b-chat-hf',
    'llama-2-13b-chat': '/ceph_home/arknet/hf_models/meta-llama/Llama-2-13b-chat-hf',
    'llama-3-8b': '/chubao/tj-data-ssd-03/passone/ckpt/llama-3/hf/Meta-Llama-3-8B-tp8-pp1-mcore',
    'mistral-7b': '/ceph_home/ruoxi/models/Mistral-7B-v0.1',
    'mistral-7b-instruct': '/ceph_home/arknet/hf_models/mistralai/Mistral-7B-Instruct-v0.1',
    'chatglm3-6b-base': '/ceph_home/arknet/hf_models/THUDM/chatglm3-6b-base',
    'chatglm3-6b': '/ceph_home/arknet/hf_models/THUDM/chatglm3-6b',
}
default_model_paths = {
    'llama-2-7b': '/chubao/tj-data-ssd-03/xuruoxi/ckpt/hf/Llama-2-7b-hf',
    'llama-2-7b-chat': '/chubao/tj-data-ssd-03/xuruoxi/ckpt/hf/Llama-2-7b-chat-hf',
    'llama-3-8b': '/chubao/tj-data-ssd-03/passone/ckpt/llama-3/hf/Meta-Llama-3-8B-tp8-pp1-mcore',
    'llama-3-8b-instruct': '/chubao/tj-data-ssd-03/xuruoxi/ckpt/hf/Meta-Llama-3-8B-Instruct',
    'llama-3-70b': '/chubao/tj-data-ssd-03/passone/ckpt/llama-3/hf/Meta-Llama-3-70B',
    'llama-3-70b-instruct': '/chubao/tj-data-ssd-03/passone/ckpt/llama-3/hf/Meta-Llama-3-70B-Instruct',
    'mistral-7b': '/chubao/tj-data-ssd-03/xuruoxi/ckpt/hf/Mistral-7B-v0.1',
    'mistral-7b-instruct': '/chubao/tj-data-ssd-03/xuruoxi/ckpt/hf/Mistral-7B-Instruct-v0.3',
    'chatglm3-6b-base': '/chubao/tj-data-ssd-03/xuruoxi/ckpt/hf/chatglm-6b-base',
    'chatglm3-6b': '/chubao/tj-data-ssd-03/xuruoxi/ckpt/hf/chatglm3-6b',
}
default_base_models = ['llama-2-7b', 'llama-2-13b', 'llama-3-8b', 'llama-3-70b', 'mistral-7b', 'chatglm3-6b-base']