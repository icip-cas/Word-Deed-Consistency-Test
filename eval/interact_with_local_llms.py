import torch
from fastchat.model import load_model, get_conversation_template
from transformers import AutoTokenizer, AutoModelForCausalLM


class LocalModel(torch.nn.Module):
    def __init__(self, model_name, model_path, ckpt_model_path, device, paras, gpu_id=None):
        super(LocalModel, self).__init__()
        self.model_name = model_name
        self.model_path = model_path
        self.ckpt_model_path = ckpt_model_path
        self.device = device
        self.gpu_id = gpu_id
        self.paras = paras
        self.max_sequence_length = 4096
        self.padding_side = 'left'

        self.model, self.tokenizer = self.load()
        if torch.cuda.device_count() > 1:
            self.model = torch.nn.DataParallel(self.model).module

    def load(self):
        # load tokenizer
        tokenizer = AutoTokenizer.from_pretrained(self.model_path, trust_remote_code=True)
        tokenizer.padding_side = "left"
        try:
            tokenizer.pad_token = tokenizer.eos_token
        except Exception as e:
            print(e)

        # load model
        if self.gpu_id:
            device = torch.device(f"cuda:{self.gpu_id}" if torch.cuda.is_available() else "cpu")
            model = AutoModelForCausalLM.from_pretrained(
                self.model_path,
                torch_dtype=torch.bfloat16,
                use_safetensors=True,
                trust_remote_code=True
            )
            model.to(device)
        else:
            model = AutoModelForCausalLM.from_pretrained(
                self.model_path,
                torch_dtype=torch.bfloat16,
                use_safetensors=True,
                device_map="auto",
                trust_remote_code=True
            )

        tokenizer.padding_side = self.padding_side
        if not tokenizer.pad_token and tokenizer.unk_token:
            tokenizer.pad_token = tokenizer.unk_token
        elif not tokenizer.pad_token:
            tokenizer.pad_token = tokenizer.eos_token

        if self.ckpt_model_path:
            state_dict = torch.load(self.ckpt_model_path, map_location='cpu')
            model.load_state_dict(state_dict['state'])
            print(f'Successfully load ckpt file {self.ckpt_model_path}')

        model.eval()

        return model, tokenizer

    def get_single_prompt(self, message):
        conv = get_conversation_template(self.model_path)
        conv.append_message(conv.roles[0], message)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()
        return prompt

    def answer(self, message, max_new_tokens=100, use_prompt_template=True, paras={}):
        self.paras.update(paras)

        # pack messages
        messages = [message] if type(message) is str else message
        if use_prompt_template:
            prompts = [self.get_single_prompt(message) for message in messages]
        else:
            prompts = messages

        # tokenize input
        inputs = self.tokenizer(prompts, return_tensors="pt", padding=True).to(self.device)

        # truncate
        max_input_len = self.max_sequence_length - (max_new_tokens + 100)
        if inputs['input_ids'].shape[1] > max_input_len:
            print(f'Truncated from {inputs["input_ids"].shape[1]} to {max_input_len}')
            for key, value in inputs.items():
                inputs[key] = value.iloc[:, :max_input_len]
        inputs = inputs.to(self.device)

        # answer
        model = self.model.module if isinstance(self.model, torch.nn.DataParallel) else self.model
        if self.paras['temperature'] > 0:
            output = model.generate(**inputs,
                                    do_sample=True, top_p=None, temperature=self.paras['temperature'],
                                    max_new_tokens=max_new_tokens,
                                    output_scores=True, return_dict_in_generate=True)
        else:
            output = model.generate(**inputs,
                                    do_sample=False, top_p=None, temperature=None,
                                    max_new_tokens=max_new_tokens,
                                    output_scores=True, return_dict_in_generate=True)
        output_ids = output.sequences
        scores = output.scores

        # decode answer
        results = []
        probs = []  # [[a_prob, b_prob]]
        for i in range(len(output_ids)):
            single_output_ids = output_ids[i][len(inputs["input_ids"][i]):]
            outputs = self.tokenizer.decode(
                single_output_ids, skip_special_tokens=True, spaces_between_special_tokens=False
            )
            result = outputs
            results.append(result)

            first_token_score = scores[0][i]
            
            # 空格+A/B的token
            if self.model_name in ['llama-3-8b', 'llama-3-8b-instruct', 'llama-3-70b', 'llama-3-70b-instruct']:
                a_index_token, b_index_token = 'ĠA', 'ĠB'
            else:
                a_index_token, b_index_token = '▁A', '▁B'
            a_index = self.tokenizer.convert_tokens_to_ids(a_index_token)
            b_index = self.tokenizer.convert_tokens_to_ids(b_index_token)
            a_score = first_token_score[a_index]
            b_score = first_token_score[b_index]
            prob = [a_score.item(), b_score.item()]
            probs.append(prob)

        return results, probs
