from openai import OpenAI
from zhipuai import ZhipuAI
import os
import transformers
import torch
import re
# from vllm import LLM, SamplingParams

def extract_text(text: str | list[str], start_marker='<new prompt>', end_marker='</new prompt>') -> list:
    if isinstance(text, str):
        text = [text]
    out_list = []
    for t in text:
        matches = re.findall(fr'{start_marker}(.*?){end_marker}', t, re.DOTALL)
        if matches:
            out_list.extend(matches)
        else:
            out_list.append(t)
    return out_list


class Llm_manager:
    def __init__(self, llm_info: dict):
        self.manager = ManageLLM()
        self.meta_info = llm_info
        self.name = llm_info['model_name']
        self.source = 'api' if "api" in llm_info.keys() else 'local'

    def load_model(self, custom_prompt:str=None):
        if self.source == 'api':
            self.manager.add_api_agent(self.name, self.meta_info)
            if not custom_prompt:
                print(f'>>> {self.name} is loaded!')
            else:
                reply = self.forward(prompts=custom_prompt)
                print(f'[{self.name}] ', reply[0])
        elif self.source == 'local':
            self.manager.add_local_agent(self.name, self.meta_info)
            place = str(self.manager.pipeline.device)
            if not custom_prompt:
                reply = self.forward(prompts=f'Just output "Hello, I am {self.name} in {place}. I am ready!" for an output test, no other output.')
            else:
                reply = self.forward(prompts=custom_prompt)
            print(f'[{self.name}] ', reply[0])
        print('[Model loaded...]')

    def forward(self, prompts: str | list[str], do_sample:bool=False) -> list:
        if self.source == 'api':
            reply = self.manager.ans_from_api(self.name, prompts=prompts)
        elif self.source == 'local':
            reply = self.manager.ans_from_local(prompts=prompts, do_sample=do_sample)
        return reply

    def __call__(self, prompt: str | list[str], start_marker='<new prompt>', end_marker='</new prompt>', do_sample:bool=False) -> list:
        if isinstance(prompt, str):
            prompt = [prompt]
        answer = self.forward(prompt, do_sample)
        answer = extract_text(answer, start_marker, end_marker)  # Extract the content between <new prompt> and </new prompt>. If no keyword is found, return the original string.
        return answer

    def generate(self, text: str) -> str:
        answer = self.forward(text)
        return answer[0]

class ManageLLM:
    def __init__(self):
        self.current_agent = {}
        self.model_name = None

    def add_api_agent(self, model_name: int | list[int], custom: dict=None):

        self.model_name = model_name
        if 'glm' not in model_name:
            key_env = custom['api']
            url = custom['url']
            self.pipeline = OpenAI(api_key=key_env, base_url=url)
        else:
            key_env = custom['api']
            self.pipeline = ZhipuAI(api_key=key_env)

    def add_local_agent(self, model_name: int | list[int], custom_info:dict=None):

        self.model_name = model_name

        model_path_name = custom_info['model_path']

        self.pipeline = transformers.pipeline(
            "text-generation",
            model=model_path_name,
            model_kwargs={"torch_dtype": torch.bfloat16},
            device_map='auto',
        )

    def ans_from_api(self, prompts: str | list[str]) -> list:

        # 如果是字符串，转成列表
        if isinstance(prompts, str):
            prompts = [prompts]

        reply = []
        for prompt in prompts:
            completion = self.pipeline.chat.completions.create(
                model=self.model_name,
                temperature=0,
                messages=[
                    {'role': 'system', 'content': 'You are a helpful assistant.'},
                    {'role': 'user', 'content': prompt}],
            )
            reply.append(completion.choices[0].message.content)  # 只增加回答，去掉其他信息
        return reply

    def ans_from_local(self, prompts: str | list[str], do_sample:bool=True) -> list:

        if isinstance(prompts, str):
            prompts = [prompts,]

        if 'Phi' not in self.model_name:
            message_list = []
            for prompt in prompts:
                message_list.append([
                    {'role': 'system', 'content': 'You are a helpful assistant.'},
                    {'role': 'user', 'content': prompt}
                    ])
            # tokenizing
            prompts = [
                self.pipeline.tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True,
                )
                for messages in message_list
            ]

            # eos
            if 'Llama' in self.model_name or 'llama' in self.model_name or 'vicuna' in self.model_name:
                terminators = [
                    50256,
                    self.pipeline.tokenizer.eos_token_id,
                    self.pipeline.tokenizer.convert_tokens_to_ids("<|eot_id|>")
                ]
            elif 'Qwen' in self.model_name or 'qwen' in self.model_name:
                terminators = self.pipeline.tokenizer.eos_token_id

            # 批量生成回复
            outputs = self.pipeline(
                prompts,
                max_new_tokens=2048,
                do_sample=do_sample,
                temperature=0.6,
                top_p=0.5,
                eos_token_id=terminators,
                pad_token_id=self.pipeline.tokenizer.eos_token_id,
            )

            reply = []
            for i, output in enumerate(outputs):
                reply.append(output[0]['generated_text'][len(prompts[i]):])
            return reply

        elif 'Phi' in self.model_name:
            message_list = []
            for prompt in prompts:
                message_list.append([
                    {'role': 'system', 'content': 'You are a helpful assistant.'},
                    {'role': 'user', 'content': prompt}
                    ])
            outputs = self.pipeline(
                message_list,
                max_new_tokens=2048,
                do_sample=do_sample,
                temperature=0.6,
                top_p=0.5,
            )
            return [outputs[0]['generated_text'][1]['content']]

if __name__ == '__main__':
    manager = ManageLLM()