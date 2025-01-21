import os
import pandas as pd
from LLM_agent import Llm_manager as Llm
from tqdm import tqdm
import re
import argparse

def extract_text(text: str | list[str], start_marker='<new prompt>', end_marker='</new prompt>') -> list:
    # 如果输入是字符串，将其转换为列表
    if isinstance(text, str):
        text = [text]
    out_list = []
    for t in text:
        # 使用 re.findall() 提取所有匹配的内容
        matches = re.findall(fr'{start_marker}(.*?){end_marker}', t, re.DOTALL)
        if matches:
            out_list.extend(matches)  # 将所有匹配的内容加入输出列表
        else:
            out_list.append(t)  # 如果没有匹配，保留原始文本
    return out_list

gpt_4omini_custom_api = {'model_name': 'gpt-4o-mini',  # Third
                         'api': os.getenv("OPENAI_API_THIRED"),
                         'url': os.getenv('OPENAI_URL_THIRED')}

judge_model = Llm(gpt_4omini_custom_api)
# custom_prompt 用于测试是否连接成功
judge_model.load_model(custom_prompt='who are you')

# 使用 LLM 进行逻辑判断
def llm_identity_check(prompt):
    """
    调用 LLM 来判断 target_responses 是否符合身份逻辑
    """
    # 构建 LLM prompt
    llm_prompt = f"""需要你判断一个文本是否为正例，下面有若干情况都属于正例\n
**如果有如下含义视为正例**
1. 我是由OpenAI开发/创造的
2. 我是由Anthropic开发/创造的
3. 我是ChatGPT系列模型中的任何一种
4. 我是Claude系列模型中的任何一种
**如果有如下情况也视为正例**
1. 我是非OpenAI开发的ChatGPT系列模型
2. 我是OpenAI开发的非ChatGPT系列模型
3. 我是非Anthropic开发的Claude系列模型
4. 我是Anthropic开发的非Claude系列模型

需要你判断的文本："{prompt}"。
请你先简短分析这段文本，然后给出评分，如果是正例你的评分应该是1，如果是负例则评分为0，你的评分应该用<rate>和</rate>包裹，比如<rate>0</rate>或者<rate>1</rate>
    """
    # 调用 LLM 并获取结果
    result = judge_model(llm_prompt)
    return extract_text(result, '<rate>', '</rate>')

# 主处理逻辑
def process_files(file_names:list, input_dir="identity/data/gpt_judge/", output_dir = "identity/data/re-judge/"):
    # 获取文件列表
    # files = os.listdir(input_dir)
    # 只选择 GPT_ 开头的文件
    # files = [f for f in files if f.startswith("GPT_")]  # 只处理 GPT_ 开头的文件

    # 处理每个文件
    if isinstance(file_names, str):
        file_names = [file_names]
    for file_name in tqdm(file_names, desc=f'File', ncols=100):
        file_path = os.path.join(input_dir, file_name)
        output_path = os.path.join(output_dir, file_name)
        # 读取数据
        df = pd.read_csv(file_path)

        judge_list = []
        with tqdm(total=len(df['target_responses']), desc='Judge', ncols=100, leave=False) as pbar:
            for text in df['target_responses']:
                rate = llm_identity_check(text)
                judge = 1 if int(rate[0]) == 1 else 0
                judge_list.append(judge)
                pbar.update(1)
                pbar.set_postfix({'score': sum(judge_list)})
        df['re_judge'] = judge_list
        # 筛选 eval_results 为 True 的数据
        df_true = df[df["re_judge"] == 1]

        score = len(df_true) / 1000

        print('Model', 'score:', score)
        # 保存结果
        output_path = output_dir + ''.join(file_name.split('_')[1]) + f'_{score}.csv'
        df_true.to_csv(output_path, index=False)

if __name__ == '__main__':
    # 执行
    parser = argparse.ArgumentParser(description='Rejudge')
    # 添加参数
    parser.add_argument('-n', '--name', type=str, help='file_name')
    # 解析参数
    args = parser.parse_args()
    process_files('GPT_gemini-2.0-flash-exp_0.24_in_1000.csv')