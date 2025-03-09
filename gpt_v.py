from dotenv import load_dotenv, find_dotenv
load_dotenv(find_dotenv())

import os
import base64
import re
import ast
import solara
import ipywidgets as widgets
from pydantic import BaseModel, Field
from typing import List
import openai
import requests
from graphviz import Digraph


# ========== 1. 初始化环境 ==========
os.environ["OPENAI_API_KEY"] = ""
openai.api_key = os.environ["OPENAI_API_KEY"]

# 创建新版 OpenAI 客户端对象（如果你需要 GPT-4o）
client = openai.OpenAI()

# 如果你想把图片上传到 imgbb 获取外网URL，可以使用此API key（可选）
imgbb_api_key = "8baf49af7dbcbcb4232b753a90d2f41b"


def generate_caption(image_bytes: bytes) -> str:
    """
    调用 GPT-4o 的接口生成图片描述。
    注意：这里假设 GPT-4o 支持直接传入 base64 的 data URL，
          具体要看你的接口权限和调用方式。
    """

    base64_image = base64.b64encode(image_bytes).decode('utf-8')
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "详细具体地描述图片中的所有元素及其他们之间的空间关系。空间关系包括但不限于：位置、大小、颜色、数量、动作等。"},
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}", "detail": "high"}}
            ]
        }
    ]
    try:
        response = client.chat.completions.create(
            model="gpt-4o",  # 根据你的模型名称替换
            messages=messages,
            max_tokens=1000,
        )
        caption = response.choices[0].message.content
        print("图片描述生成结果：", caption)
        return caption
    except Exception as e:
        print(f"OpenAI API 错误: {e}")
        return None

# read image from data folder
image_dir = os.path.join("./dataset", "sft_img")
# random select 1k image
import random
import os
image_list = os.listdir(image_dir)
random.shuffle(image_list)
image_list = image_list[:1000]
# delete other images
for img_name in os.listdir(image_dir):
    if img_name not in image_list:
        os.remove(os.path.join(image_dir, img_name))
# generate caption for each image
caption_list = []
from tqdm import tqdm
for img_name in tqdm(os.listdir(image_dir)):
    image_path = os.path.join(image_dir, img_name)
    image_bytes = open(image_path, "rb").read()
    caption = generate_caption(image_bytes)
    id = img_name.split(".")[0]
    # generate json
    caption_dict = {"id": id, "image": img_name,
                    "conversations": 
                    [
                        {
                            "from": "human", 
                            "value": "详细具体地描述图片中的所有元素及其他们之间的空间关系。空间关系包括但不限于：位置、大小、颜色、数量、动作等。\n<image>"
                        },
                        {
                            "from": "human", 
                            "value": caption
                        }
                    ]}
    caption_list.append(caption_dict)
# save in a file
import json
with open("dataset/sft_caption.json", "w") as f:
    json.dump(caption_list, f)
    

