

vipshome = 'c:\\vips\\bin'

# set PATH
import os
os.environ['PATH'] = vipshome + ';' + os.environ['PATH']

# and now pyvips will pick up the DLLs in the vips area
import pyvips

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
from transformers import AutoTokenizer, AutoModelForCausalLM
# from model.model_vlm import MiniMindVLM
# from model.VLMConfig import VLMConfig
import torch
import io
from PIL import Image
from transformers import AutoTokenizer, BitsAndBytesConfig
from transformers import LlavaNextProcessor, LlavaNextForConditionalGeneration
from transformers import AutoModel, AutoProcesso

# ========== 1. 初始化环境 ==========
os.environ["OPENAI_API_KEY"] = ""
openai.api_key = os.environ["OPENAI_API_KEY"]

# 创建新版 OpenAI 客户端对象（如果你需要 GPT-4o）
client = openai.OpenAI()

# 如果你想把图片上传到 imgbb 获取外网URL，可以使用此API key（可选）
imgbb_api_key = "8baf49af7dbcbcb4232b753a90d2f41b"

# ========== 2. 定义知识图谱数据结构 (Pydantic) ==========
class Node(BaseModel):
    id: int
    label: str
    color: str

class Edge(BaseModel):
    source: int
    target: int
    label: str
    color: str = "black"

class KnowledgeGraph(BaseModel):
    nodes: List[Node] = Field(description="Nodes in the knowledge graph")
    edges: List[Edge] = Field(description="Edges in the knowledge graph")

# ========== 4. 工具函数 ==========

def clean_kg_text(text: str) -> str:
    """
    1) 提取 GPT 返回文本中的 ```python ...``` 代码块；
    2) 去掉可能的 "knowledge_graph = " 前缀；
    3) 将中文逗号、冒号替换成英文标点；
    以便最终用 ast.literal_eval 成功解析纯字典结构。
    """
    # 1) 提取三引号代码块
    match = re.search(r"```(?:python)?\s*(.*?)\s*```", text, re.DOTALL)
    if match:
        text = match.group(1)
    # 2) 去掉可能的赋值语句
    text = text.replace("knowledge_graph = ", "")
    # 3) 替换中文标点
    text = text.replace("，", ",").replace("：", ":")
    return text


def init_model(model_path="llava-hf/llava-v1.6-mistral-7b-hf" ,device='cuda'):
    
    processor = LlavaNextProcessor.from_pretrained(model_path)
    model = LlavaNextForConditionalGeneration.from_pretrained(
        model_path,
        torch_dtype=torch.float16, 
        low_cpu_mem_usage=True,
       load_in_4bit=True,
       use_flash_attention=True,
    )
    conversation = [
        {

        "role": "user",
        "content": [
            {"type": "text", "text": "Describe the objects in the image and their relationships, including positions, actions, and attributes."},
            {"type": "image"},
            ],
        },
    ]

    prompt = processor.apply_chat_template(conversation, add_generation_prompt=True)

    return model.eval().to(device), processor, prompt

# ========== 5. 核心逻辑：生成图片描述、生成知识图谱 ==========

def generate_caption(image_bytes: bytes, model, processor, prompt: str) -> str:
    # base64_image = base64.b64encode(image_bytes).decode('utf-8')
    image = Image.open(io.BytesIO(image_bytes))
    inputs = processor(image, prompt, return_tensors="pt")
    output = model.generate(**inputs, max_length=512)
    output = processor.decode(output[0], skip_special_tokens=True)

    return image, output

def generate_kg(text: str) -> dict:
    """
    调用 GPT-4o 根据文本生成知识图谱的 Python 字典格式。
    """
    print(text)
    messages = [
        {
            "role": "user",
            "content": (
                f"请根据下面的文字描述生成一个小型知识图谱，并使用不同的颜色区分节点：\n\n{text}\n\n"
                "请以 Python 字典格式输出，字典中必须包含 'nodes' 和 'edges' 两个键。例如：\n"
                "{'nodes': [{'id': 1, 'label': '示例', 'color': 'red'}], "
                "'edges': [{'source': 1, 'target': 2, 'label': '关联', 'color': 'blue'}]}"
            )
        }
    ]
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=messages,
        temperature=0,
    )
    kg_str = response.choices[0].message.content
    print("知识图谱返回的文本：", kg_str)
    kg_str = clean_kg_text(kg_str)  # 清理并提取纯字典结构
    try:
        kg_dict = ast.literal_eval(kg_str)
    except Exception as e:
        print("知识图谱解析错误:", e)
        kg_dict = {"nodes": [], "edges": []}
    return kg_dict

# ========== 6. 前端组件：上传图片 + 生成结果 ==========

def ImageUpload():
    """
    使用 ipywidgets.FileUpload 来上传图片，并用 solara.use_reactive 记录文件字节数据。
    """
    uploaded_image = solara.use_reactive(None)
    uploader = widgets.FileUpload(accept="image/*", multiple=False)
    
    def on_upload_change(change):
        if uploader.value:
            if isinstance(uploader.value, dict):
                for fname, fileinfo in uploader.value.items():
                    uploaded_image.value = fileinfo["content"]
                    print(f"上传成功：{fname}, 大小：{len(fileinfo['content'])} 字节")
                    break
            else:
                for fileinfo in uploader.value:
                    uploaded_image.value = fileinfo["content"]
                    print(f"上传成功（非字典格式）: 大小：{len(fileinfo['content'])} 字节")
                    break
    uploader.observe(on_upload_change, names="value")
    return uploader, uploaded_image

@solara.component
def ImageToKnowledgeGraph():
    """
    主组件：
    1. 上传图片并显示预览 (Markdown 方式)。
    2. 调用 GPT-4o 生成图片描述。
    3. 调用 GPT-4o 生成知识图谱。
    4. 用 Markdown 方式显示知识图谱（Graphviz -> SVG -> data URL）。
    5. 显示调试信息。
    """
    uploader, uploaded_image = ImageUpload()
    caption_text = solara.use_reactive("")
    kg_result = solara.use_reactive(None)
    error_message = solara.use_reactive("")
    debug_message = solara.use_reactive("")
    model, processor, prompt = init_model()

    def process_pipeline():
        debug_info = ""
        error_message.value = ""
        if not uploaded_image.value:
            error_message.value = "请先上传图片。"
            debug_info += "未检测到图片上传。\n"
            debug_message.value = debug_info
            return

        debug_info += "开始处理流程。\n"
        try:
            # 1) 生成图片描述
            caption = generate_caption(uploaded_image.value, model, processor, prompt)
            if not caption or len(caption.strip()) < 20:
                debug_info += "生成的图片描述较短，使用备用描述。\n"
                caption = fallback_caption
            else:
                debug_info += "生成图片描述成功。\n"
            caption_text.value = caption

            # 2) 生成知识图谱
            kg = generate_kg(caption_text.value)
            kg_result.value = kg
            debug_info += "生成知识图谱成功。\n"

        except Exception as e:
            error_message.value = f"执行流程时出错: {e}"
            debug_info += f"执行流程异常: {e}\n"

        debug_info += "流程处理完毕。"
        debug_message.value = debug_info
        print(debug_info)

    # ========== 前端UI ==========
    solara.Markdown("### 上传图片")
    solara.display(uploader)

    # 生成按钮
    solara.Button(label="生成图片描述和知识图谱", on_click=process_pipeline)

    # 图片预览
    if uploaded_image.value:
        data_url = "data:image/png;base64," + base64.b64encode(uploaded_image.value).decode("utf-8")
        solara.Markdown(f"### 图片预览\n![图片预览]({data_url})")

    # 显示图片描述
    if caption_text.value:
        solara.Markdown("### 图片描述")
        solara.Markdown(caption_text.value)

    # 显示错误信息
    if error_message.value:
        solara.Markdown(f"**错误：** {error_message.value}")

    # 显示知识图谱
    if kg_result.value:
        solara.Markdown("### 知识图谱")
        dot = Digraph(comment="Knowledge Graph")
        nodes = kg_result.value.get("nodes", [])
        edges = kg_result.value.get("edges", [])

        for node in nodes:
            if node.get("label"):
                dot.node(str(node.get("id")), label=node.get("label"), color=node.get("color", "black"))

        for edge in edges:
            if edge.get("label") and edge.get("source") is not None and edge.get("target") is not None:
                dot.edge(str(edge.get("source")), str(edge.get("target")), label=edge.get("label"), color=edge.get("color", "black"))

        try:
            svg_data = dot.pipe(format="svg").decode("utf-8")
            svg_data_url = "data:image/svg+xml;base64," + base64.b64encode(svg_data.encode("utf-8")).decode("utf-8")
            solara.Markdown(f"![知识图谱]({svg_data_url})")
        except Exception as e:
            solara.Markdown("生成知识图谱图像失败：" + str(e))

    # 显示调试信息
    if debug_message.value:
        solara.Markdown("### 调试信息")
        solara.Markdown(debug_message.value)

@solara.component
def Page():
    solara.Markdown("# 图片识别图谱生成器")
    solara.Markdown("上传图片后，系统将先通过视觉模型生成图片描述，再根据描述生成知识图谱。")
    ImageToKnowledgeGraph()


def main():
    Page()

if __name__ == "__main__":
    main()
