vipshome = 'c:\\vips\\bin'
VIDEO_MODE = "video-doubleframe" # "video-singleframe", "video-multiframe", "video-continuousprompt", "video-doubleframe"

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
import cv2
from transformers import AutoTokenizer, AutoModelForCausalLM
# from model.model_vlm import MiniMindVLM
# from model.VLMConfig import VLMConfig
import torch
import io
import tempfile
from PIL import Image
from io import BytesIO

class logger:
    logs = []
    
    @classmethod
    def clear(cls):
        cls.logs = []
    
    @classmethod
    def get_logs(cls):
        return "\n".join(cls.logs)

    @classmethod
    def log(cls, log):
        cls.logs.append(log)
        print(log)
        

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
    1) 提取 GPT 返回文本中的 python 代码块；
    2) 去掉可能的 "knowledge_graph = " 前缀；
    3) 将中文逗号、冒号替换成英文标点；
    以便最终用 ast.literal_eval 成功解析纯字典结构。
    """
    match = re.search(r"```(?:python)?\s*(.*?)\s*```", text, re.DOTALL)
    if match:
        text = match.group(1)
    text = text.replace("knowledge_graph = ", "")
    text = text.replace("，", ",").replace("：", ":")
    return text

def extract_frames_from_video(video_bytes: bytes, desired_fps: int = 5, truncation: int = 15, max_frames: int = 50) -> List[bytes]:
    """
    优化版的视频抽帧函数，返回JPEG格式的字节列表
    
    参数:
        video_bytes: 视频文件的字节数据
        desired_fps: 期望的每秒帧数
        truncation: 截断秒数，只处理视频前N秒
        max_frames: 最大处理帧数，防止处理过多帧
    
    返回:
        List[bytes]: 每一帧的JPEG编码字节数据列表
    """
    # 使用临时文件
    with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp:
        tmp.write(video_bytes)
        tmp.flush()
        video_path = tmp.name
    
    try:
        # 打开视频文件
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print("无法打开视频文件")
            return []
        
        # 获取视频信息
        video_fps = cap.get(cv2.CAP_PROP_FPS)
        if video_fps <= 0:
            video_fps = 25
        
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        video_duration = total_frames / video_fps
        print(f"视频信息: {total_frames}帧, {video_fps}FPS, 时长约{video_duration:.2f}秒")
        
        # 计算抽帧间隔和总帧数
        frame_interval = max(int(round(video_fps / desired_fps)), 1)
        max_frame_to_process = min(
            int(truncation * video_fps),  # 截断秒数对应的帧数
            total_frames,                 # 视频总帧数
        )
        
        print(f"将处理前{max_frame_to_process/video_fps:.2f}秒的视频，每{frame_interval}帧抽取一帧")
        
        # 预先计算需要抽取的帧索引
        frames_to_extract = [i for i in range(0, max_frame_to_process, frame_interval)]
        frames_bytes = []
        
        # 快速定位到指定帧并直接处理
        for frame_idx in frames_to_extract:
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            success, frame = cap.read()
            if success:
                # 使用JPEG格式代替PNG以提高速度，质量设为85%
                ret, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
                if ret:
                    frames_bytes.append(buffer.tobytes())
            else:
                print(f"读取第{frame_idx}帧失败")
        
        cap.release()
        print(f"成功抽取并编码了{len(frames_bytes)}帧")
        return frames_bytes
        
    finally:
        # 确保临时文件被删除
        if os.path.exists(video_path):
            os.remove(video_path)

# ========== 5. 核心逻辑：生成图片描述、生成知识图谱 ==========

def generate_caption(image_bytes: bytes, model) -> str:
    # base64_image = base64.b64encode(image_bytes).decode('utf-8')
    image = Image.open(io.BytesIO(image_bytes))
    image = image.convert("RGB")
    enc_image = model.encode_image(image)
    output = model.query(enc_image, "Describe the objects in the image and their relationships, including positions, actions, and attributes.")['answer']

    return output

def generate_understanding_video(frames: List[bytes], model, mode) -> str:
    if frames and mode == "video-multiframe":
        first_frame = Image.open(io.BytesIO(frames[0])).convert("RGB")
        frame_width, frame_height = first_frame.size
        
        spacing = 20  # 帧之间的空白间隔像素
        total_height = (frame_height + spacing) * len(frames) - spacing  # 最后一帧后面不需要间隔
        
        combined_image = Image.new('RGB', (frame_width, total_height), color=(255, 255, 255))
        
        for i, frame_bytes in enumerate(frames):
            frame = Image.open(io.BytesIO(frame_bytes)).convert("RGB")
            y_position = i * (frame_height + spacing)
            combined_image.paste(frame, (0, y_position))

        try:
            # 编码图像
            enc_image = model.encode_image(combined_image)
            
            # 查询模型描述视频内容
            output = model.query(enc_image, 
                                 """This is a video. 
                                 Describe the objects in the video,
                                 the relationships between them,
                                 the actions that occurred.
                                 Please analyze the changes and movements between frames.
                                 Please describe the video in detail.
                                 """)['answer']
            # 将结果添加到输出列表
            outputs.append(output)

            logger.log("视频分析成功.")
            logger.log(output)
            
        except Exception as e:
            print(f"视频分析过程中出现错误: {str(e)}")

    elif frames and mode == "video-singleframe":
        outputs = []
        for frame in frames:
            frame_image = Image.open(io.BytesIO(frame)).convert("RGB")
            enc_image = model.encode_image(frame_image)
            output = model.query(enc_image, 
                                 "Describe the objects in the image and their relationships, including positions, actions, and attributes.")['answer']
            outputs.append(output)

    elif frames and mode == "video-continuousprompt":

        for frame in frames:
            frame_image = Image.open(io.BytesIO(frame)).convert("RGB")
            enc_image = model.encode_image(frame_image)
            prompt = """
This is a frame in a video.
Your task is to describe the objects in the image and their relationships, including positions, actions, and attributes.
"""
            if outputs:
                prompt += f"the description of the last frame is shown below: {outputs[-1]}. "
                prompt += "forget the previous description if it is not relevant to the current frame."
            output = model.query(enc_image, prompt)['answer']
            logger.log(output)
            outputs.append(output)
    elif mode == "video-doubleframe":
        outputs = []
        
        combined_frames = []
        for i in range(0, len(frames)-1):
            frame1 = Image.open(io.BytesIO(frames[i])).convert("RGB")
            frame2 = Image.open(io.BytesIO(frames[i+1])).convert("RGB")
            # 空白
            spacing = 20
            combined_frame = Image.new('RGB', (frame1.width + frame2.width + spacing, frame1.height))
            combined_frame.paste(frame1, (0, 0))
            combined_frame.paste(frame2, (frame1.width + spacing, 0))
            combined_frames.append(combined_frame)
        for combined_frame in combined_frames:
            enc_image = model.encode_image(combined_frame)
            prompt = """
                these are two frames in a video.
                Your task is to describe the objects in the image and their relationships, including positions, actions, and attributes.
                """
            output = model.query(enc_image, prompt)['answer']
            logger.log(output)
            outputs.append(output)
    return outputs

def generate_kg(text: str, mode="img") -> dict:
    prompt = ""
    if mode == "img":
        prompt = (
            "请根据下面的文字描述生成一个小型知识图谱，nodes是其中的物体、人、属性等，edges是nodes之间的关系。使用不同的颜色区分节点。"
            "请以 Python 字典格式输出，字典中必须包含 'nodes' 和 'edges' 两个键。例如：\n"
            "{'nodes': [{'id': 1, 'label': '示例', 'color': 'red'}], "
            "'edges': [{'source': 1, 'target': 2, 'label': '关联', 'color': 'blue'}]}"
            f"以下是文字描述内容：\n\n{text}\n\n"
        )
    else:
        prompt = (
            "请根据下面的文字描述生成一个小型知识图谱，nodes是其中的物体、人、属性等，edges是nodes之间的关系。使用不同的颜色区分节点。"
            "这是关于一个视频的描述，你可以适当推理视频中发生的事情，并生成相关的nodes和edges。你需要注意同一个人或物会在不同帧的描述中可能有不同的描述。"
            "请以 Python 字典格式输出，字典中必须包含 'nodes' 和 'edges' 两个键。例如：\n"
            "{'nodes': [{'id': 1, 'label': '示例', 'color': 'red'}], "
            "'edges': [{'source': 1, 'target': 2, 'label': '关联', 'color': 'blue'}]}"
            "这个知识图谱应当包含视频中出现过的所有物体，以及他们各自之间的关系。edge中可以包含时间信息"
            f"以下是文字描述内容：\n\n{text}\n\n"
        )
        
    messages = [
        {
            "role": "user",
            "content": prompt
        }
    ]
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=messages,
        temperature=0,
    )
    kg_str = response.choices[0].message.content
    kg_str = clean_kg_text(kg_str)
    try:
        kg_dict = ast.literal_eval(kg_str)
    except Exception as e:
        kg_dict = {"nodes": [], "edges": []}
    return kg_dict

# ========== 6. 前端上传组件 ==========

def ImageUpload():
    uploaded_image = solara.use_reactive(None)
    uploader = widgets.FileUpload(accept="image/*", multiple=False)
    
    def on_upload_change(change):
        if uploader.value:
            if isinstance(uploader.value, dict):
                for fname, fileinfo in uploader.value.items():
                    uploaded_image.value = fileinfo["content"]
                    break
            else:
                for fileinfo in uploader.value:
                    uploaded_image.value = fileinfo["content"]
                    break
    uploader.observe(on_upload_change, names="value")
    return uploader, uploaded_image

@solara.memoize
def load_model():
    model = AutoModelForCausalLM.from_pretrained(
        "vikhyatk/moondream2",
        revision="2025-01-09",
        trust_remote_code=True,
        device_map={"": "cuda"}  # 确保 GPU 计算
    )
    return model

def VideoUpload():
    
    uploaded_video = solara.use_reactive(None)
    uploader = widgets.FileUpload(accept="video/*", multiple=False)
    
    def on_upload_change(change):
        if uploader.value:
            if isinstance(uploader.value, dict):
                for fname, fileinfo in uploader.value.items():
                    uploaded_video.value = fileinfo["content"]
                    break
            else:
                for fileinfo in uploader.value:
                    uploaded_video.value = fileinfo["content"]
                    break
    uploader.observe(on_upload_change, names="value")
    return uploader, uploaded_video

# ========== 7. 前端主组件：支持图片与视频 ==========

@solara.component
def MediaToKnowledgeGraph():
    media_type = solara.use_reactive("Image")
    caption_text = solara.use_reactive("")
    kg_result = solara.use_reactive(None)
    error_message = solara.use_reactive("")
    debug_message = solara.use_reactive("")
    video_frames = solara.use_reactive([])

    # 定义两个上传组件（始终创建，但仅显示选中项）
    image_uploader, image_data = ImageUpload()
    video_uploader, video_data = VideoUpload()

    model = load_model()
    
    solara.Markdown("### Choose the media type")
    # 使用正确的参数 "values" 替换之前错误的 "options"
    solara.Select(label="Media Type", value=media_type, values=["Image", "Video"])

    # 根据选择显示对应的上传控件
    if media_type.value == "Image":
        solara.Markdown("#### Upload Image")
        solara.display(image_uploader)
    elif media_type.value == "Video":
        solara.Markdown("#### Upload Video")
        solara.display(video_uploader)

    def process_pipeline():
        error_message.value = ""
        if media_type.value == "Image":
            if not image_data.value:
                error_message.value = "Please upload an image first."
                logger.log("No image uploaded.")
            else:
                logger.log("Processing image.")
                caption = generate_caption(image_data.value,model)
                logger.log("Image description generated successfully.")
                caption_text.value = caption
                kg = generate_kg(caption_text.value)
                kg_result.value = kg
                logger.log("Knowledge graph generated successfully.")
        elif media_type.value == "Video":
            if not video_data.value:
                error_message.value = "Please upload a video first."
                logger.log("No video uploaded.")
            else:
                logger.log("Processing video.")
                # 使用新函数，返回 PIL.Image 对象列表
                frames = extract_frames_from_video(video_data.value, desired_fps=1, truncation=15, max_frames=10)
                # 为了显示，将 PIL.Image 对象转换为 data URL
                video_frames.value = frames
                logger.log(f"Extracted {len(frames)} frames.")
                captions = generate_understanding_video(frames,model,VIDEO_MODE)
                logger.log("Video description generated successfully.")
                if VIDEO_MODE == "video-multiframe":
                    caption_text.value = captions[0]
                else:
                    for i in range(len(captions)):
                        caption_text.value += f"Step {i+1}: {captions[i]}\n"
                kg = generate_kg(caption_text.value, mode=VIDEO_MODE)
                kg_result.value = kg
                logger.log("Knowledge graph generated successfully.")

        logger.log("Process completed.")
        debug_message.value = logger.get_logs()
        print(logger.get_logs())

    solara.Button(label="Generate Results", on_click=process_pipeline)

    # 显示结果区域
    if media_type.value == "Image" and image_data.value:
        data_url = "data:image/png;base64," + base64.b64encode(image_data.value).decode("utf-8")
        solara.Markdown(f"### Image Preview\n![Image Preview]({data_url})")
    if media_type.value == "Image" and caption_text.value:
        solara.Markdown("### Image Description")
        solara.Markdown(caption_text.value)
    if media_type.value == "Video" and video_frames.value:
        solara.Markdown("### Video Frame Preview")
        for frame_bytes in video_frames.value:
            solara.Markdown(f"![Video Frame](data:image/jpeg;base64,{base64.b64encode(frame_bytes).decode('utf-8')})")
    if media_type.value == "Video" and caption_text.value:
        solara.Markdown("### Video Description")
        solara.Markdown(caption_text.value)
    if media_type.value == "Image" and kg_result.value or media_type.value == "Video" and kg_result.value:
        solara.Markdown("### Knowledge Graph")
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
            solara.Markdown(f"![Knowledge Graph]({svg_data_url})")
        except Exception as e:
            solara.Markdown("Failed to generate knowledge graph image: " + str(e))
    
    if debug_message.value:
        solara.Markdown("### Debug Information")
        solara.Markdown(debug_message.value)

@solara.component
def Page():
    solara.Markdown("# Image/Video Knowledge Graph Generator")
    solara.Markdown("Upload an image, and the system will first generate a description of the image through the visual model, then generate a knowledge graph based on the description; when uploading a video, the system will extract 5 frames per second (output as PIL.Image, and converted to a data URL for display).")
    MediaToKnowledgeGraph()



def main():
    Page()

if __name__ == "__main__":
    
    main()
