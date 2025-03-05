# app.py
import os
import re
import ast
import json
import base64
import requests

import solara
import ipywidgets as widgets
from graphviz import Digraph

# 如果你需要从 .env 里读取 OPENAI_API_KEY，请保留以下:
try:
    from dotenv import load_dotenv, find_dotenv
    load_dotenv(find_dotenv())
except ImportError:
    pass

import openai
openai.api_key = os.getenv("OPENAI_API_KEY", "")

# ---------------------------
# 备用图片描述（当GPT或视觉模型返回不理想时使用）
# ---------------------------
fallback_caption = (
    "这张图片展示了一片公园景色，画面主要由以下元素构成：\n\n"
    "草地：\n"
    "画面大部分区域被绿色的草坪覆盖，草地整洁且富有生机。...\n\n"
    "..."
)

# ---------------------------
# 用于清理知识图谱的文本（提取代码块并替换部分中文标点）
# ---------------------------
def clean_kg_text(text: str) -> str:
    # 提取 Python 代码块
    import re
    match = re.search(r"```(?:python)?\s*(.*?)\s*```", text, re.DOTALL)
    if match:
        text = match.group(1)
    # 替换常见中文标点
    text = text.replace("，", ",").replace("：", ":")
    return text

# ---------------------------
# 生成图片描述（假设有 GPT-4o 或其他视觉模型）
# ---------------------------
def generate_caption(image_bytes: bytes) -> str:
    # 这里演示固定返回，你可自行改写为调用视觉API的逻辑
    try:
        # 假设直接返回一个描述
        return "假设这是 GPT-4o 的图片详细描述..."
    except Exception as e:
        print(f"生成图片描述异常: {e}")
        return fallback_caption

# ---------------------------
# 生成知识图谱（模拟 GPT 返回）
# ---------------------------
def generate_kg(text: str) -> dict:
    # 演示：给出一个写死的 JSON 格式字符串，无任何缩进错误
    gpt_response = """{
  "nodes": [
    {"id": 1, "label": "公园", "color": "green"},
    {"id": 2, "label": "草地", "color": "brown"},
    {"id": 3, "label": "树木", "color": "blue"}
  ],
  "edges": [
    {"source": 1, "target": 2, "label": "包含", "color": "black"},
    {"source": 1, "target": 3, "label": "拥有", "color": "black"}
  ]
}"""

    kg_str = clean_kg_text(gpt_response)
    try:
        kg_dict = ast.literal_eval(kg_str)
        return kg_dict
    except Exception as e:
        print("知识图谱解析错误:", e)
        return {"nodes": [], "edges": []}

# ---------------------------
# 文件上传组件（ipywidgets）
# ---------------------------
def ImageUpload():
    uploaded_image = solara.use_reactive(None)
    uploader = widgets.FileUpload(accept="image/*", multiple=False)

    def on_upload_change(change):
        if uploader.value:
            # 兼容：有时是 list，有时是 dict
            if isinstance(uploader.value, dict):
                for fname, fileinfo in uploader.value.items():
                    uploaded_image.value = fileinfo["content"]
                    print(f"上传成功：{fname}, 大小：{len(fileinfo['content'])} 字节")
                    break
            else:
                for fileinfo in uploader.value:
                    uploaded_image.value = fileinfo["content"]
                    print(f"上传成功（非字典格式）：大小：{len(fileinfo['content'])} 字节")
                    break

    uploader.observe(on_upload_change, names="value")
    return uploader, uploaded_image

# ---------------------------
# 主组件：图片 -> 描述 -> 知识图谱
# ---------------------------
@solara.component
def ImageToKnowledgeGraph():
    uploader, uploaded_image = ImageUpload()
    caption_text = solara.use_reactive("")
    kg_result = solara.use_reactive(None)
    error_message = solara.use_reactive("")
    debug_message = solara.use_reactive("")

    def process_pipeline():
        debug_info = ""
        error_message.value = ""

        if not uploaded_image.value:
            error_message.value = "请先上传图片。"
            debug_info += "未检测到图片上传。\n"
            debug_message.value = debug_info
            return

        debug_info += "开始处理流程。\n"

        # 1. 生成图片描述
        try:
            caption = generate_caption(uploaded_image.value)
            if not caption or len(caption.strip()) < 20:
                debug_info += "生成的图片描述较短，使用备用描述。\n"
                caption = fallback_caption
            else:
                debug_info += "生成图片描述成功。\n"
            caption_text.value = caption
        except Exception as e:
            error_message.value = f"生成图片描述时出错: {e}"
            debug_info += f"生成图片描述异常: {e}\n"
            caption_text.value = fallback_caption

        # 2. 生成知识图谱
        try:
            kg = generate_kg(caption_text.value)
            kg_result.value = kg
            debug_info += "生成知识图谱成功。\n"

            # -----------------------------
            # 把知识图谱保存到桌面
            # 请确认该路径存在：/mnt/c/Users/12849/Desktop
            # 如果路径不对，请替换为自己Windows用户名或其它目录
            # -----------------------------
            desktop_dir = "/mnt/c/Users/12849/Desktop"
            if not os.path.exists(desktop_dir):
                debug_info += f"警告：桌面目录 {desktop_dir} 不存在。\n"
            else:
                import json
                kg_json_path = os.path.join(desktop_dir, "knowledge_graph.json")
                with open(kg_json_path, "w", encoding="utf-8") as f:
                    json.dump(kg, f, indent=2, ensure_ascii=False)
                debug_info += f"已将知识图谱JSON保存到: {kg_json_path}\n"

                # 同时把 dot 和 svg 保存
                dot_file_path = os.path.join(desktop_dir, "knowledge_graph.dot")
                svg_file_path = os.path.join(desktop_dir, "knowledge_graph.svg")

                # 构造 dot
                dot = Digraph(comment="Knowledge Graph")
                for node in kg["nodes"]:
                    dot.node(str(node["id"]),
                             label=node.get("label", ""),
                             color=node.get("color", "black"))
                for edge in kg["edges"]:
                    dot.edge(str(edge["source"]),
                             str(edge["target"]),
                             label=edge.get("label", ""),
                             color=edge.get("color", "black"))

                # 保存 dot
                dot.save(dot_file_path)
                debug_info += f"已将DOT文件保存到: {dot_file_path}\n"
                # 保存 svg
                svg_data = dot.pipe(format="svg")
                with open(svg_file_path, "wb") as f:
                    f.write(svg_data)
                debug_info += f"已将知识图谱SVG保存到: {svg_file_path}\n"

        except Exception as e:
            error_message.value = f"生成知识图谱时出错: {e}"
            debug_info += f"生成知识图谱异常: {e}\n"

        debug_info += "流程处理完毕。\n"
        debug_message.value = debug_info
        print(debug_info)

    solara.Button(label="生成图片描述和知识图谱", on_click=process_pipeline)

    solara.Markdown("### 上传图片")
    solara.display(uploader)  # 在Solara界面上展示ipywidgets组件

    # 图片预览
    if uploaded_image.value:
        data_url = "data:image/png;base64," + base64.b64encode(uploaded_image.value).decode("utf-8")
        solara.Markdown("### 图片预览")
        solara.HTML(f"<img src='{data_url}' alt='上传图片预览' style='max-width:100%; height:auto;'/>")

    # 显示图片描述
    if caption_text.value:
        solara.Markdown("### 图片描述")
        solara.Markdown(caption_text.value)

    # 显示错误信息
    if error_message.value:
        solara.Markdown(f"**错误：** {error_message.value}")

    # 知识图谱可视化
    if kg_result.value:
        solara.Markdown("### 知识图谱")
        dot = Digraph(comment="Knowledge Graph")
        for node in kg_result.value["nodes"]:
            dot.node(str(node["id"]),
                     label=node.get("label", ""),
                     color=node.get("color", "black"))
        for edge in kg_result.value["edges"]:
            dot.edge(str(edge["source"]),
                     str(edge["target"]),
                     label=edge.get("label", ""),
                     color=edge.get("color", "black"))

        try:
            svg_data = dot.pipe(format="svg").decode("utf-8")
            solara.HTML(svg_data)
        except Exception as e:
            solara.Markdown("生成知识图谱图像失败：" + str(e))

    # 调试信息
    if debug_message.value:
        solara.Markdown("### 调试信息")
        solara.Markdown(debug_message.value)

# ---------------------------
# 应用主页面
# ---------------------------
@solara.component
def Page():
    solara.Markdown("# 图片到知识图谱生成器")
    solara.Markdown("上传图片后，系统将先生成图片描述，再根据描述生成知识图谱，并将结果保存到桌面。")
    ImageToKnowledgeGraph()

# Solara入口
Page()
