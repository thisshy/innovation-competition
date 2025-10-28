"""Streamlit front-end for the AI 方言保护与传播平台."""

from __future__ import annotations

import base64
import json
import os
from pathlib import Path
from typing import Dict
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

import streamlit as st

API_BASE_URL = os.getenv("API_BASE_URL", "http://localhost:8000")


def post_json(endpoint: str, payload: Dict) -> Dict:
    url = f"{API_BASE_URL}{endpoint}"
    data = json.dumps(payload).encode("utf-8")
    request = Request(url, data=data, headers={"Content-Type": "application/json"})
    with urlopen(request) as response:  # noqa: S310 - controlled URL
        return json.loads(response.read().decode("utf-8"))


def main() -> None:
    st.set_page_config(page_title="AI 方言保护与传播平台", layout="wide")
    st.title("AI 方言保护与传播平台")
    st.write("上传或录制方言音频，体验从识别到合成的一体化流程。")

    if "logs" not in st.session_state:
        st.session_state["logs"] = []

    col_left, col_right = st.columns([2, 1])

    with col_left:
        uploaded_file = st.file_uploader("上传 WAV 文件", type=["wav", "mp3", "m4a"])
        language_hint = st.text_input("语言提示", value="zh")
        hotwords = st.text_input("热词（逗号分隔）", value="方言,文化")
        speed = st.slider("合成语速", min_value=0.5, max_value=1.5, value=1.0, step=0.1)
        enable_translation = st.checkbox("输出普通话翻译", value=True)
        if st.button("一键处理", use_container_width=True):
            if not uploaded_file:
                st.warning("请先上传音频文件。")
            else:
                audio_bytes = uploaded_file.read()
                audio_b64 = base64.b64encode(audio_bytes).decode("utf-8")
                payload = {
                    "audio_base64": audio_b64,
                    "language_hint": language_hint,
                    "hotwords": [word.strip() for word in hotwords.split(",") if word.strip()],
                    "enable_translation": enable_translation,
                    "tts_speed": speed,
                }
                try:
                    response = post_json("/pipeline", payload)
                except HTTPError as err:
                    st.error(f"请求失败：{err.read().decode('utf-8')}")
                    st.session_state.logs.append(f"错误: {err}")
                except URLError as err:
                    st.error(f"无法连接到后端：{err}")
                    st.session_state.logs.append(f"网络错误: {err}")
                else:
                    st.success("处理完成！")
                    st.session_state["last_response"] = response
                    st.session_state.logs.append("处理成功: " + json.dumps(response, ensure_ascii=False))

    with col_right:
        st.subheader("参数")
        st.markdown("- API 地址: ``%s``" % API_BASE_URL)
        st.markdown("- 当前模式: %s" % ("测试" if os.getenv("PIPELINE_TEST_MODE") else "正常"))
        st.caption("首次运行会下载模型，请确保网络连接。")

    st.divider()

    if "last_response" in st.session_state:
        resp = st.session_state["last_response"]
        st.subheader("识别结果")
        st.write(resp.get("transcript", ""))

        st.subheader("规范化文本")
        st.write(resp.get("normalized", ""))

        if resp.get("translation"):
            st.subheader("普通话翻译")
            st.write(resp.get("translation"))

        audio_path = resp.get("tts_audio_path")
        if audio_path:
            st.subheader("合成音频")
            try:
                file_path = Path(audio_path)
                if not file_path.is_absolute():
                    file_path = Path.cwd() / file_path
                with open(file_path, "rb") as fp:
                    audio_bytes = fp.read()
                st.audio(audio_bytes, format="audio/wav")
                st.download_button(
                    label="下载合成音频",
                    data=audio_bytes,
                    file_name=Path(audio_path).name,
                    mime="audio/wav",
                )
            except FileNotFoundError:
                st.warning("未找到合成音频文件，请检查后端配置。")

    st.divider()
    st.subheader("日志")
    for line in st.session_state["logs"][-20:]:
        st.code(line)

    st.caption("若需录音，可使用本地工具录制后上传，或集成 WebRTC 组件实现在线录制。")


if __name__ == "__main__":
    main()
