# AI 方言保护与传播平台

面向“人工智能+”与新文科融合赛道的完整端到端解决方案：以方言音频为输入，自动完成语音识别（ASR）→文本规范化→可选普通话翻译→语音合成（TTS），并提供 FastAPI 后端、Streamlit 前端、评测脚本与可复现实验流程，支持单机部署与 Docker 一键启动。

## 平台亮点
- 🔄 **端到端流程**：统一管线 `pipeline_infer.py` 串联 ASR、文本处理、TTS，支持批量推理与热词注入。
- 🧠 **模型可替换**：默认 Whisper 小模型与 TensorFlowTTS，可扩展 Sherpa-ONNX、MeloTTS、TensorFlowASR 等方案。
- 🧰 **工程化支撑**：FastAPI 提供 `/asr` `/tts` `/pipeline` JSON API，Streamlit Web 展示上传、参数设置、日志与下载。
- 📊 **评测闭环**：`evaluate/asr_eval.py` 计算 CER/WER，`evaluate/tts_eval.py` 提供 F0/能量客观指标与 MOS 主观模板。
- 🧪 **可复现实验**：`scripts/prepare_data.py` 生成 16 kHz 处理样例，`scripts/demo_dataset_make.sh` 展示批处理进度，CI 持续验证。
- 🚀 **部署友好**：`docker-compose.yml` 支持 GPU/CPU 自动回退，`makefile` 集成常用命令，`ROADMAP.md` 指引后续增强。

## 仓库结构
```
├── asr/                    # Whisper 推理与 TensorFlowASR stub
├── tts/                    # TensorFlowTTS 推理封装
├── nlp/                    # 方言规范化规则
├── service/                # FastAPI 后端
├── web/                    # Streamlit 前端
├── scripts/                # 数据准备与 demo 脚本
├── evaluate/               # ASR/TTS 评测
├── configs/config.yaml     # 全局配置
├── pipeline_infer.py       # 命令行批处理
├── tests/test_pipeline.py  # 最小端到端单测
├── docker/                 # Dockerfile
├── docker-compose.yml      # 前后端一键编排
├── ROADMAP.md              # 后续增强规划
├── makefile                # 常用命令
└── assets/                 # 示例音频与输出缓存
```

## 安装与环境准备
> 需要 Python 3.10。首次运行可能下载预训练权重，建议确保网络畅通。

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

如需 GPU 推理，请提前安装匹配 CUDA/cuDNN，并将 `DEVICE=gpu` 传给服务端或 Docker。

## 快速开始
1. **准备样例数据**（生成 16 kHz、≤15 秒 demo）：
   ```bash
   python scripts/prepare_data.py
   ```
2. **本地运行**：
   ```bash
   # 后端
   uvicorn service.main:app --host 0.0.0.0 --port 8000
   # 前端（另启终端）
   streamlit run web/app.py
   ```
3. 打开浏览器访问 `http://localhost:8501`，上传方言 WAV，点击“一键处理”即可查看转写、规范化文本、（可选）普通话翻译、合成音频播放与下载。

### 一键演示脚本
```bash
make demo
```
会生成 demo 数据、运行管线，输出保存在 `assets/demo_outputs/`。

### Docker 单机部署
```bash
docker-compose up --build
```
默认启动两个容器：`api`（FastAPI）与 `web`（Streamlit）。设置 `DEVICE=gpu` 与 `RUNTIME=nvidia` 即可启用 GPU；若环境无 GPU，会自动退回 CPU。

## 评测方法
- **ASR**：运行 `python evaluate/asr_eval.py --ref data/processed/metadata.csv --hyp assets/demo_outputs/transcripts.csv`，输出 CER/WER。
- **TTS**：运行 `python evaluate/tts_eval.py --ref_dir data/processed --syn_dir assets/demo_outputs` 获取 F0/能量差异，并按照脚本附带的 MOS 模板开展主观听感评测。
- **流水线测试**：`pytest -q`（CI 中自动执行）会在 stub 模式下跑 15 秒以内的样例，保障端到端流程。

## 数据合规与知识产权
- 所有 demo 数据均为合成波形，不含真实语音，避免侵权风险。
- 采集真实方言时需获得说话者授权，保留原始录音与授权文件，并遵循《个人信息保护法》相关条款。
- 对外提供模型/数据时，应标注许可证、注明用途限制，避免二次传播引起的知识产权纠纷。

## 赛道适配建议
- **新文科融合**：在高校外语、汉语言课程中嵌入方言数字化实验，结合地方文化课程设计跨学科项目。
- **“人工智能+教育”**：利用平台自动生成语音教材，搭建听说训练系统；可结合学习分析追踪学生掌握情况。
- **“人工智能+文化传承”**：配合地方档案馆、非遗项目，构建方言语音库与知识图谱，展示文化故事、词条解释。
- **创新创业赛事**：基于 API 开发移动端 App，提供方言识别、翻译、语音导览等功能，打造地域品牌服务。

## 可复现实验流程
1. 使用 `scripts/prepare_data.py` 生成或整理本地语料，确保采样率 16 kHz、时长 ≤15 s。
2. 编辑 `configs/config.yaml`，指定 ASR/TTS 模型、缓存目录与默认热词。
3. 运行 `pipeline_infer.py --input_dir data/processed --output_dir assets/demo_outputs` 批量推理。
4. 调整 `nlp/normalize.py` 扩展词典、规则；在 `asr/tf_conformer_stub.py` 中填充微调脚本模板，实现本地化优化。
5. 利用评测脚本对比模型版本，记录指标与主观反馈。

## 评测与演示记录建议
- 使用 `scripts/demo_dataset_make.sh` 或 `make demo` 自动生成样例，配合 OBS 等工具录制前后端实操视频。
- 视频脚本参考：介绍项目背景 → 展示数据准备 → 演示网页上传、识别、规范化与合成 → 展示评测报告 → 说明部署与扩展计划。

## 单机部署要点
- 若无 GPU，请保持 `DEVICE=cpu`，Whisper/TTS 将自动切换至 CPU 或 stub 推理。
- 首次运行建议预下载模型并缓存至 `assets/cache/`，减少离线环境重复下载。
- 使用 `make docker-up` 可将缓存卷映射到容器，避免重复加载模型。

## 常用命令速查
```bash
make setup       # 创建虚拟环境并安装依赖
make run_api     # 启动 FastAPI 服务
make run_web     # 启动 Streamlit 前端
make demo        # 生成 demo 数据并跑通管线
make eval        # 运行 ASR 评测示例
make lint        # PEP 8 基础语法检查
make test        # 运行单元测试
make docker-build
make docker-up
```

## 知识产权与数据合规提醒
- 落地时请遵守国家/地方数据出境与隐私法规，必要时对音频做匿名化处理。
- 若平台开放给公众使用，应在用户协议中明确上传数据的用途、存储周期与授权方式。
- 对引入的第三方模型/权重进行许可证审查（如 Apache 2.0、MIT 等），避免商业化冲突。

## 更多资源
- `ROADMAP.md`：未来计划，包括方言微调、说话人自适应、移动端部署等。
- `asr/tf_conformer_stub.py`：TensorFlowASR 微调占位脚本，可替换为真实训练流程。
- `tts/tensorflowtts_infer.py`：TTS 推理封装，便于切换 MeloTTS、VITS 等模型。

欢迎在 Issues/PR 中反馈问题或贡献功能，让方言数字化更易落地。
