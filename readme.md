# 🎙️ Audio2Face-Unity-LiveStreaming

> **Real-time speech-to-lip sync streaming from Python to Unity, inspired by the FaceFormer architecture.**
>
> 基于 FaceFormer 架构的语音驱动口型生成，与 Unity 实时推送演示系统。

---
## 🎬 效果演示 (Demo Showcase)

| 🟢 仅眨眼 (Blink Only) | 🗣️ 语音驱动口型 (Speech-Driven LipSync) |
| :---: | :---: |


https://github.com/user-attachments/assets/e703f0ef-78f0-4bd1-a7cd-90ea98a26f80

 

https://github.com/user-attachments/assets/614bbe50-fbdb-4f28-ae0c-8056a7203781




>  Speak 视频包含Elevenlabs生成的音频，请开启声音以获得最佳体验！

---

## 🛠️ 资源与数据准备 (Resources & Data Prep)

### 1. 🎨 模型资源 (Model Assets)
* **[CN]** 本项目使用的是从 [模之屋 (Aplaybox)](https://www.aplaybox.com/) 下载的原神 MMD 模型。为了让模型“动”起来，我使用了 Blender 的 MMD 插件导入模型，并借助 **Faceit** 插件为其制作了符合 ARKit 标准的 52 个面部 Blendshapes。（*小贴士：捏脸这一步需要一点耐心和经验，强烈建议跟着 Faceit 官网的手把手教程来做！*）
* **[EN]** The adorable Genshin Impact MMD models were downloaded from [Aplaybox](https://www.aplaybox.com/). I used the MMD plugin to import them into Blender, and the **Faceit** plugin to generate the 52 ARKit-standard facial Blendshapes. (*Tip: This step requires a bit of practice. Highly recommend following the step-by-step tutorials on the Faceit website!*)

### 2. 🎬 训练数据 (Training Data - DIY Time!)
* **[CN]** 如果你想从头开始“炼丹”，你需要准备自己的数据集。掏出你的 iPhone，下载 **Live Link Face** App 并开启 ARKit 模式。尽情录制各种说话视频（比如绕口令、激情的角色对白、夸张的元音发音等，越丰富越好）。录制完成后导出到电脑，解压 zip 文件，你就能得到包含 Blendshapes 数据的 `.csv` 文件和包含音频的 `.mov` 文件啦。
* **[EN]** Want to train the model from scratch? Grab your iPhone, download the **Live Link Face** app, and switch to ARKit mode. Record yourself speaking a variety of things (tongue twisters, dramatic monologues, exaggerated vowel sounds... go wild!). Export and unzip the file to get the `.csv` (Blendshapes data) and `.mov` (audio) files.

---

## 🚀 如何运行？(How to Use)

	0. **Python 端依赖安装**：
请确保你安装了 Python 3.10。进入 Python 项目目录，运行以下命令安装所需依赖：

```bash
cd Python_Server
pip install -r requirements.txt
```

如果你只想看看效果（不想训练），请遵循以下步骤：
1. **启动 Unity 接收端：** 打开 Unity 工程，在场景中新建一个空对象 (Empty Object)。将 `Assets/Scripts` 目录下的相关接收脚本挂载到该对象上，并在 Inspector 面板中将端口号 (Port) 修改为你想要的数字。点击 Play 运行场景。
2. **运行 Python 推理端：** 找到 `inference_unity.py` 文件，在代码中填入你本机的 IP 地址以及刚才在 Unity 中设置的端口号。运行脚本，见证奇迹的时刻！

* **[EN]** Don't want to train? Just run the inference!
  1. Open the Unity project, create an Empty GameObject, and attach the scripts found in `Assets/Scripts`. Configure your preferred Port number in the Inspector, then hit Play in Unity.
  2. Open `inference_unity.py`, fill in your IP address and the corresponding Port. Run the script and watch the magic happen!

### 🏋️ 硬核训练 (Training Mode)
* **[CN]** 如果你对底层逻辑感兴趣，所有的训练代码都包含在 Python 文件夹中了，里面不仅有训练脚本，还附带了数据裁剪和清洗的预处理代码。
* **[EN]** If you're interested in the training process, all the training codes (including data cropping and cleaning scripts) are available in the Python folder. Dive in!

---

## 💻 运行环境 (Environment & Versions)
* **Unity:** `2022.3.30f1`
* **Python:** `3.10` (Running on WSL: Ubuntu-22.04)
