import speech_recognition as sr

# 初始化识别器
r = sr.Recognizer()

# 从麦克风捕获音频
with sr.Microphone() as source:
    print("请说些什么：")
    audio = r.listen(source)

# 尝试识别音频中的语音
try:
    # 使用Google网络语音API来识别音频
    text = r.recognize_google(audio, language="zh-CN")  # 设置语言为中文
    print("你说了：" + text)
except sr.UnknownValueError:
    print("Google网络语音API无法理解音频")
except sr.RequestError as e:
    print(f"无法从Google网络语音API请求结果；{e}")

print("Google")