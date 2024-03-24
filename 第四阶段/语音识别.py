import RPi.GPIO as GPIO
import speech_recognition as sr
import time

# 设置GPIO模式为BCM
GPIO.setmode(GPIO.BCM)

# 设置风扇控制的引脚编号，这里假设是17号引脚
FAN_PIN = 17

# 设置引脚为输出模式
GPIO.setup(FAN_PIN, GPIO.OUT)

# 初始化语音识别器
r = sr.Recognizer()

def turn_on_fan():
    # 打开风扇
    GPIO.output(FAN_PIN, GPIO.HIGH)
    print("风扇已打开")

def turn_off_fan():
    # 关闭风扇
    GPIO.output(FAN_PIN, GPIO.LOW)
    print("风扇已关闭")

def listen_for_commands():
    with sr.Microphone() as source:
        print("请说出命令（打开/关闭）:")
        audio = r.listen(source)

    try:
        # 使用Google语音识别服务
        command = r.recognize_google(audio, language='zh-CN').lower()
        print("你说的命令是: " + command)

        if "打开" in command:
            turn_on_fan()
        elif "关闭" in command:
            turn_off_fan()
        else:
            print("未知命令，请说'打开'或'关闭'")

    except sr.UnknownValueError:
        print("无法识别音频")
    except sr.RequestError as e:
        print("无法从Google语音识别服务获取结果; {0}".format(e))

try:
    print("程序开始运行，请等待语音命令...")
    while True:
        listen_for_commands()
        time.sleep(1)  # 等待1秒，避免频繁监听

except KeyboardInterrupt:
    # 当用户按下Ctrl+C时，清理GPIO并退出程序
    print("程序已退出")
    GPIO.cleanup()