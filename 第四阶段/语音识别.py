from cgq_api import led




while True:
    a = input("请说出你的操作：")

    if a == '开灯':
        led.led_control("ON")
    elif a == '关灯':
        led.led_control("OFF")
