import time

def led_control(status):
    try:
        if status == "ON":
            print("灯亮了")
            return '灯亮了'
        elif status == "OFF":

            print("灯关了")
            return '灯关了'


    except:
        print('')

if __name__ == '__main__':
    while True:
        led_control("NO")
        led_control("OFF")