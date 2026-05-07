import json
import os

import face_recognition
from django.shortcuts import render
from django.http import JsonResponse
from myapp.util.ImageUtil import *
from myapp.util.RandomUtil import *
import cv2 as cv
import pymysql
import uuid
from django.conf import settings
from ultralytics import YOLO
# Create your views here.
"""
jsonResponse:返回json数据
httpResponse：返回字符  单开页面
render：返回html页面 可以携带参数
redirect：重定向
流式输出

"""
ccpd_pt_path=os.path.join(settings.BASE_DIR,"myapp","car.pt")
ccpd_model=YOLO(ccpd_pt_path)
bone_pt_path=os.path.join(settings.BASE_DIR,"myapp","bone_best.pt")
bone_model=YOLO(bone_pt_path)
bottlecap_path=os.path.join(settings.BASE_DIR,"myapp","bottlecap.pt")
bottlecap_model=YOLO(bottlecap_path)
# 定义图像保存地址
face_save_path=os.path.join(settings.BASE_DIR, "faceImages") + "\\"
def index(request):
    return render(request,"index.html")
def login(request):
    return render(request,"login.html")
def bone(request):
    return render(request,"bone.html")
def bottlecap(request):
    return render(request,"bottlecap.html")
def main_page(request):
    return render(request,"main_page.html")

# 返回人脸采集页面
def face_page(request):
    return render(request,"face_collect.html")
def register_page(request):
    return render(request,"register.html")
def detection_page(request):
    return render(request,"detection_detect.html")
def match_page(request):
    return render(request,"face_match.html")
def login_message_collect(request):
    if request.method=="POST":
        data=json.loads(request.body.decode('utf-8'))
        print(f"获取到的前端传来的个人信息是：{data}")
        action=data.get("action")  # 区分登录还是注册
        username=data["username"]
        password=data["password"]

        if action == "register":
            # 注册操作
            print(f"注册账号：{username}，密码：{password}")
            # 检查用户是否已存在
            if select_user_by_username(username):
                return JsonResponse({
                    "status": 500,
                    "msg": "用户名已存在！"
                })
            # 插入新用户
            dst = insert_info_test(username, password)
            if dst:
                return JsonResponse({
                    "status": 200,
                    "msg": "注册成功！"
                })
            else:
                return JsonResponse({
                    "status": 500,
                    "msg": "注册失败！"
                })
        else:
            # 登录操作
            print(f"登录账号：{username}，密码：{password}")
            # 查询用户验证
            result = select_user_by_username(username)
            # 表结构: id, username, password, face_image_path
            if result and result[1] == password:
                return JsonResponse({
                    "status": 200,
                    "msg": "登录成功！"
                })
            else:
                return JsonResponse({
                    "status": 500,
                    "msg": "账号或密码错误！"
                })
    return JsonResponse({
        "status": 500,
        "msg": "请求方式错误，请检查请求方式！"
    })
# 采集人脸信息
def face_collect(request):
    # 判断是否是post请求   GET POST
    if request.method=="POST":
        # 同时获取用户名和图片矩阵（解决request.body只能读取一次的问题）
        username, image_array = get_image_and_username(request)
        print(f"采集人脸的用户名：{username}")

        # 判断当前这帧画面里面有没有人脸
        image_array=cv.cvtColor(image_array, cv.COLOR_BGR2RGB)
        locations_array=face_recognition.face_locations(image_array)
        if len(locations_array)==0:
            return JsonResponse({
                "status":500,
                "msg":"没有检测到人脸！"
            })
        # 否则就是检测到了
        else:
            # 获取图像二进制码 存到本地
            image_byte=get_image_byte(request)

            if not username:
                return JsonResponse({
                    "status": 500,
                    "msg": "用户名不能为空！"
                })

            # 根据用户名查询用户
            user = select_user_by_username(username)
            if not user:
                return JsonResponse({
                    "status": 500,
                    "msg": "用户不存在！"
                })

            # 获取用户 id 作为文件名
            user_id = user[0]
            # 保存人脸图片路径
            face_image_path = face_save_path + str(user_id) + ".jpg"
            # 写入操作
            with open(face_image_path, "wb") as f:
                f.write(image_byte)
                f.close()
            print(f"人脸图片保存路径：{face_image_path}")

            # 更新用户的 face_image_path
            dst = update_user_face_image(username, face_image_path)
            # 判断是否写入成功
            if dst:
                return JsonResponse({
                    "status":200,
                    "msg":"人脸采集成功！"
                })
            return JsonResponse({
                "status":500,
                "msg":"人脸采集失败！"
            })
    return JsonResponse({
        "status":500,
        "msg":"请求方式错误，请检查请求方式！"
    })

#识别人脸
def face_match(request):
    if request.method=="POST":
        image_array=cv.cvtColor(get_image_array(request),cv.COLOR_BGR2RGB)
        locations=face_recognition.face_locations(image_array)
        if len(locations)==0:
            return JsonResponse({
                "status":500,
                "msg":"没有检测到人脸！",
            })
        else:
            image_encodings=face_recognition.face_encodings(image_array,locations)
            face_dir_list=os.listdir(face_save_path)
            print(f"获取到的文件夹列表是：{face_dir_list}")
            for file_name in face_dir_list:
                if not file_name.endswith(".jpg"):
                    continue
                image_path=face_save_path+file_name
                file_image=face_recognition.load_image_file(image_path)
                file_locations=face_recognition.face_locations(file_image)
                if len(file_locations)==0:
                    continue
                file_encodings=face_recognition.face_encodings(file_image,file_locations)
                dst=face_recognition.compare_faces(file_encodings,image_encodings[0],tolerance=0.5)
                if dst[0]:
                    # 从文件名获取用户id，再查用户名
                    try:
                        select_id=int(file_name.split(".")[0])
                        result=select_info(select_id)
                        if result and len(result)>0:
                            username=result[0][1]  # username在第二列
                            return JsonResponse({
                                "status":200,
                                "msg":f"人脸识别成功！用户名是:{username}"
                            })
                    except Exception as e:
                        print(f"识别过程出错：{e}")
                        continue
            return JsonResponse({
                "status":500,
                "msg":"人脸识别失败！"
            })
    return JsonResponse({
        "status":500,
        "msg":"请求方式错误，请检查请求方式！"
    })
DB_CONFIG = {
    "host": "localhost",
    "port": 3306,
    "user": "root",
    "password": "123456",
    "database": "cdut_face_collect",
    "charset": "utf8"
}

def get_db_conn():
    return pymysql.connect(**DB_CONFIG)

def insert_info_test(username, password):
    try:
        conn = get_db_conn()
        cursor = conn.cursor()
        sql = "insert into login(username,password) values(%s,%s)"
        cursor.execute(sql, (username, password))
        conn.commit()
        print("插入成功！")
        return True
    except Exception as e:
        print(f"插入失败！错误信息是：{e}")
        return False

def select_info(id):
    try:
        conn = get_db_conn()
        cursor = conn.cursor()
        sql = "select * from login where id=%s"
        cursor.execute(sql, (id,))
        result = cursor.fetchall()
        print(f"查询到的结果是：{result}")
        return result
    except Exception as e:
        print(f"查询失败！错误信息是：{e}")
        return None

def select_user_by_username(username):
    try:
        conn = get_db_conn()
        cursor = conn.cursor()
        sql = "select * from login where username=%s"
        cursor.execute(sql, (username,))
        result = cursor.fetchall()
        print(f"查询用户结果：{result}")
        return result[0] if result else None
    except Exception as e:
        print(f"查询失败！错误信息是：{e}")
        return None

def update_user_face_image(username, face_image_path):
    try:
        conn = get_db_conn()
        cursor = conn.cursor()
        sql = "UPDATE login SET face_image_path=%s WHERE username=%s"
        cursor.execute(sql, (face_image_path, username))
        conn.commit()
        print("更新用户人脸路径成功！")
        return True
    except Exception as e:
        print(f"更新失败！错误信息是：{e}")
        return False

def ccpd_detect(request):
    if request.method != "POST":
        return JsonResponse({
            "status": 500,
            "msg": "请求方式错误，请使用POST请求！"
        })
    image=cv.cvtColor(get_image_array(request),cv.COLOR_BGR2RGB)
    image_name=f"{uuid.uuid4()}.jpg"
    result_name=f"result_{image_name}"
    image_save_path=os.path.join(settings.BASE_DIR,"myapp","static","imgs",image_name)
    result_save_path=os.path.join(settings.BASE_DIR,"myapp","static","imgs",result_name)
    image_url=f"/static/imgs/{image_name}"
    result_url = f"/static/imgs/{result_name}"
    cv.imwrite(image_save_path,image)
    results=ccpd_model(image_save_path)
    if len(results[0].boxes)==0:
        return JsonResponse({
            "status":500,
            "msg":"没有检测到车牌！",
            "prossed_image_url":image_url,
        })
    else:
        result_image=results[0].plot()
        cv.imwrite(result_save_path,result_image)
        return JsonResponse({
            "status":200,
            "msg":"车牌检测成功！",
            "prossed_image_url":result_url,
        })
def bone_detect(request):
    if request.method != "POST":
        return JsonResponse({
            "status": 500,
            "msg": "请求方式错误，请使用POST请求！"
        })
    image=cv.cvtColor(get_image_array(request),cv.COLOR_BGR2RGB)
    image_name=f"{uuid.uuid4()}.jpg"
    result_name=f"result_{image_name}"
    image_save_path=os.path.join(settings.BASE_DIR,"myapp","static","imgs",image_name)
    result_save_path=os.path.join(settings.BASE_DIR,"myapp","static","imgs",result_name)
    image_url=f"/static/imgs/{image_name}"
    result_url = f"/static/imgs/{result_name}"
    cv.imwrite(image_save_path,image)
    results=bone_model(image_save_path)
    if len(results[0].boxes)==0:
        return JsonResponse({
            "status":500,
            "msg":"没有检测到骨折！",
            "prossed_image_url":image_url,
        })
    else:
        result_image=results[0].plot()
        cv.imwrite(result_save_path,result_image)
        return JsonResponse({
            "status":200,
            "msg":"骨折检测成功！",
            "prossed_image_url":result_url,
        })
def bottlecap_detect(request):
    if request.method != "POST":
        return JsonResponse({
            "status": 500,
            "msg": "请求方式错误，请使用POST请求！"
        })
    image=cv.cvtColor(get_image_array(request),cv.COLOR_BGR2RGB)
    image_name=f"{uuid.uuid4()}.jpg"
    result_name=f"result_{image_name}"
    image_save_path=os.path.join(settings.BASE_DIR,"myapp","static","imgs",image_name)
    result_save_path=os.path.join(settings.BASE_DIR,"myapp","static","imgs",result_name)
    image_url=f"/static/imgs/{image_name}"
    result_url = f"/static/imgs/{result_name}"
    cv.imwrite(image_save_path,image)
    results=bottlecap_model(image_save_path)
    if len(results[0].boxes)==0:
        return JsonResponse({
            "status":500,
            "msg":"没有检测到瓶盖！",
            "prossed_image_url":image_url,
        })
    else:
        result_image=results[0].plot()
        cv.imwrite(result_save_path,result_image)
        return JsonResponse({
            "status":200,
            "msg":"瓶盖检测成功！",
            "prossed_image_url":result_url,
        })
if __name__=="__main__":
    select_info(80)