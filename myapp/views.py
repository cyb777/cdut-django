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
Django 视图层 — 骨折 / 车牌 / 瓶盖检测 + 人脸识别 4 类业务,详见
docs/lectureNotes.md。

jsonResponse: 返回 JSON 数据
httpResponse: 返回字符 / 单开页面
render:       返回 HTML 页面,可携带参数
redirect:     重定向
流式输出
"""


# ==================== 模型加载(进程级单例) ====================
# 启动时一次性把 3 个 YOLO 权重读进内存,首请求不再读盘。
# 详见 docs/lectureNotes.md 重点 1。
WEIGHTS_DIR = os.path.join(settings.BASE_DIR, "myapp", "weights")
ccpd_model      = YOLO(os.path.join(WEIGHTS_DIR, "car.pt"))        # 车牌检测
bone_model      = YOLO(os.path.join(WEIGHTS_DIR, "bone_best.pt"))  # 骨折检测(10 类)
bottlecap_model = YOLO(os.path.join(WEIGHTS_DIR, "bottlecap.pt"))  # 瓶盖检测

# 人脸采集图片存盘根目录,文件名约定为 <user_id>.jpg,见重点 3。
face_save_path = os.path.join(settings.BASE_DIR, "faceImages") + "\\"


# ==================== 页面跳转(render) ====================
# 详见 docs/lectureNotes.md 文件 3 路由表,page_xxx 类都走这里。

def index(request):
    """首页 / 项目主页。"""
    return render(request, "index.html")

def login(request):
    """登录页面。"""
    return render(request, "login.html")

def main_page(request):
    """主导航页。"""
    return render(request, "main_page.html")

def register_page(request):
    """注册页面。"""
    return render(request, "register.html")

def detection_page(request):
    """通用检测入口页(下拉选车牌/骨折/瓶盖)。"""
    return render(request, "detection_detect.html")

def face_page(request):
    """人脸采集页面(摄像头帧 + 用户名)。"""
    return render(request, "face_collect.html")

def match_page(request):
    """人脸识别匹配页面(摄像头帧,1:N 识别)。"""
    return render(request, "face_match.html")

def bone(request):
    """骨折检测页面。"""
    return render(request, "bone.html")

def bottlecap(request):
    """瓶盖检测页面。"""
    return render(request, "bottlecap.html")


# ==================== 登录 / 注册(API) ====================
# POST /login_message_collect/, body.action 区分 login / register。
# SQL 走 select_user_by_username / insert_info_test,见重点 4。

def login_message_collect(request):
    """登录 / 注册 双 action 入口。"""
    if request.method == "POST":
        data = json.loads(request.body.decode('utf-8'))
        print(f"获取到的前端传来的个人信息是:{data}")
        action = data.get("action")  # 区分登录还是注册
        username = data["username"]
        password = data["password"]

        if action == "register":
            # 注册操作
            print(f"注册账号:{username},密码:{password}")
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
            print(f"登录账号:{username},密码:{password}")
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


# ==================== 人脸识别(API) ====================
# 采集 face_collect / 匹配 face_match,见重点 3。

def face_collect(request):
    """采集人脸 → 按 user_id 存 jpg → 更新 DB face_image_path。"""
    # 判断是否是 post 请求   GET POST
    if request.method == "POST":
        # 同时获取用户名和图片矩阵(解决 request.body 只能读取一次的问题)
        username, image_array = get_image_and_username(request)
        print(f"采集人脸的用户名:{username}")

        # 判断当前这帧画面里面有没有人脸
        image_array = cv.cvtColor(image_array, cv.COLOR_BGR2RGB)
        locations_array = face_recognition.face_locations(image_array)
        if len(locations_array) == 0:
            return JsonResponse({
                "status": 500,
                "msg": "没有检测到人脸！"
            })
        # 否则就是检测到了
        else:
            # 获取图像二进制码 存到本地
            image_byte = get_image_byte(request)

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
            print(f"人脸图片保存路径:{face_image_path}")

            # 更新用户的 face_image_path
            dst = update_user_face_image(username, face_image_path)
            # 判断是否写入成功
            if dst:
                return JsonResponse({
                    "status": 200,
                    "msg": "人脸采集成功！"
                })
            return JsonResponse({
                "status": 500,
                "msg": "人脸采集失败！"
            })
    return JsonResponse({
        "status": 500,
        "msg": "请求方式错误，请检查请求方式！"
    })


def face_match(request):
    """人脸识别 1:N — 遍历 faceImages/,compare_faces(tolerance=0.5)。"""
    if request.method == "POST":
        image_array = cv.cvtColor(get_image_array(request), cv.COLOR_BGR2RGB)
        locations = face_recognition.face_locations(image_array)
        if len(locations) == 0:
            return JsonResponse({
                "status": 500,
                "msg": "没有检测到人脸！",
            })
        else:
            image_encodings = face_recognition.face_encodings(image_array, locations)
            face_dir_list = os.listdir(face_save_path)
            print(f"获取到的文件夹列表是:{face_dir_list}")
            for file_name in face_dir_list:
                if not file_name.endswith(".jpg"):
                    continue
                image_path = face_save_path + file_name
                file_image = face_recognition.load_image_file(image_path)
                file_locations = face_recognition.face_locations(file_image)
                if len(file_locations) == 0:
                    continue
                file_encodings = face_recognition.face_encodings(file_image, file_locations)
                dst = face_recognition.compare_faces(file_encodings, image_encodings[0], tolerance=0.5)
                if dst[0]:
                    # 从文件名获取用户 id,再查用户名
                    try:
                        select_id = int(file_name.split(".")[0])
                        result = select_info(select_id)
                        if result and len(result) > 0:
                            username = result[0][1]  # username 在第二列
                            return JsonResponse({
                                "status": 200,
                                "msg": f"人脸识别成功！用户名是:{username}"
                            })
                    except Exception as e:
                        print(f"识别过程出错:{e}")
                        continue
            return JsonResponse({
                "status": 500,
                "msg": "人脸识别失败！"
            })
    return JsonResponse({
        "status": 500,
        "msg": "请求方式错误，请检查请求方式！"
    })


# ==================== 数据库(pymysql 直连) ====================
# 4 个函数走 cursor.execute(sql, params) 参数化,见重点 4。
# 见 docs/baseMem.md A1 — 不接 Django ORM 的原因。

DB_CONFIG = {
    "host": "localhost",
    "port": 3306,
    "user": "root",
    "password": "123456",
    "database": "cdut_face_collect",
    "charset": "utf8"
}

def get_db_conn():
    """建一个新连接(每次调用都新建,简单但不高效,见 devPlan T-002)。"""
    return pymysql.connect(**DB_CONFIG)

def insert_info_test(username, password):
    """注册 — 插 login 表(参数化防 SQL 注入)。"""
    try:
        conn = get_db_conn()
        cursor = conn.cursor()
        sql = "insert into login(username,password) values(%s,%s)"
        cursor.execute(sql, (username, password))
        conn.commit()
        print("插入成功！")
        return True
    except Exception as e:
        print(f"插入失败！错误信息是:{e}")
        return False

def select_info(id):
    """人脸识别命中 — 按 id 查 username。"""
    try:
        conn = get_db_conn()
        cursor = conn.cursor()
        sql = "select * from login where id=%s"
        cursor.execute(sql, (id,))
        result = cursor.fetchall()
        print(f"查询到的结果是:{result}")
        return result
    except Exception as e:
        print(f"查询失败！错误信息是:{e}")
        return None

def select_user_by_username(username):
    """登录 / 采集人脸 — 按 username 查(返回第一条)。"""
    try:
        conn = get_db_conn()
        cursor = conn.cursor()
        sql = "select * from login where username=%s"
        cursor.execute(sql, (username,))
        result = cursor.fetchall()
        print(f"查询用户结果:{result}")
        return result[0] if result else None
    except Exception as e:
        print(f"查询失败！错误信息是:{e}")
        return None

def update_user_face_image(username, face_image_path):
    """采集人脸 — 更新 face_image_path。"""
    try:
        conn = get_db_conn()
        cursor = conn.cursor()
        sql = "UPDATE login SET face_image_path=%s WHERE username=%s"
        cursor.execute(sql, (face_image_path, username))
        conn.commit()
        print("更新用户人脸路径成功！")
        return True
    except Exception as e:
        print(f"更新失败！错误信息是:{e}")
        return False


# ==================== YOLOv8 通用推理管线 ====================
# 3 个 detect 退化为 1 行包装,见重点 2。
# 详见 docs/lectureNotes.md 重点 2。

def _run_yolo(request, model, no_target_msg, success_msg):
    """3 个 YOLO detect 共用的推理管线。

    参数:
        request         Django request(必须 POST,且 body 含 base64 图片)
        model           全局预加载的 YOLO 实例
        no_target_msg   没有检测到目标时的中文字段
        success_msg     检测成功时的中文字段
    """
    if request.method != "POST":
        return JsonResponse({
            "status": 500,
            "msg": "请求方式错误，请使用POST请求！"
        })
    image = cv.cvtColor(get_image_array(request), cv.COLOR_BGR2RGB)
    image_name = f"{uuid.uuid4()}.jpg"
    result_name = f"result_{image_name}"
    image_save_path = os.path.join(settings.BASE_DIR, "myapp", "static", "imgs", image_name)
    result_save_path = os.path.join(settings.BASE_DIR, "myapp", "static", "imgs", result_name)
    image_url = f"/static/imgs/{image_name}"
    result_url = f"/static/imgs/{result_name}"
    cv.imwrite(image_save_path, image)
    results = model(image_save_path)
    if len(results[0].boxes) == 0:
        return JsonResponse({
            "status": 500,
            "msg": no_target_msg,
            "prossed_image_url": image_url,
        })
    else:
        result_image = results[0].plot()
        cv.imwrite(result_save_path, result_image)
        return JsonResponse({
            "status": 200,
            "msg": success_msg,
            "prossed_image_url": result_url,
        })


def ccpd_detect(request):
    """车牌检测(车牌是单类,这里复用 _run_yolo,但模型内部仍是 1 类目标)。"""
    return _run_yolo(request, ccpd_model, "没有检测到车牌！", "车牌检测成功！")


def bone_detect(request):
    """骨折检测(10 类:肘/指/前臂/肱骨/肩/腕/...)。"""
    return _run_yolo(request, bone_model, "没有检测到骨折！", "骨折检测成功！")


def bottlecap_detect(request):
    """瓶盖检测。"""
    return _run_yolo(request, bottlecap_model, "没有检测到瓶盖！", "瓶盖检测成功！")


if __name__ == "__main__":
    select_info(80)
