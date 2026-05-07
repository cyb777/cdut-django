# CDUT 智能视觉检测平台

基于 Django 的多模型计算机视觉检测平台，集成 YOLO 目标检测与人脸识别功能。

## 功能

- **车牌检测** — 基于 CCPD 数据集的 YOLO 车牌检测模型
- **骨折检测** — YOLOv8s 骨折部位检测（7 类：肘部、手指、前臂、肱骨、肩部、腕部）
- **瓶盖检测** — YOLO 瓶盖目标检测
- **人脸识别** — 人脸注册、采集与 1:N 匹配识别
- **用户系统** — 登录/注册，每个用户关联人脸数据

## 技术栈

- 后端：Django 5 + PyMySQL
- 视觉：YOLOv8 (Ultralytics) + face_recognition + OpenCV
- 前端：HTML + Vue.js + jQuery
- 模型部署：ONNX Runtime 兼容

## 快速开始

```bash
# 1. 克隆仓库
git clone <repo-url>
cd cdut_django

# 2. 安装依赖
pip install -r requirements.txt

# 3. 创建 MySQL 数据库
mysql -u root -p -e "CREATE DATABASE cdut_face_collect CHARACTER SET utf8;"

# 4. 修改数据库配置
# 编辑 cdut_django/settings.py，修改 DATABASES 配置为本地 MySQL 连接信息

# 5. 建表（在 MySQL 中执行）
CREATE TABLE login (
    id INT AUTO_INCREMENT PRIMARY KEY,
    username VARCHAR(50) UNIQUE NOT NULL,
    password VARCHAR(100) NOT NULL,
    face_image_path VARCHAR(255)
);

# 6. 启动
python manage.py runserver
```

## 项目结构

```
cdut_django/
├── cdut_django/          # Django 项目配置
│   ├── settings.py       # 配置文件（数据库等）
│   ├── urls.py           # 路由配置
│   └── wsgi.py           # WSGI 入口
├── myapp/                # 主应用
│   ├── views.py          # 视图（检测/人脸/登录逻辑）
│   ├── models.py         # 数据模型
│   ├── templates/        # HTML 模板
│   ├── static/           # 静态资源（JS/CSS）
│   ├── util/             # 工具函数
│   ├── car.pt            # 车牌检测模型
│   ├── bone_best.pt      # 骨折检测模型
│   └── bottlecap.pt      # 瓶盖检测模型
├── manage.py
├── requirements.txt
└── README.md
```

## 模型

| 模型 | 文件 | 类别数 | 说明 |
|------|------|--------|------|
| 车牌检测 | `car.pt` | - | CCPD 数据集 |
| 骨折检测 | `bone_best.pt` | 7 | 骨折部位分类 |
| 瓶盖检测 | `bottlecap.pt` | - | 瓶盖目标检测 |

## 环境要求

- Python >= 3.9
- MySQL >= 5.7
- dlib / face_recognition（可能需要 C++ 编译环境）
