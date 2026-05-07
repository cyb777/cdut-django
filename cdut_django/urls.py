"""
URL configuration for cdut_django project.

The `urlpatterns` list routes URLs to views. For more information please see:
    https://docs.djangoproject.com/en/5.2/topics/http/urls/
Examples:
Function views
    1. Add an import:  from my_app import views
    2. Add a URL to urlpatterns:  path('', views.home, name='home')
Class-based views
    1. Add an import:  from other_app.views import Home
    2. Add a URL to urlpatterns:  path('', Home.as_view(), name='home')
Including another URLconf
    1. Import the include() function: from django.urls import include, path
    2. Add a URL to urlpatterns:  path('blog/', include('blog.urls'))
"""

from django.contrib import admin
from django.urls import path
from myapp import views
urlpatterns = [
    path("admin/", admin.site.urls),
    path("index/",views.index, name="index.html"),
    path("",views.main_page,name="main_page"),
    path("face_page/",views.face_page,name="face_page"),
    path("face_collect/",views.face_collect),
    path("match_page/",views.match_page),
    path("face_match/",views.face_match,name="match_page"),
    path("detect_page/",views.detection_page),
    path("detection_detect/",views.ccpd_detect),
    path("login/",views.login),
    path("login_message_collect/",views.login_message_collect),
    path("main_page/",views.main_page,name="main_page"),
    path("register_page/",views.register_page,name="register_page"),
    path("bone_page/",views.bone,name="bone"),
    path("bone_detect/",views.bone_detect,name="bone_detect"),
    path("bottlecap_page/",views.bottlecap,name="bottlecap"),
    path("bottlecap_detect/",views.bottlecap_detect,name="bottlecap_detect")

]
