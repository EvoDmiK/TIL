from django.urls import path
from . import views

urlpatterns = [
    path('', views.main),
    path('translate', views.translate, name = 'translate'),
    path('object_detection', views.object_detection, name = 'object_detection'),
    path('segmentation', views.segmentation, name = 'segmentation'),
]