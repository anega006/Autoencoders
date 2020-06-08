from django.urls import path

from . import views

urlpatterns = [
    path('', views.index, name='app3_index'),
    path('segmentImage', views.segmentImage, name='segmentImage'),
    path('viewDb', views.viewDb, name='app3_viewDb'),
]
