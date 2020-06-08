from django.urls import path

from . import views

urlpatterns = [
    path('', views.index, name='app2_index'),
    path('similarImage', views.similarImage, name='similarImage'),
    path('viewDb', views.viewDb, name='app2_viewDb'),
]
