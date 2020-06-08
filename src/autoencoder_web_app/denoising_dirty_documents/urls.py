from django.urls import path

from . import views

urlpatterns = [
    path('', views.index, name='app1_index'),
    path('denoiseImage', views.denoiseImage, name='denoiseImage'),
    path('viewDb', views.viewDb, name='app1_viewDb'),
]
