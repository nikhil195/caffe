from django.conf.urls import url
from django.views.generic import TemplateView

from . import views

urlpatterns = [
    url(r'^$', views.index, name='index'),
    url(r'^uploadImage/', views.uploadImage, name='uploadImage'),
    url(r'^submitUrl/', views.submitUrl, name='submitUrl'),
]
