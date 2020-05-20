from django.conf.urls import url, include
from rest_framework.routers import DefaultRouter

from apps.endpoints.views import EndpointViewSet
from apps.endpoints.views import AlgorithmViewSet
from apps.endpoints.views import RequestViewSet

router = DefaultRouter(trailing_slash=False)
router.register(r"endpoints", EndpointViewSet, basename="endpoints")
router.register(r"mlalgorithms", AlgorithmViewSet, basename="algorithms")
router.register(r"mlrequests", RequestViewSet, basename="requests")

urlpatterns = [
    url(r"^api/v1/", include(router.urls)),
]