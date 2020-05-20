from django.shortcuts import render
from rest_framework import viewsets
from rest_framework import mixins

from apps.endpoints.models import Endpoint
from apps.endpoints.serializers import EndpointSerializer

from apps.endpoints.models import Algorithm
from apps.endpoints.serializers import AlgorithmSerializer

from apps.endpoints.models import Request
from apps.endpoints.serializers import RequestSerializer

class EndpointViewSet(mixins.RetrieveModelMixin, mixins.ListModelMixin, viewsets.GenericViewSet):
    serializer_class = EndpointSerializer
    queryset = Endpoint.objects.all()

class AlgorithmViewSet(mixins.RetrieveModelMixin, mixins.ListModelMixin, viewsets.GenericViewSet):
    serializer_class = AlgorithmSerializer
    queryset = Algorithm.objects.all()

class RequestViewSet(mixins.RetrieveModelMixin, mixins.ListModelMixin, viewsets.GenericViewSet, mixins.UpdateModelMixin):
    serializer_class = RequestSerializer
    queryset = Request.objects.all()

