from rest_framework import serializers
from apps.endpoints.models import Endpoint
from apps.endpoints.models import Algorithm
from apps.endpoints.models import Request

class EndpointSerializer(serializers.ModelSerializer):
    class Meta:
        model = Endpoint
        read_only_fields = ("id", "name")
        fields = read_only_fields

class AlgorithmSerializer(serializers.ModelSerializer):
    class Meta:
        model = Algorithm
        read_only_fields = ("id", "name", "code", "parent_endpoint")
        fields = read_only_fields

class RequestSerializer(serializers.ModelSerializer):
    class Meta:
        model = Request
        read_only_fields = ("id", "input_data", "full_response", "response", "parent_algorithm")
        fields = ("id", "input_data", "full_response", "response", "feedback", "parent_algorithm")