from django.db import models

# Create your models here.
class Endpoint(models.Model):
    name = models.CharField(max_length=128)

class Algorithm(models.Model):
    name = models.CharField(max_length=128)
    code = models.CharField(max_length=50000)
    parent_endpoint = models.ForeignKey(Endpoint, on_delete=models.CASCADE)

class Request(models.Model):
    input_data = models.CharField(max_length=10000)
    response = models.CharField(max_length=10000)
    full_response = models.CharField(max_length=10000)
    feedback = models.CharField(max_length=10000)
    parent_algorithm = models.ForeignKey(Algorithm, on_delete=models.CASCADE)
