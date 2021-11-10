from django.db import models
import ast

class ListField(models.TextField):

    def __init__(self, *args, **kwargs):
        super(ListField, self).__init__(*args, **kwargs)

    def from_db_value(self, value, expression, connection):
        if not value:
            value = []

        if isinstance(value, list):
            return value

        return ast.literal_eval(value)

    def get_prep_value(self, value):
        if value is None:
            return value

        return str(value)

    def value_to_string(self, obj):
        value = self._get_val_from_obj(obj)
        return self.get_db_prep_value(value)

# Create your models here.
class Article(models.Model):
    index = models.IntegerField(default=0)
    title = models.TextField(default="")
    abstract = models.TextField(default="")

class Inverted_index(models.Model):
    word = models.TextField(default="")
    content_index = ListField()