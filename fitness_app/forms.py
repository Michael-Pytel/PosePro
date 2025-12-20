from django import forms

from fitness_app.models import PushupVideosModel

class PushupVideosForm(forms.ModelForm):
    class Meta:
        model = PushupVideosModel
        fields = ('name', 'video')