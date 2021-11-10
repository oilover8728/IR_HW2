from django import forms

class UploadDocumentForm(forms.Form):
        file_field = forms.FileField(label='',max_length=5,widget=forms.ClearableFileInput(attrs={'multiple': True}))
