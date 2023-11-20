from django.shortcuts import render

# Create your views here.
def main(request):

    return render(request, 'main.html', context = {})


def translate(request):

    return render(request, 'translate.html', context = {})


def object_detection(request):

    return render(request, 'object_detection.html', context = {})


def segmentation(request):

    return render(request, 'segmentation.html', context = {})