import timm

def get_model(model_name='densenet201', num_classes=6, pretrained=True):
    model = timm.create_model(model_name, pretrained=pretrained, num_classes=num_classes)
    return model