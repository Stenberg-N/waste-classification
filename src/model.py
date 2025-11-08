import timm

def get_model(model_name='mobilenetv4_hybrid_medium.e500_r224_in1k', num_classes=6, pretrained=True):
    model = timm.create_model(model_name, pretrained=pretrained, num_classes=num_classes)
    return model