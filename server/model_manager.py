from models.cnn import CNN

class ModelManger:

    def __init__(self):
        self.model = CNN(num_classes=10)

    def get_model(self):
        return self.model
    
    def get_weights(self):
        return self.model.state_dict()
    
    def set_weights(self,weights):
        self.model.load_state_dict(weights)