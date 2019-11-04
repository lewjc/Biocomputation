class BinaryData:
    
    feature = None
    label = None

    def __init__(self, feature: list, label: str):
        self.feature = feature
        self.label = label  