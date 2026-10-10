class RecognitionResult:

    def __init__(
        self,
        identity="Unknown",
        similarity=0.0,
        distance=999.0,
        recognized=False
    ):
        self.identity = identity
        self.similarity = similarity
        self.distance = distance
        self.recognized = recognized

